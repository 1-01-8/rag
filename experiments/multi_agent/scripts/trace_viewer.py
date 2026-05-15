"""Streamlit trace viewer (spec §7.11)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import streamlit as st

from multi_agent.eval.latency import LatencyProfiler


def _events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _wm_snapshot(run_dir: Path) -> dict | None:
    p = run_dir / "artifacts" / "working_memory.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _depth_by_id(events: list[dict]) -> dict[str, int]:
    by_id = {e["event_id"]: e for e in events}
    depth: dict[str, int] = {}

    def walk(eid: str) -> int:
        if eid in depth:
            return depth[eid]
        ev = by_id.get(eid)
        if ev is None or ev.get("parent_id") is None:
            depth[eid] = 0
            return 0
        depth[eid] = 1 + walk(ev["parent_id"])
        return depth[eid]

    for e in events:
        walk(e["event_id"])
    return depth


def render(run_dir: Path) -> None:
    st.set_page_config(page_title=f"Trace {run_dir.name}", layout="wide")
    st.title(f"Trace viewer — {run_dir.name}")
    events = _events(run_dir)
    if not events:
        st.error(f"No events.jsonl at {run_dir}")
        return

    depths = _depth_by_id(events)
    col_left, col_mid, col_right = st.columns([2, 3, 2])

    with col_left:
        st.subheader("Timeline")
        for e in events:
            ind = "·   " * depths.get(e["event_id"], 0)
            label = f"{ind}{e['event_type']}"
            if e["event_type"] == "AgentInvoked":
                label += f" ({e.get('agent_name', '?')})"
            elif e["event_type"] == "ToolCalled":
                label += f" ({e.get('tool_name', '?')})"
            elif e["event_type"] == "LLMRequested":
                label += f" ({e.get('model', '?')})"
            if st.button(label, key=e["event_id"]):
                st.session_state["selected"] = e["event_id"]

    with col_mid:
        st.subheader("Event detail")
        sel = st.session_state.get("selected") or (events[0]["event_id"] if events else None)
        if sel:
            ev = next((x for x in events if x["event_id"] == sel), None)
            if ev:
                st.json(ev)

    with col_right:
        st.subheader("Aggregates")
        try:
            profile = LatencyProfiler().profile(run_dir)
            st.metric("Total ms", profile.total_ms)
            if profile.by_agent:
                st.write("**by_agent:**", profile.by_agent)
            if profile.by_tool:
                st.write("**by_tool:**", profile.by_tool)
            if profile.by_provider:
                st.write("**by_provider:**", profile.by_provider)
        except Exception as e:
            st.warning(f"LatencyProfiler failed: {e}")

        wm = _wm_snapshot(run_dir)
        if wm:
            st.subheader("WorkingMemory")
            n_ev = len(wm.get("retrieved_evidence") or [])
            st.metric("Retrieved evidence", n_ev)
            if n_ev:
                with st.expander("Show evidence"):
                    st.json(wm["retrieved_evidence"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    # Streamlit may pass its own argv; only parse known args
    args, _ = parser.parse_known_args(sys.argv[1:])
    render(args.run_dir)


main()
