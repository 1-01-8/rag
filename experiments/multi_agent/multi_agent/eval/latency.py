"""LatencyProfiler — derive span tree + aggregates from events.jsonl (Phase 5e §7.10)."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field


class SpanTiming(BaseModel):
    span_id: str
    kind: str                  # "run" | "agent" | "llm" | "tool"
    label: str
    inclusive_ms: int = 0
    exclusive_ms: int = 0
    children: list["SpanTiming"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LatencyProfile(BaseModel):
    run_id: str
    total_ms: int
    spans: SpanTiming
    by_agent: dict[str, int] = Field(default_factory=dict)
    by_tool: dict[str, int] = Field(default_factory=dict)
    by_provider: dict[str, int] = Field(default_factory=dict)
    by_kind: dict[str, int] = Field(default_factory=dict)


# Pairing: start event_type -> (end event_type, span kind)
_PAIRS: dict[str, tuple[str, str]] = {
    "RunStarted": ("RunFinished", "run"),
    "AgentInvoked": ("AgentResponded", "agent"),
    "LLMRequested": ("LLMResponded", "llm"),
    "ToolCalled": ("ToolReturned", "tool"),
}


def _parse_ts(s: str) -> datetime:
    """Parse ISO-8601 timestamp, allowing trailing Z."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s.endswith("Z") else datetime.fromisoformat(s)


class LatencyProfiler:
    def profile(self, run_dir: Path) -> LatencyProfile:
        events_path = Path(run_dir) / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Index events by their id
        by_id: dict[str, dict] = {e["event_id"]: e for e in events}

        # Map: start event_id -> end event (for start events that have a matching end)
        _end_event_types = {v[0] for v in _PAIRS.values()}
        end_by_parent: dict[str, dict] = {
            e["parent_id"]: e
            for e in events
            if e["event_type"] in _end_event_types and e.get("parent_id")
        }

        # Count every event by type
        by_kind: dict[str, int] = {}
        for e in events:
            by_kind[e["event_type"]] = by_kind.get(e["event_type"], 0) + 1

        run_id = events[0]["run_id"] if events else ""

        # Build child-map: parent_id -> [start events]
        children_of: dict[str | None, list[dict]] = {}
        for e in events:
            if e["event_type"] in _PAIRS:
                children_of.setdefault(e.get("parent_id"), []).append(e)

        # Mutable aggregates captured by closure
        by_agent: dict[str, int] = {}
        by_tool: dict[str, int] = {}
        by_provider: dict[str, int] = {}

        def make_span(start_event: dict) -> SpanTiming:
            etype = start_event["event_type"]
            _end_etype, kind = _PAIRS[etype]
            end_ev = end_by_parent.get(start_event["event_id"])

            # Determine label and update aggregates
            if kind == "run":
                label = "run"
            elif kind == "agent":
                label = start_event.get("agent_name", "?")
                dur = int(end_ev.get("duration_ms", 0)) if end_ev else 0
                by_agent[label] = by_agent.get(label, 0) + dur
            elif kind == "tool":
                label = start_event.get("tool_name", "?")
                dur = int(end_ev.get("duration_ms", 0)) if end_ev else 0
                by_tool[label] = by_tool.get(label, 0) + dur
            elif kind == "llm":
                label = f"{start_event.get('provider', '?')}:{start_event.get('model', '?')}"
                prov = start_event.get("provider", "?")
                dur = int(end_ev.get("duration_ms", 0)) if end_ev else 0
                by_provider[prov] = by_provider.get(prov, 0) + dur
            else:
                label = etype

            # Compute inclusive_ms: prefer explicit duration_ms, fall back to timestamp diff
            if end_ev and "duration_ms" in end_ev:
                inc = int(end_ev["duration_ms"])
            elif end_ev:
                try:
                    delta = _parse_ts(end_ev["timestamp"]) - _parse_ts(start_event["timestamp"])
                    inc = int(delta.total_seconds() * 1000)
                except Exception:
                    inc = 0
            else:
                inc = 0

            # Recurse into children
            kid_starts = children_of.get(start_event["event_id"], [])
            kid_spans = [make_span(k) for k in kid_starts]
            exc = inc - sum(c.inclusive_ms for c in kid_spans)

            return SpanTiming(
                span_id=start_event["event_id"],
                kind=kind,
                label=label,
                inclusive_ms=inc,
                exclusive_ms=max(0, exc),
                children=kid_spans,
                metadata={
                    k: v for k, v in start_event.items()
                    if k not in {"event_id", "run_id", "timestamp", "parent_id", "event_type", "messages"}
                },
            )

        roots = children_of.get(None, [])
        if not roots:
            spans = SpanTiming(span_id="", kind="run", label="run", inclusive_ms=0)
        else:
            spans = make_span(roots[0])

        return LatencyProfile(
            run_id=run_id,
            total_ms=spans.inclusive_ms,
            spans=spans,
            by_agent=by_agent,
            by_tool=by_tool,
            by_provider=by_provider,
            by_kind=by_kind,
        )

    @staticmethod
    def render_flame(profile: LatencyProfile) -> str:
        lines: list[str] = []

        def walk(s: SpanTiming, depth: int = 0) -> None:
            pad = "  " * depth
            lines.append(f"{pad}{s.kind}:{s.label}  inc={s.inclusive_ms}ms exc={s.exclusive_ms}ms")
            for c in s.children:
                walk(c, depth + 1)

        walk(profile.spans)

        if profile.by_agent:
            lines.append("")
            lines.append("by_agent: " + ", ".join(f"{k}={v}ms" for k, v in profile.by_agent.items()))
        if profile.by_tool:
            lines.append("by_tool: " + ", ".join(f"{k}={v}ms" for k, v in profile.by_tool.items()))
        if profile.by_provider:
            lines.append("by_provider: " + ", ".join(f"{k}={v}ms" for k, v in profile.by_provider.items()))

        return "\n".join(lines)
