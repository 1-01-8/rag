# Phase 5e — LatencyProfiler Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Spec §7.10 — pure-derivation `LatencyProfiler` over an `events.jsonl` file. Produces `SpanTiming` tree + aggregates by agent / tool / provider / kind + CLI flame-graph (indented tree). No new schema additions to the trace.

**Phase 5d starting point:** Tag `phase5d-ablation`. 217 unit tests + 1 skipped.

---

## Event-pair model

Inspecting `multi_agent/schemas/events.py` and existing `events.jsonl`:

| Start event | End event | Has `duration_ms` | Span "kind" |
|---|---|---|---|
| `RunStarted` | `RunFinished` | derive from timestamps | `run` |
| `AgentInvoked` | `AgentResponded` | yes (on end) | `agent` |
| `LLMRequested` | `LLMResponded` | yes (on end) | `llm` |
| `ToolCalled` | `ToolReturned` | yes (on end) | `tool` |

End events have `parent_id == <start event>.event_id`. Start events have `parent_id == <enclosing start event>.event_id` (or null at the root). This gives us a span tree.

Standalone events (`MemoryRead`, `MemoryWritten`, `ReceptionistDecision`, `SupervisorVerdict`, etc.) are leaf events with no start/end pair — count them in `by_kind` but skip in the span tree.

---

## Out of scope (Phase 5f+)

- Streamlit trace viewer (Phase 5f)
- Statistical aggregation across runs (do it as a separate scriptlet in Phase 5g)
- Cost derivation (cost_usd from tokens — could fold in later)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   └── eval/
│       └── latency.py          # LatencyProfiler + SpanTiming + LatencyProfile + render
└── tests/
    └── unit/
        └── test_latency_profiler.py
```

Single file. Single test file. Single task.

---

## Task 1: LatencyProfiler

**Files:**
- Create: `multi_agent/eval/latency.py`
- Create: `tests/unit/test_latency_profiler.py`

### Step 1: Failing test

```python
# tests/unit/test_latency_profiler.py
import json
import pytest
from pathlib import Path
from multi_agent.eval.latency import (
    LatencyProfiler, SpanTiming, LatencyProfile,
)


def _ev(eid, etype, parent, ts, **extra):
    return {
        "event_id": eid, "run_id": "r1", "timestamp": ts,
        "parent_id": parent, "event_type": etype, **extra,
    }


def _write(run_dir, events):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
    )


def test_profile_builds_span_tree(tmp_path):
    run_dir = tmp_path / "run"
    events = [
        _ev("R0", "RunStarted", None, "2026-05-15T00:00:00", query="q"),
        _ev("A1", "AgentInvoked", "R0", "2026-05-15T00:00:00.100",
            agent_name="lawyer", role="advisor"),
        _ev("L1", "LLMRequested", "A1", "2026-05-15T00:00:00.200",
            provider="openai_compat", model="qwen3.5-9b", messages=[]),
        _ev("L1e", "LLMResponded", "L1", "2026-05-15T00:00:01.700",
            raw_response="", usage={}, duration_ms=1500, finish_reason="end_turn"),
        _ev("T1", "ToolCalled", "A1", "2026-05-15T00:00:01.800",
            tool_name="statute_search", agent_name="lawyer", args={}),
        _ev("T1e", "ToolReturned", "T1", "2026-05-15T00:00:02.300",
            duration_ms=500, result={}),
        _ev("A1e", "AgentResponded", "A1", "2026-05-15T00:00:02.500",
            agent_name="lawyer", duration_ms=2400, output={}),
        _ev("R0e", "RunFinished", "R0", "2026-05-15T00:00:02.600",
            status="ok"),
    ]
    _write(run_dir, events)
    p = LatencyProfiler().profile(run_dir)
    assert isinstance(p, LatencyProfile)
    assert p.run_id == "r1"
    assert 2500 <= p.total_ms <= 2700
    # Top-level span = run; one child = lawyer agent
    assert p.spans.kind == "run"
    assert len(p.spans.children) == 1
    agent_span = p.spans.children[0]
    assert agent_span.kind == "agent"
    assert agent_span.label == "lawyer"
    assert agent_span.inclusive_ms == 2400  # from AgentResponded.duration_ms
    # Agent has 2 children: 1 llm (1500ms) + 1 tool (500ms)
    kinds = sorted(c.kind for c in agent_span.children)
    assert kinds == ["llm", "tool"]
    # Exclusive ms = inclusive - sum(children inclusive)
    assert agent_span.exclusive_ms == 2400 - 1500 - 500


def test_profile_by_agent_tool_provider_kind(tmp_path):
    run_dir = tmp_path / "run"
    events = [
        _ev("R0", "RunStarted", None, "2026-05-15T00:00:00", query="q"),
        _ev("A1", "AgentInvoked", "R0", "2026-05-15T00:00:00",
            agent_name="lawyer", role="advisor"),
        _ev("L1", "LLMRequested", "A1", "2026-05-15T00:00:00",
            provider="openai_compat", model="qwen3.5-9b", messages=[]),
        _ev("L1e", "LLMResponded", "L1", "2026-05-15T00:00:01",
            raw_response="", usage={}, duration_ms=1000, finish_reason="end_turn"),
        _ev("T1", "ToolCalled", "A1", "2026-05-15T00:00:01",
            tool_name="statute_search", agent_name="lawyer", args={}),
        _ev("T1e", "ToolReturned", "T1", "2026-05-15T00:00:01.300",
            duration_ms=300, result={}),
        _ev("A1e", "AgentResponded", "A1", "2026-05-15T00:00:01.500",
            agent_name="lawyer", duration_ms=1500, output={}),
        _ev("R0e", "RunFinished", "R0", "2026-05-15T00:00:01.600",
            status="ok"),
    ]
    _write(run_dir, events)
    p = LatencyProfiler().profile(run_dir)
    assert p.by_agent["lawyer"] == 1500
    assert p.by_tool["statute_search"] == 300
    assert p.by_provider["openai_compat"] == 1000
    # by_kind counts ALL events, not just paired spans
    assert p.by_kind["RunStarted"] == 1
    assert p.by_kind["LLMRequested"] == 1


def test_render_flame_indented_tree(tmp_path):
    run_dir = tmp_path / "run"
    events = [
        _ev("R0", "RunStarted", None, "2026-05-15T00:00:00", query="q"),
        _ev("A1", "AgentInvoked", "R0", "2026-05-15T00:00:00",
            agent_name="lawyer", role="advisor"),
        _ev("A1e", "AgentResponded", "A1", "2026-05-15T00:00:01",
            agent_name="lawyer", duration_ms=1000, output={}),
        _ev("R0e", "RunFinished", "R0", "2026-05-15T00:00:01.100",
            status="ok"),
    ]
    _write(run_dir, events)
    p = LatencyProfiler().profile(run_dir)
    text = LatencyProfiler.render_flame(p)
    assert "run" in text
    assert "lawyer" in text
    assert "1000" in text or "1.0" in text
    # Indented child line
    assert "  " in text  # has at least one indented row


def test_unpaired_events_handled_gracefully(tmp_path):
    """A start event without a matching end shouldn't crash."""
    run_dir = tmp_path / "run"
    events = [
        _ev("R0", "RunStarted", None, "2026-05-15T00:00:00", query="q"),
        _ev("A1", "AgentInvoked", "R0", "2026-05-15T00:00:00",
            agent_name="lawyer", role="advisor"),
        # NO AgentResponded
        _ev("R0e", "RunFinished", "R0", "2026-05-15T00:00:01", status="error"),
    ]
    _write(run_dir, events)
    p = LatencyProfiler().profile(run_dir)
    # Lawyer span should still appear with inclusive_ms derived from timestamps or 0
    agent_span = p.spans.children[0]
    assert agent_span.kind == "agent"
    assert agent_span.inclusive_ms >= 0
```

### Step 2: Implement

```python
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


# Pairing: start event_type -> end event_type
_PAIRS = {
    "RunStarted": ("RunFinished", "run"),
    "AgentInvoked": ("AgentResponded", "agent"),
    "LLMRequested": ("LLMResponded", "llm"),
    "ToolCalled": ("ToolReturned", "tool"),
}


def _parse_ts(s: str) -> datetime:
    # Allow trailing Z
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s.endswith("Z") else datetime.fromisoformat(s)


class LatencyProfiler:
    def profile(self, run_dir: Path) -> LatencyProfile:
        events_path = Path(run_dir) / "events.jsonl"
        events = [json.loads(l) for l in events_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_id = {e["event_id"]: e for e in events}
        # end-by-parent: parent_id -> end event
        end_by_parent: dict[str, dict] = {
            e["parent_id"]: e for e in events
            if e["event_type"] in {p[0] for p in _PAIRS.values()} and e.get("parent_id")
        }

        by_kind: dict[str, int] = {}
        by_agent: dict[str, int] = {}
        by_tool: dict[str, int] = {}
        by_provider: dict[str, int] = {}
        for e in events:
            by_kind[e["event_type"]] = by_kind.get(e["event_type"], 0) + 1

        run_id = events[0]["run_id"] if events else ""

        # Build a child-map of start-events by their parent_id
        children_of: dict[str | None, list[dict]] = {}
        for e in events:
            if e["event_type"] in _PAIRS:
                children_of.setdefault(e.get("parent_id"), []).append(e)

        def make_span(start_event: dict) -> SpanTiming:
            etype = start_event["event_type"]
            end_etype, kind = _PAIRS[etype]
            end_ev = end_by_parent.get(start_event["event_id"])

            # Label depends on kind
            if kind == "run":
                label = "run"
            elif kind == "agent":
                label = start_event.get("agent_name", "?")
                by_agent[label] = by_agent.get(label, 0) + (end_ev.get("duration_ms", 0) if end_ev else 0)
            elif kind == "tool":
                label = start_event.get("tool_name", "?")
                by_tool[label] = by_tool.get(label, 0) + (end_ev.get("duration_ms", 0) if end_ev else 0)
            elif kind == "llm":
                label = f"{start_event.get('provider','?')}:{start_event.get('model','?')}"
                prov = start_event.get("provider", "?")
                by_provider[prov] = by_provider.get(prov, 0) + (end_ev.get("duration_ms", 0) if end_ev else 0)
            else:
                label = etype

            # inclusive_ms
            if end_ev and "duration_ms" in end_ev:
                inc = int(end_ev["duration_ms"])
            elif end_ev:
                try:
                    inc = int((_parse_ts(end_ev["timestamp"]) - _parse_ts(start_event["timestamp"])).total_seconds() * 1000)
                except Exception:
                    inc = 0
            else:
                inc = 0

            kid_starts = children_of.get(start_event["event_id"], [])
            kid_spans = [make_span(k) for k in kid_starts]
            exc = inc - sum(c.inclusive_ms for c in kid_spans)
            return SpanTiming(
                span_id=start_event["event_id"], kind=kind, label=label,
                inclusive_ms=inc, exclusive_ms=max(0, exc),
                children=kid_spans,
                metadata={k: v for k, v in start_event.items()
                         if k not in {"event_id", "run_id", "timestamp", "parent_id", "event_type", "messages"}},
            )

        roots = children_of.get(None, [])
        if not roots:
            # synthesise an empty root
            spans = SpanTiming(span_id="", kind="run", label="run", inclusive_ms=0)
        else:
            spans = make_span(roots[0])

        return LatencyProfile(
            run_id=run_id, total_ms=spans.inclusive_ms, spans=spans,
            by_agent=by_agent, by_tool=by_tool, by_provider=by_provider, by_kind=by_kind,
        )

    @staticmethod
    def render_flame(profile: LatencyProfile) -> str:
        lines: list[str] = []
        def walk(s: SpanTiming, depth: int = 0):
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
```

### Step 3: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_latency_profiler.py -v"
```

Expected: 4 tests pass.

### Step 4: Commit + tag

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/latency.py experiments/multi_agent/tests/unit/test_latency_profiler.py
git commit -m "phase5e(eval): LatencyProfiler — span tree + by-agent/tool/provider aggregates + flame"
git tag -a phase5e-latency -m "Phase 5e: LatencyProfiler (derived span tree, no schema changes)"
git tag -l "phase*"
```

---

## Acceptance Criteria

1. 4 unit tests pass (span tree, aggregates, flame render, unpaired-event resilience)
2. `LatencyProfiler.profile(run_dir)` derives from `events.jsonl` alone — no new schema fields
3. `render_flame` produces a readable indented tree
4. Tag `phase5e-latency` exists

## Out of Scope (Phase 5f+)

- Streamlit Trace Viewer
- Cross-run aggregation (p50/p95 bottleneck by agent/tool across a RunGroup)
- Per-event cost USD derivation
