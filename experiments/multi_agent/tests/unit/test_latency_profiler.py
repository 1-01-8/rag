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
