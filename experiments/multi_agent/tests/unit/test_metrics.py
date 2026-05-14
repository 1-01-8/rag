"""Tests for derive_run_metrics (Phase 5b §7.6).

Uses the actual event format written by Recorder/JsonlEventWriter:
  - field: event_type (not "kind")
  - field: timestamp (ISO datetime string, not "ts_ms")
  - usage sub-fields: input_tokens, output_tokens, cache_read_tokens
    (no nested "data" wrapper — all fields are top-level)
  - RunFinished carries status + final_answer (not answer_mode)
  - ToolReturned has error field (not a separate ToolFailed event type)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from multi_agent.eval.metrics import derive_run_metrics, RunMetrics


def _ts(offset_ms: int) -> str:
    """Return an ISO-8601 UTC timestamp string offset by offset_ms from epoch."""
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t = base + timedelta(milliseconds=offset_ms)
    return t.isoformat().replace("+00:00", "Z")


def test_derive_metrics_from_synthetic_events(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"

    # Build events using the real field names from schemas/events.py
    evs = [
        {
            "event_id": "1",
            "event_type": "RunStarted",
            "run_id": "r1",
            "timestamp": _ts(1000),
            "parent_id": None,
            "query": "test query",
            "config": {},
        },
        {
            "event_id": "2",
            "event_type": "AgentInvoked",
            "run_id": "r1",
            "timestamp": _ts(1100),
            "parent_id": "1",
            "agent_name": "lawyer",
            "role": "advisor",
            "input": {},
        },
        {
            "event_id": "3",
            "event_type": "ToolCalled",
            "run_id": "r1",
            "timestamp": _ts(1200),
            "parent_id": "2",
            "tool_name": "statute_search",
            "args": {},
            "agent_name": "lawyer",
        },
        {
            "event_id": "4",
            "event_type": "LLMResponded",
            "run_id": "r1",
            "timestamp": _ts(1500),
            "parent_id": "2",
            "raw_response": "answer text",
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 250,
                "cache_read_tokens": 600,
                "cache_creation_tokens": 0,
            },
            "duration_ms": 300,
            "finish_reason": "end_turn",
        },
        {
            "event_id": "5",
            "event_type": "RunFinished",
            "run_id": "r1",
            "timestamp": _ts(5000),
            "parent_id": "1",
            "status": "ok",
            "final_answer": "evidence_grounded",
            "error": None,
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in evs),
        encoding="utf-8",
    )

    m = derive_run_metrics(run_dir)

    # Latency = timestamp of RunFinished - timestamp of RunStarted = 5000-1000 = 4000ms
    assert m.total_latency_ms == 4000
    assert m.total_input_tokens == 1200
    assert m.total_output_tokens == 250
    assert m.cache_read_tokens == 600
    assert m.cache_hit_rate == pytest.approx(0.5)
    assert m.agent_invocations == 1
    assert m.tool_calls_total == 1
    assert m.errors == 0


def test_derive_metrics_counts_errors(tmp_path):
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()

    evs = [
        {
            "event_id": "1",
            "event_type": "RunStarted",
            "run_id": "r2",
            "timestamp": _ts(1000),
            "parent_id": None,
            "query": "test",
            "config": {},
        },
        {
            # ToolReturned with error field set — the real "error" path in actual events
            "event_id": "2",
            "event_type": "ToolReturned",
            "run_id": "r2",
            "timestamp": _ts(1100),
            "parent_id": "1",
            "result": None,
            "error": "connection refused",
            "duration_ms": 100,
        },
        {
            "event_id": "3",
            "event_type": "RunFinished",
            "run_id": "r2",
            "timestamp": _ts(2000),
            "parent_id": "1",
            "status": "error",
            "final_answer": None,
            "error": "connection refused",
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in evs),
        encoding="utf-8",
    )

    m = derive_run_metrics(run_dir)
    assert m.errors == 1


def test_derive_metrics_missing_events_file(tmp_path):
    run_dir = tmp_path / "run-3"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        derive_run_metrics(run_dir)
