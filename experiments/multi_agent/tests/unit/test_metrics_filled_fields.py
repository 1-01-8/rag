"""Phase 5q: react_steps_total / final_answer_mode / citation_count are populated."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from multi_agent.eval.metrics import derive_run_metrics


def _write(run_dir: Path, events: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events), encoding="utf-8"
    )


def test_final_answer_mode_extracted_from_run_finished(tmp_path):
    final_answer_payload = {
        "mode": "consultation",
        "primary_answer": "不合法",
        "citations": [
            {"law_short": "民法典", "article_no": "703", "excerpt": "..."},
            {"law_short": "民法典", "article_no": "510", "excerpt": "..."},
        ],
    }
    events = [
        {"event_id": "R0", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": None, "event_type": "RunStarted", "query": "q", "config": {}},
        {"event_id": "R0e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "R0", "event_type": "RunFinished",
         "status": "ok", "final_answer": json.dumps(final_answer_payload), "error": None},
    ]
    run_dir = tmp_path / "r"
    _write(run_dir, events)
    m = derive_run_metrics(run_dir)
    assert m.final_answer_mode == "consultation"
    assert m.citation_count == 2


def test_react_steps_counts_agent_responded(tmp_path):
    events = [
        {"event_id": "R0", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": None, "event_type": "RunStarted", "query": "q", "config": {}},
        {"event_id": "A1", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": "R0", "event_type": "AgentInvoked",
         "agent_name": "lawyer", "role": "advisor"},
        {"event_id": "A1e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "A1", "event_type": "AgentResponded",
         "agent_name": "lawyer", "duration_ms": 1000, "output": {}},
        {"event_id": "A2", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "R0", "event_type": "AgentInvoked",
         "agent_name": "secretary", "role": "assistant"},
        {"event_id": "A2e", "run_id": "r1", "timestamp": "2026-05-15T00:00:02",
         "parent_id": "A2", "event_type": "AgentResponded",
         "agent_name": "secretary", "duration_ms": 500, "output": {}},
        {"event_id": "R0e", "run_id": "r1", "timestamp": "2026-05-15T00:00:02",
         "parent_id": "R0", "event_type": "RunFinished",
         "status": "ok", "final_answer": None, "error": None},
    ]
    run_dir = tmp_path / "r"
    _write(run_dir, events)
    m = derive_run_metrics(run_dir)
    assert m.react_steps_total == 2


def test_malformed_final_answer_does_not_raise(tmp_path):
    events = [
        {"event_id": "R0", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": None, "event_type": "RunStarted", "query": "q", "config": {}},
        {"event_id": "R0e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "R0", "event_type": "RunFinished",
         "status": "ok", "final_answer": "not json at all", "error": None},
    ]
    run_dir = tmp_path / "r"
    _write(run_dir, events)
    m = derive_run_metrics(run_dir)
    assert m.final_answer_mode is None
    assert m.citation_count == 0
