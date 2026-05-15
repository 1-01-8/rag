"""Streamlit AppTest smoke check — runs the app in-process without browser."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


SCRIPT = Path(__file__).parents[2] / "scripts" / "trace_viewer.py"


def _seed_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"event_id": "R0", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": None, "event_type": "RunStarted", "query": "q", "config": {}},
        {"event_id": "A1", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": "R0", "event_type": "AgentInvoked",
         "agent_name": "lawyer", "role": "advisor"},
        {"event_id": "A1e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "A1", "event_type": "AgentResponded",
         "agent_name": "lawyer", "duration_ms": 1000, "output": {}},
        {"event_id": "R0e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "R0", "event_type": "RunFinished",
         "status": "ok", "final_answer": None, "error": None},
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events), encoding="utf-8"
    )


def test_trace_viewer_smoke(tmp_path):
    run_dir = tmp_path / "r1"
    _seed_run(run_dir)
    # AppTest passes argv directly. Inject script-level args.
    saved = sys.argv
    sys.argv = ["trace_viewer.py", "--run-dir", str(run_dir)]
    try:
        at = AppTest.from_file(str(SCRIPT)).run(timeout=15)
    finally:
        sys.argv = saved
    assert not at.exception, str(at.exception)
    # The page title and subheaders should appear
    titles = [t.value for t in at.title]
    assert any("Trace viewer" in t for t in titles)
    subheaders = [s.value for s in at.subheader]
    assert any("Timeline" in s for s in subheaders)
    assert any("Event detail" in s for s in subheaders)
    assert any("Aggregates" in s for s in subheaders)


def test_trace_viewer_missing_run_dir_shows_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    saved = sys.argv
    sys.argv = ["trace_viewer.py", "--run-dir", str(empty)]
    try:
        at = AppTest.from_file(str(SCRIPT)).run(timeout=15)
    finally:
        sys.argv = saved
    assert not at.exception
    # Either an error widget or an empty timeline
    errors = [e.value for e in at.error]
    assert any("events.jsonl" in e for e in errors), errors
