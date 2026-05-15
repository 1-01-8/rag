"""Tests for scripts/profile_run.py CLI."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "profile_run.py"


def _events_jsonl(events: list[dict]) -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in events)


def _seed(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"event_id": "R0", "run_id": "r1", "timestamp": "2026-05-15T00:00:00",
         "parent_id": None, "event_type": "RunStarted", "query": "q", "config": {}},
        {"event_id": "A1", "run_id": "r1", "timestamp": "2026-05-15T00:00:00.100",
         "parent_id": "R0", "event_type": "AgentInvoked",
         "agent_name": "lawyer", "role": "advisor"},
        {"event_id": "A1e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01",
         "parent_id": "A1", "event_type": "AgentResponded",
         "agent_name": "lawyer", "duration_ms": 900, "output": {}},
        {"event_id": "R0e", "run_id": "r1", "timestamp": "2026-05-15T00:00:01.100",
         "parent_id": "R0", "event_type": "RunFinished",
         "status": "ok", "final_answer": None, "error": None},
    ]
    (run_dir / "events.jsonl").write_text(_events_jsonl(events), encoding="utf-8")


def test_profile_run_flame_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    _seed(run_dir)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True, text=True, check=True,
    )
    assert "run_id=r1" in out.stdout
    assert "lawyer" in out.stdout
    assert "agent:lawyer" in out.stdout


def test_profile_run_json_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    _seed(run_dir)
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    assert data["run_id"] == "r1"
    assert data["spans"]["children"][0]["kind"] == "agent"


def test_profile_run_missing_events_exits_nonzero(tmp_path: Path) -> None:
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir)],
        capture_output=True, text=True,
    )
    assert out.returncode != 0
    assert "events.jsonl" in out.stderr
