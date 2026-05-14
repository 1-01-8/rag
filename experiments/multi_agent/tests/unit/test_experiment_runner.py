"""Tests for ExperimentRunner + RunGroup (Phase 5b Task 4).

Fixture event format matches the REAL recorder format used by derive_run_metrics:
  - event_type (not "kind")
  - timestamp as ISO-8601 string (not ts_ms int)
  - usage fields top-level inside "usage" dict: input_tokens, output_tokens, cache_read_tokens
  - no nested "data" wrapper
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from multi_agent.eval.runner import ExperimentRunner, RunGroup
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query


def _write_fake_events(run_dir: Path) -> None:
    """Write minimal valid events.jsonl so derive_run_metrics succeeds."""
    run_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "event_id": "1",
            "event_type": "RunStarted",
            "run_id": "x",
            "timestamp": "2026-01-01T00:00:01Z",
            "parent_id": None,
            "query": "test",
            "config": {},
        },
        {
            "event_id": "2",
            "event_type": "RunFinished",
            "run_id": "x",
            "timestamp": "2026-01-01T00:00:02Z",
            "parent_id": "1",
            "status": "ok",
            "final_answer": None,
            "error": None,
        },
    ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_experiment_runner_writes_results_jsonl(tmp_path):
    qs = QuerySet(
        meta=QuerySetMeta(name="t"),
        queries=[
            Query(id="a", text="qa", jurisdiction="CN", cause="c", source="s"),
            Query(id="b", text="qb", jurisdiction="CN", cause="c", source="s"),
        ],
    )

    async def fake_runner(q):
        run_dir = tmp_path / "runs" / f"run-{q.id}"
        _write_fake_events(run_dir)
        return {
            "run_id": f"run-{q.id}",
            "status": "ok",
            "lawyer_output": {"citations": [], "primary_answer": f"answer for {q.id}"},
            "run_dir": run_dir,
        }

    runner = ExperimentRunner(
        query_set=qs,
        run_group_name="test-group",
        runs_root=tmp_path,
        query_runner=fake_runner,
    )
    group = await runner.run()

    assert group.group_dir.exists()
    results_path = group.group_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {r["query_id"] for r in rows} == {"a", "b"}
    assert all(r["status"] == "ok" for r in rows)
    assert all("metrics" in r for r in rows)


@pytest.mark.asyncio
async def test_runner_records_failures(tmp_path):
    qs = QuerySet(
        meta=QuerySetMeta(name="t"),
        queries=[
            Query(id="x", text="boom", jurisdiction="CN", cause="c", source="s"),
        ],
    )

    async def bad_runner(q):
        raise RuntimeError("simulated provider failure")

    runner = ExperimentRunner(
        query_set=qs,
        run_group_name="g",
        runs_root=tmp_path,
        query_runner=bad_runner,
    )
    group = await runner.run()

    rows = [
        json.loads(line)
        for line in (group.group_dir / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert rows[0]["status"] == "error"
    assert "simulated" in rows[0]["error"]
