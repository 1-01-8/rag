"""Tests for ExperimentRunner optional judges integration (Phase 5c Task 4)."""
import json
from pathlib import Path

import pytest

from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query
from multi_agent.eval.judges.groundedness import GroundednessJudge
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_runner_attaches_judge_results(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="a", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        '{"event_id":"1","event_type":"RunStarted","timestamp":"2026-05-15T00:00:00","run_id":"x","parent_id":null}\n'
        '{"event_id":"2","event_type":"RunFinished","timestamp":"2026-05-15T00:00:02","run_id":"x","parent_id":"1"}\n'
    )

    async def runner(q):
        return {"run_id": "run-a", "status": "ok",
                "lawyer_output": {"primary_answer": "answer"}, "run_dir": run_dir}

    judge_provider = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.9, "ungrounded_claims": [], "rationale": "ok"}',
                         finish_reason="end_turn"),
    ])
    judges = [GroundednessJudge(provider=judge_provider, model="stub")]

    exp = ExperimentRunner(
        query_set=qs, run_group_name="g", runs_root=tmp_path,
        query_runner=runner, judges=judges,
    )
    group = await exp.run()
    rows = [json.loads(l) for l in (group.group_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert "judges" in rows[0]
    assert rows[0]["judges"]["groundedness"]["score"] == 0.9
