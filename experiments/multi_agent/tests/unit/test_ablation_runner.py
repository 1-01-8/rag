# tests/unit/test_ablation_runner.py
import asyncio
import json
import pytest
from pathlib import Path
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query
from multi_agent.eval.ablations import DisableTool, DisableMemory
from multi_agent.eval.ablation_runner import AblationRunner


def _fake_events(run_dir: Path, latency_ms: int = 1000):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        f'{{"event_id":"1","event_type":"RunStarted","timestamp":"2026-05-15T00:00:00","run_id":"x","parent_id":null}}\n'
        f'{{"event_id":"2","event_type":"RunFinished","timestamp":"2026-05-15T00:00:0{latency_ms//1000}","run_id":"x","parent_id":"1"}}\n'
    )


@pytest.mark.asyncio
async def test_ablation_runner_runs_baseline_plus_ablations(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="q1", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])

    seen_configs: list[dict] = []

    async def query_runner_factory(config: dict):
        async def runner(q):
            seen_configs.append(dict(config))
            run_dir = tmp_path / "runs" / f"{config.get('label','base')}-{q.id}"
            _fake_events(run_dir)
            return {"run_id": run_dir.name, "status": "ok",
                    "lawyer_output": {"citations": []}, "run_dir": run_dir}
        return runner

    ar = AblationRunner(
        query_set=qs,
        runs_root=tmp_path,
        query_runner_factory=query_runner_factory,
        run_group_base="ab-test",
    )
    report = await ar.run(ablations=[DisableTool(tool="case_search"), DisableMemory()])
    assert report.n_ablations == 2
    assert report.baseline.group_dir.exists()
    assert len(report.ablations) == 2
    # baseline + 2 ablations = 3 config invocations × 1 query = 3 seen
    assert len(seen_configs) == 3
    # ablation configs should differ from baseline
    assert any("disabled_tools" in c for c in seen_configs)
    assert any(c.get("disable_memory") is True for c in seen_configs)


@pytest.mark.asyncio
async def test_ablation_report_writes_summary_md(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="q1", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])

    async def factory(config):
        async def runner(q):
            run_dir = tmp_path / "runs" / f"r-{q.id}-{id(config)}"
            _fake_events(run_dir)
            return {"run_id": run_dir.name, "status": "ok",
                    "lawyer_output": {}, "run_dir": run_dir}
        return runner

    ar = AblationRunner(query_set=qs, runs_root=tmp_path,
                        query_runner_factory=factory, run_group_base="ab")
    report = await ar.run(ablations=[DisableMemory()])
    summary_path = report.group_dir / "ablation_summary.md"
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "baseline" in md.lower()
    assert "disable_memory" in md
