import json
import pytest
from multi_agent.runner import run_query
from multi_agent.agents.stub_echo import EchoStubAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_run_query_writes_meta_and_run_finished(tmp_path):
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hi back"}'),
    ])
    result = await run_query(
        query="hi",
        agent_factory=lambda provider, recorder: EchoStubAgent(
            name="echo", role="stub", provider=provider, recorder=recorder,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub"},
    )
    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "events.db").exists()

    lines = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [l["event_type"] for l in lines]
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert lines[-1]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_query_emits_run_finished_on_exception(tmp_path):
    """Critical invariant: even when agent raises, events.jsonl ends with RunFinished(status='error')."""
    runs_root = tmp_path / "runs"

    class _BoomAgent(EchoStubAgent):
        async def run(self, input):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_query(
            query="hi",
            agent_factory=lambda provider, recorder: _BoomAgent(
                name="boom", role="t", provider=provider, recorder=recorder,
            ),
            provider=StubProvider(responses=[]),
            runs_root=runs_root,
            config={"profile": "stub"},
        )

    # Find the produced run dir
    run_dirs = list(runs_root.glob("r_*"))
    assert len(run_dirs) == 1
    lines = [json.loads(l) for l in (run_dirs[0] / "events.jsonl").read_text().splitlines()]
    assert lines[-1]["event_type"] == "RunFinished"
    assert lines[-1]["status"] == "error"
    assert "boom" in lines[-1]["error"]
