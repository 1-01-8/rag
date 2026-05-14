import pytest
from multi_agent.agents.stub_echo import EchoStubAgent, EchoStubOutput
from multi_agent.agents.base import AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_echo_stub_runs_end_to_end(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hello back"}'),
    ])
    agent = EchoStubAgent(name="echo", role="stub",
                          provider=provider, recorder=rec)
    out = await agent.run(AgentInput(payload={"query": "hello"}))
    rec.close()
    assert isinstance(out.payload, EchoStubOutput)
    assert out.payload.echoed == "hello back"


def test_echo_stub_system_prompt_mentions_role(tmp_path):
    from multi_agent.providers.stub import StubProvider
    p = StubProvider(responses=[])
    rec = Recorder(run_id="r-prompt-test", run_dir=tmp_path / "runs" / "x")
    a = EchoStubAgent(name="echo", role="stub", provider=p, recorder=rec)
    assert "echo" in a.system_prompt().lower()
    rec.close()
