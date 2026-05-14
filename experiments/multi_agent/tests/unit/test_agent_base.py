import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput, AgentOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


class _EchoOutput(BaseModel):
    answer: str


class _DummyAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "you are a test agent"

    def output_schema(self):
        return _EchoOutput


@pytest.mark.asyncio
async def test_agent_construction(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text='{"answer": "hi"}')])
    agent = _DummyAgent(
        name="dummy", role="test",
        provider=p, recorder=rec,
    )
    rec.close()
    assert agent.name == "dummy"
    assert agent.max_steps == 10  # default
