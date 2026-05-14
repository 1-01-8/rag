import pytest
from multi_agent.agents.secretary import (
    SecretaryAgent, SecretaryResponse, SecretaryAsTool, SecretaryRequest,
)
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


def test_secretary_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SecretaryAgent(name="secretary", role="research",
                          provider=p, recorder=rec, model="stub-1")
    prompt = agent.system_prompt()
    assert "秘书" in prompt
    assert "statute_search" in prompt
    rec.close()


def test_secretary_output_schema(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SecretaryAgent(name="secretary", role="research",
                          provider=p, recorder=rec)
    assert agent.output_schema() is SecretaryResponse
    rec.close()


@pytest.mark.asyncio
async def test_secretary_as_tool_dispatches_to_agent(tmp_path):
    """SecretaryAsTool should run the wrapped SecretaryAgent and return its output."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"summary": "found Article 510", "evidences": [], "notes": "", "confidence": 0.9}',
            finish_reason="end_turn",
        ),
    ])
    secretary = SecretaryAgent(name="secretary", role="research",
                              provider=p, recorder=rec, model="stub-1",
                              max_pre_tool_rejections=10)
    tool = SecretaryAsTool(secretary_agent=secretary)
    result = await tool.call(
        SecretaryRequest(task="search", payload={"query": "民法典 510"}),
        rec,
    )
    rec.close()
    assert result.error is None
    assert result.payload["summary"] == "found Article 510"
