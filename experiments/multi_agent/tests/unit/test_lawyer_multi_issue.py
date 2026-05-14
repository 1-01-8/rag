import pytest
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.schemas.receptionist import SubCase
from multi_agent.providers.stub import StubProvider
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_lawyer_render_input_with_sub_cases(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                       provider=p, recorder=rec)
    input = AgentInput(payload={
        "query": "我要离婚也想申请保护令",
        "sub_cases": [
            SubCase(issue="离婚诉讼", specialty="家事", priority=2).model_dump(),
            SubCase(issue="人身保护令", specialty="治安", priority=1).model_dump(),
        ],
    })
    rendered = lawyer._render_input(input)
    assert "子议题" in rendered or "sub_cases" in rendered
    assert "离婚诉讼" in rendered
    assert "人身保护令" in rendered
    rec.close()


def test_lawyer_render_input_no_sub_cases(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                       provider=p, recorder=rec)
    input = AgentInput(payload={"query": "房东涨租"})
    rendered = lawyer._render_input(input)
    assert rendered == "房东涨租"
    rec.close()
