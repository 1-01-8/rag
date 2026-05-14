import pytest
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_receptionist_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec)
    prompt = agent.system_prompt()
    assert "接待员" in prompt or "分诊员" in prompt
    assert "is_multi_issue" in prompt
    rec.close()


def test_receptionist_output_schema_is_correct(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec)
    assert agent.output_schema() is ReceptionistOutput
    rec.close()


@pytest.mark.asyncio
async def test_receptionist_runs_to_output(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"primary_specialty": "民事", "case_type": "租赁", "urgency": "中",'
                 ' "is_multi_issue": false, "sub_cases": [],'
                 ' "initial_facts": ["合同一年"], "normalized_query": "涨租",'
                 ' "need_clarification": false, "clarification_q": null, "risk_flag": null}',
            finish_reason="end_turn",
        ),
    ])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec, model="stub-1")
    out = await agent.run(AgentInput(payload={"query": "房东涨租"}))
    rec.close()
    assert isinstance(out.payload, ReceptionistOutput)
    assert out.payload.primary_specialty == "民事"
    assert out.payload.urgency == "中"
