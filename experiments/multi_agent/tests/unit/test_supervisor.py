import pytest
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.schemas.supervisor import SupervisorVerdict
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from multi_agent.agents.base import AgentInput


def test_supervisor_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    prompt = agent.system_prompt()
    assert "审核员" in prompt or "Supervisor" in prompt
    assert "verify_citation" in prompt
    rec.close()


def test_supervisor_output_schema(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    assert agent.output_schema() is SupervisorVerdict
    rec.close()


@pytest.mark.asyncio
async def test_supervisor_renders_lawyer_output_in_prompt(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SupervisorAgent(name="supervisor", role="qa",
                           provider=p, recorder=rec)
    rendered = agent._render_input(AgentInput(payload={
        "user_query": "房东涨租合法吗?",
        "lawyer_output": {"mode": "consultation", "primary_answer": "不合法",
                         "citations": [{"law_short": "民法典", "article_no": "510",
                                        "excerpt": "合同补充"}]},
        "evidence_pool": [{"doc_id": "民法典-510", "law_name": "民法典",
                          "law_short": "民法典", "article_no": "510",
                          "text": "当事人就合同补充内容...", "score": 0.9,
                          "retriever": "hybrid"}],
    }))
    assert "房东涨租" in rendered
    assert "民法典" in rendered
    assert "510" in rendered
    rec.close()
