import pytest
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.providers.stub import StubProvider
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_receptionist_render_with_sticky_context(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="r", role="t", provider=p, recorder=rec)

    input = AgentInput(payload={
        "query": "那依据哪条法律?",
        "sticky_context": {
            "session_id": "s_x",
            "legal_domain": "民事",
            "case_type": "租赁纠纷",
            "last_law_name": "民法典",
            "mentioned_laws": ["民法典"],
            "entity_state": {
                "active_subjects": [{"role": "原告", "identifier": "用户", "attributes": []}],
                "key_facts": [{"fact": "租期1年", "confidence": "high", "source_turn": 1}],
            },
        },
    })
    rendered = agent._render_input(input)
    assert "那依据哪条法律?" in rendered
    assert "上一轮主题" in rendered
    assert "租赁纠纷" in rendered
    assert "民法典" in rendered
    assert "租期1年" in rendered
    rec.close()


def test_receptionist_render_without_sticky(tmp_path):
    """No sticky → behaves like before, just query."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="r", role="t", provider=p, recorder=rec)
    input = AgentInput(payload={"query": "房东涨租"})
    rendered = agent._render_input(input)
    assert rendered == "房东涨租"
    rec.close()
