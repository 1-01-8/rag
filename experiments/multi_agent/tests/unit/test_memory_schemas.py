import pytest
from datetime import datetime
from multi_agent.schemas.memory import (
    EntityState, ActiveSubject, KeyFact, RejectedPath,
    StickyContext, Turn, AgentNote,
)


def test_entity_state_minimal():
    es = EntityState()
    assert es.active_subjects == []
    assert es.key_facts == []
    assert es.open_questions == []
    assert es.rejected_paths == []
    assert es.legal_objectives == []


def test_entity_state_full():
    es = EntityState(
        active_subjects=[
            ActiveSubject(role="原告", identifier="用户", attributes=["房屋承租人"]),
        ],
        key_facts=[KeyFact(fact="租期1年", confidence="high", source_turn=1)],
        rejected_paths=[RejectedPath(path="走刑事路径", reason="未涉及胁迫")],
    )
    assert es.active_subjects[0].role == "原告"
    assert es.key_facts[0].fact == "租期1年"


def test_sticky_context_required_fields():
    s = StickyContext(
        session_id="s_test_2026-05-14",
        legal_domain="民事",
        case_type="租赁纠纷",
        last_law_name="民法典",
    )
    assert s.session_id.startswith("s_")
    assert s.mentioned_laws == []
    assert s.cited_articles == []
    assert s.linked_runs == []
    assert isinstance(s.entity_state, EntityState)


def test_turn_record():
    t = Turn(
        turn=1, run_id="r_abc",
        started_at=datetime(2026, 5, 14, 14, 0),
        finished_at=datetime(2026, 5, 14, 14, 1),
        question="房东涨租?", final_answer='{"answer": "..."}',
        answer_mode="evidence_grounded",
        agents_invoked=["receptionist", "lawyer"],
    )
    assert t.duration_ms == 60000


def test_agent_note():
    n = AgentNote(
        name="lawyer-misses-rental-mgmt-rules",
        description="涨租漏引租赁管理办法",
        produced_by="supervisor", about_agent="lawyer",
        tags=["涨租", "民法典-510"], triggered_by_run="r_abc",
    )
    assert n.usage_count == 0
    assert "涨租" in n.tags
