import pytest
from datetime import datetime
from pathlib import Path
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import (
    StickyContext, EntityState, ActiveSubject, KeyFact,
    Turn, AgentNote,
)


@pytest.fixture
def store(tmp_path):
    return MarkdownMemoryStore(root=tmp_path / "memory_store")


def test_sticky_initially_absent(store):
    assert store.read_sticky("s_test") is None


def test_sticky_write_and_read_roundtrip(store):
    s = StickyContext(
        session_id="s_test_2026", legal_domain="民事", case_type="租赁",
        last_law_name="民法典", mentioned_laws=["民法典", "商品房屋租赁管理办法"],
        entity_state=EntityState(
            active_subjects=[ActiveSubject(role="原告", identifier="用户")],
            key_facts=[KeyFact(fact="租期1年", confidence="high", source_turn=1)],
        ),
        body="租房涨租案",
    )
    store.write_sticky(s)
    loaded = store.read_sticky("s_test_2026")
    assert loaded is not None
    assert loaded.legal_domain == "民事"
    assert loaded.mentioned_laws == ["民法典", "商品房屋租赁管理办法"]
    assert loaded.entity_state.active_subjects[0].role == "原告"
    assert loaded.body == "租房涨租案"


def test_append_turn_creates_numbered_file(store):
    s = StickyContext(session_id="s_x")
    store.write_sticky(s)
    t = Turn(
        turn=1, run_id="r_001",
        started_at=datetime(2026, 5, 14, 14, 0),
        finished_at=datetime(2026, 5, 14, 14, 1),
        question="房东涨租?", final_answer='{"answer": "..."}',
        agents_invoked=["lawyer"],
    )
    path = store.append_turn("s_x", t)
    assert path.exists()
    assert "001" in path.name


def test_recent_turns_sorted_descending(store):
    s = StickyContext(session_id="s_y")
    store.write_sticky(s)
    for i in range(1, 4):
        store.append_turn("s_y", Turn(
            turn=i, run_id=f"r_{i:03d}",
            started_at=datetime(2026, 5, 14, 14, 0),
            finished_at=datetime(2026, 5, 14, 14, i),
            question=f"q{i}", final_answer=f"a{i}",
        ))
    recent = store.recent_turns("s_y", n=2)
    assert len(recent) == 2
    assert recent[0].turn == 3
    assert recent[1].turn == 2


def test_agent_note_write_and_find(store):
    note = AgentNote(
        name="test-note", description="x",
        produced_by="supervisor", about_agent="lawyer",
        tags=["涨租", "民法典-510"],
        triggered_by_run="r_abc",
    )
    store.write_note(note)
    found = store.find_notes(tags=["涨租"])
    assert len(found) == 1
    assert found[0].name == "test-note"
    assert store.find_notes(tags=["unrelated_tag"]) == []


def test_index_regenerated_after_writes(store):
    s = StickyContext(session_id="s_z")
    store.write_sticky(s)
    store.append_turn("s_z", Turn(
        turn=1, run_id="r_x",
        started_at=datetime.now(), finished_at=datetime.now(),
        question="q", final_answer="a",
    ))
    import json as _j
    index = _j.loads((store.root / "_index.json").read_text(encoding="utf-8"))
    assert "s_z" in index["sessions"]
    assert index["sessions"]["s_z"]["turn_count"] == 1
