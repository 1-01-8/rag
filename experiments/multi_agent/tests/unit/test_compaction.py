"""Tests for cross-turn compaction (spec §5.4.1)."""
import pytest
from datetime import datetime, timezone
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.memory.compaction import maybe_compact, COMPACTION_THRESHOLD
from multi_agent.schemas.memory import StickyContext, Turn
from multi_agent.providers.stub import StubProvider, ScriptedResponse


def _new_turn(n: int) -> Turn:
    return Turn(
        turn=n,
        run_id=f"r{n}",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        question=f"Q{n}",
        final_answer=f"A{n}",
        agents_invoked=["lawyer"],
    )


@pytest.mark.asyncio
async def test_no_compaction_when_below_threshold(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_short"
    store.write_sticky(StickyContext(session_id=sid,
                                     created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, COMPACTION_THRESHOLD):    # 4 turns < threshold 5
        store.append_turn(sid, _new_turn(i))

    p = StubProvider(responses=[])    # would error if called
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is False
    sticky = store.read_sticky(sid)
    assert sticky is not None
    assert sticky.history_summary == ""


@pytest.mark.asyncio
async def test_compaction_summarizes_old_turns(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid,
                                     created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, 8):   # 7 turns > threshold 5
        store.append_turn(sid, _new_turn(i))

    p = StubProvider(responses=[
        ScriptedResponse(text="第 1-4 轮: 用户就 X 进行咨询,Lawyer 引用 Y 法条建议 Z。",
                         finish_reason="end_turn"),
    ])
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is True
    sticky = store.read_sticky(sid)
    assert sticky.history_summary.startswith("第 1-4 轮")


@pytest.mark.asyncio
async def test_compaction_keeps_recent_turn_files(tmp_path):
    """Old turn files are NOT deleted — only summarized into sticky."""
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid,
                                     created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, 8):
        store.append_turn(sid, _new_turn(i))

    turn_files_before = sorted((tmp_path / "sessions" / sid / "turns").glob("*.md"))
    assert len(turn_files_before) == 7

    p = StubProvider(responses=[
        ScriptedResponse(text="summary", finish_reason="end_turn"),
    ])
    await maybe_compact(sid, store, provider=p, model="stub")

    turn_files_after = sorted((tmp_path / "sessions" / sid / "turns").glob("*.md"))
    assert len(turn_files_after) == 7    # nothing deleted


@pytest.mark.asyncio
async def test_idempotent_when_already_compacted(tmp_path):
    """Calling maybe_compact twice doesn't re-summarize (history_summary already set)."""
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid,
                                     created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc),
                                     history_summary="already compacted"))
    for i in range(1, 8):
        store.append_turn(sid, _new_turn(i))
    p = StubProvider(responses=[])   # would error if called
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is False
    sticky = store.read_sticky(sid)
    assert sticky.history_summary == "already compacted"
