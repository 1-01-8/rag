import pytest
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock
from multi_agent.runner import run_query
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


class _Out(BaseModel):
    answer: str


class _Lawyer(BaseAgent):
    def system_prompt(self) -> str: return "test"
    def output_schema(self): return _Out


@pytest.mark.asyncio
async def test_run_query_indexes_turn_when_indexer_supplied(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    indexer = MagicMock()
    indexer.index_turn = AsyncMock()
    result = await run_query(
        query="测试问题",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        session_id="s1",
        memory_store=store,
        turn_indexer=indexer,
    )
    assert result["status"] == "ok"
    indexer.index_turn.assert_awaited_once()
    call_kwargs = indexer.index_turn.await_args.kwargs
    assert call_kwargs["session_id"] == "s1"
    turn = call_kwargs["turn"]
    assert turn.question == "测试问题"
    assert turn.turn == 1


@pytest.mark.asyncio
async def test_run_query_does_not_index_without_indexer(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    # No indexer supplied — must not raise
    result = await run_query(
        query="测试",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        session_id="s1",
        memory_store=store,
        # turn_indexer omitted
    )
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_indexer_only_called_when_session_id_supplied(tmp_path):
    """If no session_id, no turn is appended → no indexing."""
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    indexer = MagicMock()
    indexer.index_turn = AsyncMock()
    await run_query(
        query="q",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        # session_id omitted, memory_store omitted
        turn_indexer=indexer,
    )
    indexer.index_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_compaction_called_when_provider_and_model_set(tmp_path, monkeypatch):
    """Fix 4: maybe_compact() is called when compaction_provider + compaction_model given."""
    from multi_agent import runner as runner_mod

    compact_calls = []

    async def _fake_compact(session_id, store, *, provider, model):
        compact_calls.append({"session_id": session_id, "model": model})
        return False

    monkeypatch.setattr("multi_agent.memory.compaction.maybe_compact", _fake_compact)

    store = MarkdownMemoryStore(root=tmp_path / "mem")
    main_provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    compact_provider = StubProvider(responses=[])

    result = await run_query(
        query="test compaction",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=main_provider,
        runs_root=tmp_path / "runs",
        session_id="s_compact",
        memory_store=store,
        compaction_provider=compact_provider,
        compaction_model="cheap-model",
    )
    assert result["status"] == "ok"
    assert len(compact_calls) == 1
    assert compact_calls[0]["session_id"] == "s_compact"
    assert compact_calls[0]["model"] == "cheap-model"


@pytest.mark.asyncio
async def test_compaction_skipped_when_no_provider(tmp_path, monkeypatch):
    """Fix 4: maybe_compact() is NOT called when compaction_provider is omitted."""
    compact_calls = []

    async def _fake_compact(session_id, store, *, provider, model):
        compact_calls.append(session_id)
        return False

    monkeypatch.setattr("multi_agent.memory.compaction.maybe_compact", _fake_compact)

    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])

    await run_query(
        query="no compact",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        session_id="s_no_compact",
        memory_store=store,
        # compaction_provider omitted
    )
    assert compact_calls == []
