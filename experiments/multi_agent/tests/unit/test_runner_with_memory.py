import json
import pytest
from datetime import datetime
from pydantic import BaseModel

from multi_agent.runner import run_query
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import StickyContext


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_run_query_appends_turn_when_session_id_given(tmp_path):
    runs_root = tmp_path / "runs"
    store = MarkdownMemoryStore(root=tmp_path / "memory_store")
    store.write_sticky(StickyContext(session_id="s_test"))

    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "hi"}'),
    ])
    result = await run_query(
        query="hello?",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=runs_root,
        config={},
        session_id="s_test",
        memory_store=store,
    )
    assert result["status"] == "ok"
    sticky = store.read_sticky("s_test")
    assert sticky is not None
    assert result["run_id"] in sticky.linked_runs
    turns = store.recent_turns("s_test", n=5)
    assert len(turns) == 1
    assert turns[0].question == "hello?"
    assert turns[0].run_id == result["run_id"]


@pytest.mark.asyncio
async def test_run_query_works_without_session_id(tmp_path):
    """Backward-compat: existing callers don't pass session_id; should still work."""
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "x"}')])
    result = await run_query(
        query="hi",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider, runs_root=runs_root, config={},
    )
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_query_creates_session_if_missing(tmp_path):
    """If session_id given but no sticky exists, create one."""
    runs_root = tmp_path / "runs"
    store = MarkdownMemoryStore(root=tmp_path / "memory_store")
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "y"}')])
    result = await run_query(
        query="new session",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider, runs_root=runs_root, config={},
        session_id="s_new",
        memory_store=store,
    )
    assert result["status"] == "ok"
    sticky = store.read_sticky("s_new")
    assert sticky is not None
    assert sticky.session_id == "s_new"
