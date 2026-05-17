"""Phase 6m: Turn 字段补全测试 — answer_mode / citations / total_tokens / agents."""
from __future__ import annotations
import pytest
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal

from multi_agent.runner import run_query
from multi_agent.agents.base import BaseAgent
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.providers.stub import StubProvider, ScriptedResponse


class _LawyerLikeOutput(BaseModel):
    mode: Literal["consultation", "clarification"] = "consultation"
    primary_answer: str = ""
    citations: list[dict] = Field(default_factory=list)


class _StubLawyer(BaseAgent):
    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _LawyerLikeOutput


@pytest.mark.asyncio
async def test_turn_fields_filled_from_final_answer(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode":"consultation","primary_answer":"不合法",'
                 '"citations":[{"law_short":"民法典","article_no":"703","excerpt":"x"},'
                 '{"law_short":"民法典","article_no":"510","excerpt":"y"}]}',
            finish_reason="end_turn",
        ),
    ])
    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )
    await run_query(query="房东涨租", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s1", memory_store=store,
                    extra_agents_invoked=["receptionist"])
    turns = store.recent_turns("s1", n=1)
    assert len(turns) == 1
    t = turns[0]
    assert t.answer_mode == "consultation"
    assert len(t.citations) == 2
    laws_arts = {(c.law, c.article) for c in t.citations}
    assert laws_arts == {("民法典", "703"), ("民法典", "510")}
    assert "receptionist" in t.agents_invoked
    assert "lawyer" in t.agents_invoked
    # total_tokens 由 events.jsonl 累加, StubProvider 不填 usage → 应为 0 但不抛
    assert t.total_tokens >= 0


@pytest.mark.asyncio
async def test_patch_turn_supervisor_verdict(tmp_path):
    """Phase 6m: patch_turn 能事后补 supervisor_verdict."""
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode":"consultation","primary_answer":"ok","citations":[]}',
            finish_reason="end_turn",
        ),
    ])
    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )
    await run_query(query="q", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s2", memory_store=store)
    # 初始 verdict 空
    turns = store.recent_turns("s2", n=1)
    assert turns[0].supervisor_verdict == ""

    # patch: 补 verdict + 加 supervisor 进 agents_invoked
    path = store.patch_turn("s2", 1,
                            supervisor_verdict="pass",
                            agents_invoked=["lawyer", "supervisor"])
    assert path is not None and path.exists()

    # 重读, 字段已更新
    turns2 = store.recent_turns("s2", n=1)
    assert turns2[0].supervisor_verdict == "pass"
    assert turns2[0].agents_invoked == ["lawyer", "supervisor"]


def test_patch_turn_missing_session_returns_none(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    assert store.patch_turn("nonexistent", 1, supervisor_verdict="x") is None


def test_patch_turn_missing_turn_returns_none(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    (store.root / "sessions" / "s3" / "turns").mkdir(parents=True)
    assert store.patch_turn("s3", 999, supervisor_verdict="x") is None


@pytest.mark.asyncio
async def test_turn_clarification_mode_recorded(tmp_path):
    """clarification mode 也应该写进 Turn.answer_mode."""
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode":"clarification","primary_answer":"信息不足","citations":[]}',
            finish_reason="end_turn",
        ),
    ])
    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )
    await run_query(query="q", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s4", memory_store=store)
    turns = store.recent_turns("s4", n=1)
    assert turns[0].answer_mode == "clarification"
    assert turns[0].citations == []
