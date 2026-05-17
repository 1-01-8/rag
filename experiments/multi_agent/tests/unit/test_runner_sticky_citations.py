"""Phase 6k: runner 从 Lawyer JSON 抽 citations 更新 sticky 累积字段."""
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
    """Minimal Lawyer-shaped output schema."""
    mode: Literal["consultation", "clarification"] = "consultation"
    primary_answer: str = ""
    citations: list[dict] = Field(default_factory=list)


class _StubLawyer(BaseAgent):
    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _LawyerLikeOutput


@pytest.mark.asyncio
async def test_sticky_cited_articles_accumulate_across_turns(tmp_path):
    """跨多 turn, sticky.cited_articles 应累积去重, mentioned_laws / last_law_name 更新."""
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        # Turn 1: 引 民法典 703 + 510
        ScriptedResponse(
            text='{"mode":"consultation","primary_answer":"涨租不合法",'
                 '"citations":[{"law_short":"民法典","article_no":"703","excerpt":"x"},'
                 '{"law_short":"民法典","article_no":"510","excerpt":"y"}]}',
            finish_reason="end_turn",
        ),
        # Turn 2: 引 民法典 1167 (新) + 民法典 703 (已有, 应去重) + 治安法 58 (新法)
        ScriptedResponse(
            text='{"mode":"consultation","primary_answer":"可起诉",'
                 '"citations":[{"law_short":"民法典","article_no":"1167","excerpt":"z"},'
                 '{"law_short":"民法典","article_no":"703","excerpt":"x"},'
                 '{"law_short":"治安管理处罚法","article_no":"58","excerpt":"w"}]}',
            finish_reason="end_turn",
        ),
    ])

    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )

    # Turn 1
    await run_query(query="房东涨租合法吗", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s1", memory_store=store)
    sticky1 = store.read_sticky("s1")
    assert sticky1 is not None
    assert len(sticky1.cited_articles) == 2
    assert {(c.law, c.article) for c in sticky1.cited_articles} == {("民法典", "703"), ("民法典", "510")}
    assert sticky1.mentioned_laws == ["民法典"]
    assert sticky1.last_law_name == "民法典"

    # Turn 2 — 累积去重
    await run_query(query="可以起诉吗", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s1", memory_store=store)
    sticky2 = store.read_sticky("s1")
    # 应该 4 条 (民法典-703/510/1167 + 治安法-58), 703 不重复
    assert len(sticky2.cited_articles) == 4
    laws_arts = {(c.law, c.article) for c in sticky2.cited_articles}
    assert laws_arts == {("民法典", "703"), ("民法典", "510"),
                        ("民法典", "1167"), ("治安管理处罚法", "58")}
    # mentioned_laws 累积去重
    assert set(sticky2.mentioned_laws) == {"民法典", "治安管理处罚法"}


@pytest.mark.asyncio
async def test_sticky_unchanged_when_citations_empty(tmp_path):
    """clarification mode (citations=[]) 不应往 sticky 写空 / 不污染."""
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode":"clarification","primary_answer":"需要更多信息","citations":[]}',
            finish_reason="end_turn",
        ),
    ])
    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )
    await run_query(query="qq", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s2", memory_store=store)
    sticky = store.read_sticky("s2")
    assert sticky is not None
    assert sticky.cited_articles == []
    assert sticky.mentioned_laws == []
    assert sticky.last_law_name == ""


@pytest.mark.asyncio
async def test_sticky_skips_empty_law_or_article_entries(tmp_path):
    """citations 里 law_short 或 article_no 为空的条目应跳过, 不污染 sticky."""
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode":"consultation","primary_answer":"ok","citations":['
                 '{"law_short":"民法典","article_no":"703","excerpt":"x"},'
                 '{"law_short":"","article_no":"123","excerpt":"e1"},'       # 空 law, 应跳过
                 '{"law_short":"民法典","article_no":"","excerpt":"e2"},'    # 空 article, 应跳过
                 '{"law_short":"刑法","article_no":"234","excerpt":"y"}]}',
            finish_reason="end_turn",
        ),
    ])
    factory = lambda p, r: _StubLawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )
    await run_query(query="q", agent_factory=factory,
                    provider=provider, runs_root=tmp_path / "runs",
                    session_id="s3", memory_store=store)
    sticky = store.read_sticky("s3")
    laws_arts = {(c.law, c.article) for c in sticky.cited_articles}
    # 只 2 条好的进
    assert laws_arts == {("民法典", "703"), ("刑法", "234")}
