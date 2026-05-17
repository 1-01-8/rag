"""Phase 6h: Lawyer._render_input 接收 recent_turns / cited_articles / history_summary
   并把它们注入 user message, 让模型能消解指代 ("这个法律" / "上面提到的...").
"""
from __future__ import annotations
import pytest
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.base import AgentInput
from multi_agent.providers.stub import StubProvider
from multi_agent.tracing.recorder import Recorder


def _make_lawyer(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    return LawyerAgent(
        name="lawyer", role="advisor", provider=p, recorder=rec,
        tools=[], model="stub", specialty="民事",
    )


def test_recent_turns_injected_into_render(tmp_path):
    """有 recent_turns 时, prompt 含历史 Q/A 原文."""
    lawyer = _make_lawyer(tmp_path)
    rendered = lawyer._render_input(AgentInput(payload={
        "query": "我要根据这个法律起诉他",
        "recent_turns": [
            {"question": "房东涨租 30% 合法吗", "final_answer": "不合法, 根据民法典 703 条..."},
            {"question": "如果起诉应该怎么准备", "final_answer": "建议先收集合同..."},
        ],
    }))
    assert "本会话历史" in rendered
    assert "房东涨租 30%" in rendered
    assert "民法典 703" in rendered
    assert "我要根据这个法律起诉他" in rendered
    # 提醒消解指代
    assert "指代" in rendered or "前文" in rendered


def test_cited_articles_injected(tmp_path):
    lawyer = _make_lawyer(tmp_path)
    rendered = lawyer._render_input(AgentInput(payload={
        "query": "继续上面的问题",
        "cited_articles": [
            {"law": "民法典", "article": "703"},
            {"law": "民法典", "article": "510"},
        ],
    }))
    assert "本会话累积引用过的法条" in rendered
    assert "《民法典》第703条" in rendered
    assert "《民法典》第510条" in rendered


def test_history_summary_injected(tmp_path):
    lawyer = _make_lawyer(tmp_path)
    rendered = lawyer._render_input(AgentInput(payload={
        "query": "下一步怎么做",
        "history_summary": "前 3 轮: 用户咨询涨租问题, 已确定民法典 703 / 510 适用",
    }))
    assert "早期对话摘要" in rendered
    assert "民法典 703 / 510 适用" in rendered


def test_no_context_keeps_old_behavior(tmp_path):
    """没传 context 字段时, 行为跟 Phase 6 之前一致 (单 query)."""
    lawyer = _make_lawyer(tmp_path)
    rendered = lawyer._render_input(AgentInput(payload={
        "query": "房东涨租合法吗",
    }))
    # 没有 context, 不该出现 "本会话历史" header
    assert "本会话历史" not in rendered
    assert rendered == "房东涨租合法吗"


def test_prefetched_evidences_still_works_with_context(tmp_path):
    """Phase 6f 快路径跟 6h 上下文叠加: context prefix → prefetched → 当前问题."""
    lawyer = _make_lawyer(tmp_path)
    rendered = lawyer._render_input(AgentInput(payload={
        "query": "起诉准备",
        "prefetched_evidences": [{"doc_id": "民法典-703", "text": "..."}],
        "recent_turns": [{"question": "涨租", "final_answer": "不合法"}],
    }))
    assert "本会话历史" in rendered            # context 在前
    assert "已经检索好的相关法条" in rendered    # prefetch 在中
    assert "起诉准备" in rendered               # 当前问题
