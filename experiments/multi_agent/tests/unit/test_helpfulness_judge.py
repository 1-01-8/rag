"""Unit tests for HelpfulnessJudge (Phase 5c Task 3)."""
from __future__ import annotations
import pytest
from multi_agent.eval.judges.helpfulness import HelpfulnessJudge, HelpfulnessOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_helpfulness_judge_helpful_answer(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.9, "missing_aspects": [], "rationale": "Answer is direct, actionable, and complete"}',
            finish_reason="end_turn",
        ),
    ])
    j = HelpfulnessJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(
        query="房东合同期内涨租 30% 合法吗?",
        lawyer_output={"primary_answer": "不合法。根据民法典第703条,合同期内不得单方面调整租金。建议向住建部门投诉或向法院提起诉讼。",
                       "next_steps": ["向住建部门投诉", "收集证据", "提起诉讼"]},
        evidence_pool=[{"doc_id": "民法典-703", "law_short": "民法典", "article_no": "703",
                        "text": "租赁合同期内，不得单方面调整租金..."}],
    )
    assert result.error is None
    assert result.score == 0.9
    assert result.parsed.missing_aspects == []


@pytest.mark.asyncio
async def test_helpfulness_judge_unhelpful_answer(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.2, "missing_aspects": ["actionable next steps", "specific legal remedies"], "rationale": "Answer is vague and provides no guidance"}',
            finish_reason="end_turn",
        ),
    ])
    j = HelpfulnessJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(
        query="我被公司无故辞退,怎么办?",
        lawyer_output={"primary_answer": "这是一个复杂的法律问题,需要具体分析。"},
        evidence_pool=[],
    )
    assert result.score == 0.2
    assert len(result.parsed.missing_aspects) == 2
