"""Unit tests for GroundednessJudge (Phase 5c Task 2)."""
from __future__ import annotations
import pytest
from multi_agent.eval.judges.groundedness import GroundednessJudge, GroundednessOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_groundedness_judge_grounded_answer(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.95, "ungrounded_claims": [], "rationale": "All claims have evidence"}',
            finish_reason="end_turn",
        ),
    ])
    j = GroundednessJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(
        query="房东合同期内涨租 30% 合法吗?",
        lawyer_output={"primary_answer": "不合法", "citations": [
            {"law_short": "民法典", "article_no": "703", "excerpt": "..."}
        ]},
        evidence_pool=[{"doc_id": "民法典-703", "law_short": "民法典", "article_no": "703",
                        "text": "租赁合同..."}],
    )
    assert result.error is None
    assert result.score == 0.95
    assert result.parsed.ungrounded_claims == []


@pytest.mark.asyncio
async def test_groundedness_judge_flags_hallucination(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.3, "ungrounded_claims": ["claim about 民法典-999 with no source"], "rationale": "Hallucinated cite"}',
            finish_reason="end_turn",
        ),
    ])
    j = GroundednessJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(query="Q", lawyer_output={"primary_answer": "..."}, evidence_pool=[])
    assert result.score == 0.3
    assert len(result.parsed.ungrounded_claims) == 1
