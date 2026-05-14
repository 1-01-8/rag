"""Phase 5c integration — REAL Anthropic API. ~$0.10/run.

Gated by ANTHROPIC_API_KEY env var; skipped when the key is absent so the test
never incurs cost in CI / local unit runs.  Run manually with:

    ANTHROPIC_API_KEY=sk-... conda run -n qwen35 bash -c \
        "cd /home/xxm/rag/experiments/multi_agent && \
         pytest tests/integration/test_claude_judges_e2e.py -v -m expensive"
"""
from __future__ import annotations

import os
import pytest

from multi_agent.eval.judges.groundedness import GroundednessJudge
from multi_agent.eval.judges.helpfulness import HelpfulnessJudge
from multi_agent.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.expensive,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set — skipping paid Anthropic API call",
    ),
]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_QUERY = "什么是租赁合同?"

_LAWYER_OUTPUT = {
    "primary_answer": (
        "根据《民法典》第703条,租赁合同是出租人将租赁物交付承租人使用、"
        "收益,承租人支付租金的合同。"
    ),
    "citations": [
        {
            "law_short": "民法典",
            "article_no": "703",
            "excerpt": "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同",
        }
    ],
}

_EVIDENCE_POOL = [
    {
        "doc_id": "民法典-703",
        "law_short": "民法典",
        "article_no": "703",
        "text": "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。",
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_claude_groundedness_judge_scores_grounded_answer(tmp_path):
    """GroundednessJudge against a well-cited answer should score >= 0.7."""
    provider = AnthropicProvider()
    judge = GroundednessJudge(
        provider=provider,
        model="claude-opus-4-7",
        judge_run_dir=tmp_path / "judges",
    )
    result = await judge.judge(
        query=_QUERY,
        lawyer_output=_LAWYER_OUTPUT,
        evidence_pool=_EVIDENCE_POOL,
    )
    assert result.error is None, f"Judge returned error: {result.error}"
    assert result.score >= 0.7, (
        f"Expected groundedness >= 0.7 for a well-cited answer, got {result.score}. "
        f"raw={result.raw!r}"
    )


@pytest.mark.asyncio
async def test_real_claude_helpfulness_judge_scores_responsive_answer(tmp_path):
    """HelpfulnessJudge against an answer that addresses the question should score >= 0.5."""
    provider = AnthropicProvider()
    judge = HelpfulnessJudge(
        provider=provider,
        model="claude-opus-4-7",
        judge_run_dir=tmp_path / "judges",
    )
    result = await judge.judge(
        query=_QUERY,
        lawyer_output=_LAWYER_OUTPUT,
        evidence_pool=_EVIDENCE_POOL,
    )
    assert result.error is None, f"Judge returned error: {result.error}"
    assert result.score >= 0.5, (
        f"Expected helpfulness >= 0.5 for a responsive answer, got {result.score}. "
        f"raw={result.raw!r}"
    )
