import pytest
from multi_agent.schemas.lawyer import (
    LawyerOutput, FiveSection, Citation, RiskItem,
)


def test_lawyer_output_consultation_mode():
    out = LawyerOutput(
        mode="consultation",
        primary_answer="房东不能单方涨租。",
        citations=[
            Citation(law_short="民法典", article_no="510", excerpt="按照交易习惯..."),
        ],
        five_section=FiveSection(
            dispute_analysis="租赁合同期内房东要求涨租 30%, 用户拒绝。",
            applicable_laws="《民法典》第 510 条规定...",
            similar_cases="（无类案）",
            remedy_suggestions="1. 与房东协商 2. 拒绝缴纳超额租金 3. 必要时仲裁",
            risk_assessment="胜诉可能性较高,因合同未约定涨租条款。",
        ),
    )
    assert out.mode == "consultation"
    assert out.primary_answer.startswith("房东")
    assert len(out.citations) == 1
    assert out.citations[0].article_no == "510"


def test_lawyer_output_other_modes_have_no_five_section():
    """contract_review / doc_generation / doc_interpret are Phase 4 modes
    — five_section may be None for those."""
    out = LawyerOutput(
        mode="contract_review",
        primary_answer="合同存在 2 个风险条款。",
        citations=[],
        risk_items=[
            RiskItem(level="high", clause="第 5 条", reason="霸王条款", suggestion="改为..."),
        ],
    )
    assert out.mode == "contract_review"
    assert out.five_section is None


def test_lawyer_output_rejects_unknown_mode():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LawyerOutput(mode="bogus", primary_answer="", citations=[])
