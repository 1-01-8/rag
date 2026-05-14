import pytest
from multi_agent.schemas.contract_review import ContractReviewResult
from multi_agent.schemas.doc_generation import GeneratedDoc, DocGenRequest
from multi_agent.schemas.doc_interpret import InterpretResult, DocInterpretRequest
from multi_agent.schemas.lawyer import RiskItem


def test_contract_review_result():
    r = ContractReviewResult(
        risk_items=[RiskItem(level="high", clause="第5条", reason="霸王", suggestion="改为...")],
        missing_clauses=["违约金条款", "争议解决条款"],
        summary="合同存在 1 个高风险条款,缺 2 个必要条款",
        score=65,
    )
    assert r.score == 65
    assert len(r.risk_items) == 1


def test_contract_review_score_must_be_in_range():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ContractReviewResult(risk_items=[], missing_clauses=[], summary="x", score=150)


def test_generated_doc():
    g = GeneratedDoc(
        doc_type="离婚协议",
        content="甲方:...\n乙方:...",
        placeholders_filled={"甲方姓名": "张三", "乙方姓名": "李四"},
        meta={"effective_date": "2026-05-14"},
    )
    assert g.doc_type == "离婚协议"
    assert g.placeholders_filled["甲方姓名"] == "张三"


def test_doc_gen_request():
    r = DocGenRequest(
        doc_type="民事起诉状",
        case_facts="原告与被告...",
        parties={"plaintiff": "张三", "defendant": "李四"},
    )
    assert r.doc_type == "民事起诉状"


def test_interpret_result():
    i = InterpretResult(
        key_clauses=[{"clause": "第3条", "summary": "保密条款"}],
        rights_obligations="...",
        risks=["违约风险"],
        plain_language_summary="这份合同主要规定...",
    )
    assert len(i.key_clauses) == 1
    assert "违约风险" in i.risks


def test_doc_interpret_request():
    r = DocInterpretRequest(doc_text="本合同由甲乙双方签订...")
    assert "甲乙双方" in r.doc_text
