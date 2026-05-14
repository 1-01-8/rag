from multi_agent.schemas.evidence import Evidence


def test_evidence_fields():
    e = Evidence(
        doc_id="民法典-510",
        law_name="中华人民共和国民法典",
        article_no="510",
        text="当事人就合同补充内容...",
        score=0.85,
        retriever="hybrid",
        metadata={"book": "合同编"},
    )
    assert e.doc_id == "民法典-510"
    assert e.score == 0.85
    assert e.retriever == "hybrid"


def test_evidence_rejects_unknown_retriever():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Evidence(
            doc_id="x", law_name="y", article_no="1", text="t",
            score=0.5, retriever="banana",
        )
