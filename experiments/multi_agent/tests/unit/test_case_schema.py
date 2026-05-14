from multi_agent.schemas.case import CaseQA


def test_caseqa_required_fields():
    c = CaseQA(
        case_id="train_001234",
        cause="房产纠纷",
        question="房东要涨房租怎么办?",
        answer="可以与房东协商,不成可起诉。",
        extracted_cite_ids=["民法典-510", "民法典-563"],
    )
    assert c.case_id == "train_001234"
    assert c.cause == "房产纠纷"
    assert len(c.extracted_cite_ids) == 2


def test_caseqa_optional_fields_default():
    c = CaseQA(
        case_id="x", cause="y", question="q", answer="a",
        extracted_cite_ids=[],
    )
    assert c.candidate_answers == []
    assert c.extraction_confidence == 0.0
