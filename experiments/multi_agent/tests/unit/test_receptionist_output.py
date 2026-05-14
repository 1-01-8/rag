import pytest
from multi_agent.schemas.receptionist import ReceptionistOutput, SubCase


def test_subcase_required_fields():
    sc = SubCase(issue="责任归属确认", specialty="侵权", priority=1)
    assert sc.priority == 1
    assert sc.requires_separate_retrieval is True


def test_receptionist_output_single_issue():
    out = ReceptionistOutput(
        primary_specialty="民事", case_type="租赁纠纷", urgency="中",
        is_multi_issue=False, sub_cases=[],
        initial_facts=["合同期一年", "涨幅30%"],
        normalized_query="房东合同期内涨租 30% 是否合法",
    )
    assert out.primary_specialty == "民事"
    assert out.is_multi_issue is False
    assert out.risk_flag is None


def test_receptionist_output_multi_issue():
    out = ReceptionistOutput(
        primary_specialty="家事", case_type="离婚+保护令", urgency="高",
        is_multi_issue=True,
        sub_cases=[
            SubCase(issue="离婚诉讼", specialty="家事", priority=2),
            SubCase(issue="人身安全保护令", specialty="治安", priority=1),
        ],
        initial_facts=["原告不想出庭", "被告威胁"],
        normalized_query="离婚诉讼 + 人身保护令",
    )
    assert out.is_multi_issue is True
    assert len(out.sub_cases) == 2
    assert out.sub_cases[1].issue == "人身安全保护令"


def test_receptionist_output_safety_refusal():
    out = ReceptionistOutput(
        primary_specialty="(safety)", case_type="safety_refusal", urgency="高",
        risk_flag="safety_refusal",
        is_multi_issue=False, sub_cases=[],
        initial_facts=[], normalized_query="",
    )
    assert out.risk_flag == "safety_refusal"


def test_receptionist_output_rejects_unknown_urgency():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ReceptionistOutput(
            primary_specialty="x", case_type="y", urgency="bogus",
            is_multi_issue=False, sub_cases=[],
            initial_facts=[], normalized_query="",
        )
