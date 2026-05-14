import pytest
from multi_agent.schemas.supervisor import (
    SupervisorVerdict, CitationCheckResult, GroundednessCheck,
)


def test_supervisor_verdict_pass():
    v = SupervisorVerdict(
        verdict="pass",
        confidence=0.85,
        issues=[],
        suggested_fix=None,
        citation_checks=[
            CitationCheckResult(citation_index=0, valid=True, reason="matches text"),
        ],
    )
    assert v.verdict == "pass"
    assert v.is_valid is True


def test_supervisor_verdict_reject():
    v = SupervisorVerdict(
        verdict="reject",
        confidence=0.9,
        issues=["citation 民法典-999 not in retrieved evidence"],
        suggested_fix="Re-retrieve and cite only verified articles",
    )
    assert v.verdict == "reject"
    assert v.is_valid is False
    assert "民法典-999" in v.issues[0]


def test_supervisor_verdict_unknown_kind_rejected():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SupervisorVerdict(verdict="bogus", confidence=0.5, issues=[])


def test_citation_check_result():
    c = CitationCheckResult(citation_index=2, valid=False, reason="text doesn't match")
    assert c.valid is False


def test_groundedness_check():
    g = GroundednessCheck(score=0.7, ungrounded_claims=["claim about 民法典 999"])
    assert g.score == 0.7
    assert len(g.ungrounded_claims) == 1
