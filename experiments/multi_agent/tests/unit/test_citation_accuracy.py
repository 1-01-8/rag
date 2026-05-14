import pytest
from multi_agent.eval.judges.citation_accuracy import (
    CitationAccuracyJudge, CitationAccuracyResult,
)
from multi_agent.eval.queryset import Query, ExpectedAnswer


def test_citation_hit():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s",
              expected=ExpectedAnswer(should_cite_any=["民法典-510", "民法典-563"]))
    lawyer_output = {
        "citations": [
            {"law_short": "民法典", "article_no": "510", "excerpt": "..."},
        ],
    }
    j = CitationAccuracyJudge()
    r = j.judge(q, lawyer_output)
    assert r.hit is True
    assert "民法典-510" in r.matched


def test_citation_miss():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s",
              expected=ExpectedAnswer(should_cite_any=["民法典-510"]))
    lawyer_output = {"citations": [{"law_short": "民法典", "article_no": "999", "excerpt": ""}]}
    r = CitationAccuracyJudge().judge(q, lawyer_output)
    assert r.hit is False
    assert r.matched == []


def test_no_expectation_skipped():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s")
    r = CitationAccuracyJudge().judge(q, {"citations": []})
    assert r.skipped is True
