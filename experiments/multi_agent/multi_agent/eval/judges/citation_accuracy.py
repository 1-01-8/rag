"""Rule-based citation accuracy judge (Phase 5b §7.7)."""
from __future__ import annotations
from pydantic import BaseModel, Field
from multi_agent.eval.queryset import Query


class CitationAccuracyResult(BaseModel):
    hit: bool = False
    matched: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)
    actual: list[str] = Field(default_factory=list)
    skipped: bool = False
    reason: str = ""


class CitationAccuracyJudge:
    """Pass if any expected citation appears in lawyer output."""

    def judge(self, query: Query, lawyer_output: dict) -> CitationAccuracyResult:
        expected = query.expected.should_cite_any
        actual = [
            f"{c.get('law_short','')}-{c.get('article_no','')}"
            for c in (lawyer_output.get("citations") or [])
        ]
        if not expected:
            return CitationAccuracyResult(
                skipped=True, expected=[], actual=actual,
                reason="No should_cite_any expectation set",
            )
        matched = [c for c in actual if c in expected]
        return CitationAccuracyResult(
            hit=len(matched) > 0,
            matched=matched,
            expected=expected,
            actual=actual,
            reason="" if matched else "no expected citation present",
        )
