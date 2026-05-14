"""Supervisor agent output schema."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class CitationCheckResult(BaseModel):
    citation_index: int
    valid: bool
    reason: str


class GroundednessCheck(BaseModel):
    score: float
    ungrounded_claims: list[str] = Field(default_factory=list)


class SupervisorVerdict(BaseModel):
    verdict: Literal["pass", "revise", "reject"]
    confidence: float
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None
    citation_checks: list[CitationCheckResult] = Field(default_factory=list)
    groundedness: GroundednessCheck | None = None

    @property
    def is_valid(self) -> bool:
        return self.verdict == "pass"
