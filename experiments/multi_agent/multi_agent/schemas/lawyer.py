"""Lawyer agent output schema — five-section structured answers per spec §3.5."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A specific article citation with excerpt for grounding."""
    law_short: str
    article_no: str
    excerpt: str = ""


class FiveSection(BaseModel):
    """The five-section consultation framework (spec §3.5.1)."""
    dispute_analysis: str
    applicable_laws: str
    similar_cases: str
    remedy_suggestions: str
    risk_assessment: str


class RiskItem(BaseModel):
    """Used by contract_review mode (Phase 4)."""
    level: Literal["high", "medium", "low"]
    clause: str
    reason: str
    suggestion: str


class LawyerOutput(BaseModel):
    """Top-level lawyer output. Mode selects sub-fields (spec §3.5.2)."""
    mode: Literal["consultation", "contract_review", "doc_generation", "doc_interpret"]
    primary_answer: str
    citations: list[Citation] = Field(default_factory=list)
    five_section: FiveSection | None = None
    risk_items: list[RiskItem] | None = None
    generated_doc: str | None = None
    interpretation: dict | None = None
