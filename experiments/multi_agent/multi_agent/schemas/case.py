"""Case Q&A schema — one entry from laws_data after LLM extraction."""
from __future__ import annotations
from pydantic import BaseModel, Field


class CaseQA(BaseModel):
    """A single legal Q&A pair with extracted citations.

    Sourced from laws_data train/*.json. The `extracted_cite_ids` field is
    populated by Phase 2d Task 3's extraction script — list of doc_ids (e.g.
    "民法典-510") that the lawyer answer references.
    """
    case_id: str                                    # e.g. "train_001234"
    cause: str                                      # 5 categories: 交通事故 / 婚姻家庭 / 债权债务 / 劳动纠纷 / 房产纠纷
    question: str
    answer: str                                     # primary lawyer answer
    candidate_answers: list[str] = Field(default_factory=list)
    extracted_cite_ids: list[str] = Field(default_factory=list)  # ["民法典-510", ...]
    extraction_confidence: float = 0.0              # 0..1, from extraction LLM
