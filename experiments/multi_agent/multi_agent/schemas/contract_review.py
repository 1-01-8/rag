"""Contract review output schema (Phase 4 business tool)."""
from __future__ import annotations
from typing import Annotated
from pydantic import BaseModel, Field
from multi_agent.schemas.lawyer import RiskItem


class ContractReviewResult(BaseModel):
    risk_items: list[RiskItem] = Field(default_factory=list)
    missing_clauses: list[str] = Field(default_factory=list)
    summary: str
    score: Annotated[int, Field(ge=0, le=100)]
