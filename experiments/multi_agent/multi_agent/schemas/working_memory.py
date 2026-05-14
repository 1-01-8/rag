from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from multi_agent.schemas.evidence import Evidence


class Hypothesis(BaseModel):
    statement: str
    supporting_evidence: list[str] = Field(default_factory=list)  # evidence doc_ids
    confidence: float
    status: Literal["active", "verified", "rejected"] = "active"


class DiscardedEvidence(BaseModel):
    evidence: Evidence
    reason: str


class WorkingMemory(BaseModel):
    """Run-internal scratchpad shared between agents in one run.

    Written to trace artifacts on RunFinished; NOT persisted to memory_store.
    """
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    retrieved_evidence: list[Evidence] = Field(default_factory=list)
    discarded_evidence: list[DiscardedEvidence] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    intermediate_drafts: list[str] = Field(default_factory=list)

    def add_evidence(self, e: Evidence) -> None:
        self.retrieved_evidence.append(e)

    def discard(self, e: Evidence, reason: str) -> None:
        self.discarded_evidence.append(DiscardedEvidence(evidence=e, reason=reason))
