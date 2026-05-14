"""Legal document interpretation schemas (Phase 4 business tool)."""
from __future__ import annotations
from pydantic import BaseModel, Field


class DocInterpretRequest(BaseModel):
    doc_text: str


class KeyClause(BaseModel):
    clause: str
    summary: str


class InterpretResult(BaseModel):
    key_clauses: list[dict] = Field(default_factory=list)
    rights_obligations: str
    risks: list[str] = Field(default_factory=list)
    plain_language_summary: str
