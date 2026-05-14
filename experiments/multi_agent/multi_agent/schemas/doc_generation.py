"""Legal document generation schemas (Phase 4 business tool)."""
from __future__ import annotations
from pydantic import BaseModel, Field


class DocGenRequest(BaseModel):
    doc_type: str
    case_facts: str
    parties: dict[str, str] = Field(default_factory=dict)
    extra_context: str = ""


class GeneratedDoc(BaseModel):
    doc_type: str
    content: str
    placeholders_filled: dict[str, str] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)
