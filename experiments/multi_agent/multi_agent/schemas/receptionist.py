"""Receptionist output schema (spec §3.5)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class SubCase(BaseModel):
    issue: str
    specialty: str
    priority: int = 1
    requires_separate_retrieval: bool = True


class ReceptionistOutput(BaseModel):
    """Triage + decomposition output (spec §3.5)."""
    primary_specialty: str
    case_type: str
    urgency: Literal["低", "中", "高"]
    is_multi_issue: bool = False
    sub_cases: list[SubCase] = Field(default_factory=list)
    initial_facts: list[str] = Field(default_factory=list)
    normalized_query: str = ""
    need_clarification: bool = False
    clarification_q: str | None = None
    risk_flag: str | None = None
