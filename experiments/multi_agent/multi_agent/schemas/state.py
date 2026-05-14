from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class RunState(BaseModel):
    """Top-level state for a single run. Passed by value through agents.

    Phase 1: minimal. Will be expanded in later phases with
    retrieval / memory / planner fields.
    """
    run_id: str
    session_id: str
    user_query: str
    history_messages: list[dict[str, Any]] = Field(default_factory=list)
    failed_queries: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
