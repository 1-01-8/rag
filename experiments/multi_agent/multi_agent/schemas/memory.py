"""Schemas for the file-based memory store (spec §5)."""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


# --- EntityState components (spec §5.4 frontmatter) ---

class ActiveSubject(BaseModel):
    role: str
    identifier: str
    attributes: list[str] = Field(default_factory=list)


class KeyFact(BaseModel):
    fact: str
    confidence: Literal["low", "medium", "high"] = "high"
    source_turn: int = 0


class RejectedPath(BaseModel):
    path: str
    reason: str


class EntityState(BaseModel):
    """Structured facts extracted from a session (spec §5.4)."""
    active_subjects: list[ActiveSubject] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    rejected_paths: list[RejectedPath] = Field(default_factory=list)
    legal_objectives: list[str] = Field(default_factory=list)


class CitedArticle(BaseModel):
    law: str
    article: str
    from_turn: int = 0


class StickyContext(BaseModel):
    """Running session state — sticky.md frontmatter (spec §5.4)."""
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    legal_domain: str = ""
    case_type: str = ""
    last_law_name: str = ""
    mentioned_laws: list[str] = Field(default_factory=list)
    cited_articles: list[CitedArticle] = Field(default_factory=list)
    linked_runs: list[str] = Field(default_factory=list)
    entity_state: EntityState = Field(default_factory=EntityState)
    history_summary: str = ""
    body: str = ""


# --- Intent-based read_sticky views (spec §5.4.3) ---
#
# Each view exposes only the slice an agent needs, so prompt token usage stays
# low. All views always include session_id so the caller can correlate.

StickyIntent = Literal["full", "entities_only", "recent_citations", "summary_only"]


class StickyEntitiesView(BaseModel):
    """Slice returned by read_sticky(intent='entities_only').

    Used by Receptionist to decide follow-up resolution without paying the
    full StickyContext token cost.
    """
    session_id: str
    entity_state: EntityState = Field(default_factory=EntityState)


class StickyCitationsView(BaseModel):
    """Slice returned by read_sticky(intent='recent_citations').

    Used by Secretary to decide whether to query user_history for related
    prior turns.
    """
    session_id: str
    cited_articles: list[CitedArticle] = Field(default_factory=list)


class StickySummaryView(BaseModel):
    """Slice returned by read_sticky(intent='summary_only').

    The compressed prose summary of older turns (post-compaction, spec §5.4.1).
    """
    session_id: str
    history_summary: str = ""


class Turn(BaseModel):
    """One conversation turn — turns/NNN-slug.md (spec §5.5)."""
    turn: int
    run_id: str
    started_at: datetime
    finished_at: datetime
    question: str
    final_answer: str
    answer_mode: str = "evidence_grounded"
    supervisor_verdict: str = ""
    agents_invoked: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    citations: list[CitedArticle] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class AgentNote(BaseModel):
    """Cross-session learning — agent_notes/<slug>.md (spec §5.6)."""
    name: str
    description: str
    produced_by: str
    about_agent: str
    verdict_that_triggered: str = ""
    tags: list[str] = Field(default_factory=list)
    triggered_by_run: str = ""
    used_in_runs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = 0
    body: str = ""
