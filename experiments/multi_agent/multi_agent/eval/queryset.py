"""QuerySet schema + YAML loader (Phase 5b)."""
from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class ExpectedAnswer(BaseModel):
    should_cite_any: list[str] = Field(default_factory=list)
    expected_answer_mode: Literal[
        "evidence_grounded", "clarification_or_refusal", "advisory"
    ] | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class Query(BaseModel):
    id: str
    text: str
    jurisdiction: str = "CN"
    cause: str
    source: str
    source_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    expected: ExpectedAnswer = Field(default_factory=ExpectedAnswer)


class QuerySetMeta(BaseModel):
    name: str
    description: str = ""
    created: date | None = None


class QuerySet(BaseModel):
    meta: QuerySetMeta
    queries: list[Query]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "QuerySet":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
