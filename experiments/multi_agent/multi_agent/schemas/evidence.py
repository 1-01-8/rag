from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    doc_id: str
    law_name: str
    article_no: str
    text: str
    score: float
    retriever: Literal["bm25", "dense", "hybrid", "exact", "memory", "case", "history"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def law_short(self) -> str:
        """Convenience accessor for metadata['law_short']."""
        return self.metadata.get("law_short", "")
