from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One law article — the unit of retrieval (per spec §4.4).

    `text` is the raw article body. `embedding_text()` prepends
    law/book/chapter/article context to improve dense recall on
    short articles (spec §4.4 'Embedding 拼接').
    """
    doc_id: str                                 # e.g. "民法典-510"
    law_name: str                               # e.g. "中华人民共和国民法典"
    law_short: str                              # e.g. "民法典"
    article_no: str                             # e.g. "510"
    text: str                                   # article body only
    # Optional structural context (spec §4.4 says skip in V0,
    # but the field stays — populated when chapters get added later)
    book: str = ""
    chapter: str = ""
    # Optional enrichment
    cross_refs: list[str] = Field(default_factory=list)
    preceding_text: str = ""
    following_text: str = ""
    concepts: list[str] = Field(default_factory=list)
    # Free-form metadata (source file path, version, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text fed to the dense encoder. Includes structural context
        so short articles have richer embeddings."""
        parts = [f"《{self.law_short}》"]
        if self.book:
            parts.append(self.book)
        if self.chapter:
            parts.append(self.chapter)
        parts.append(f"第{self.article_no}条")
        head = "·".join(parts)
        return f"{head}: {self.text}"


class Document(BaseModel):
    """One law file (e.g. 民法典 全文). Contains many Chunks."""
    law_name: str
    law_short: str
    source_path: str
    chunks: list[Chunk] = Field(default_factory=list)
