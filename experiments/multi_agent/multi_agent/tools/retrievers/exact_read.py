"""Look up a specific article by (law_short, article_no).

Uses Qdrant's scroll with a payload filter — no embedding required.
"""
from __future__ import annotations
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client


class ExactReadArgs(BaseModel):
    law_short: str
    article_no: str


class ExactReadTool(Tool):
    name: str = "read_article"
    description: str = (
        "Look up the full text of a specific article by law name and number. "
        "Use when the user asks 'what does Article X of Law Y say'."
    )
    args_schema: type[BaseModel] = ExactReadArgs
    collection_name: str

    async def call(self, args: ExactReadArgs, recorder: Recorder) -> ToolResult:
        client = get_qdrant_client()
        result, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_short",
                        match=models.MatchValue(value=args.law_short),
                    ),
                    models.FieldCondition(
                        key="article_no",
                        match=models.MatchValue(value=args.article_no),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not result:
            return ToolResult(
                tool_use_id="",
                payload=None,
                error=f"article not found: {args.law_short} 第 {args.article_no} 条",
            )
        payload = result[0].payload or {}
        ev = Evidence(
            doc_id=payload.get("doc_id", ""),
            law_name=payload.get("law_name", ""),
            law_short=payload.get("law_short", ""),
            article_no=payload.get("article_no", ""),
            text=payload.get("text", ""),
            score=1.0,                  # exact match
            retriever="exact",
            metadata={
                "book": payload.get("book", ""),
                "chapter": payload.get("chapter", ""),
            },
        )
        return ToolResult(tool_use_id="", payload={"evidence": ev.model_dump()})
