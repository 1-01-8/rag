"""Hybrid (dense+sparse, RRF-fused) search over the `ma_statutes` collection.

Implemented as a Tool so Phase 2c Lawyer can call it via the standard
ReAct dispatch path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


class StatuteSearchArgs(BaseModel):
    query: str
    k: int = 10
    law_short: str | None = None     # filter: only this law (e.g. "民法典")


class StatuteSearchTool(Tool):
    name: str = "statute_search"
    description: str = (
        "Search Chinese statutes using hybrid retrieval "
        "(dense BAAI/bge-m3 + sparse jieba+IDF, fused via RRF). "
        "Returns up to k Evidence objects. Optional law_short filter."
    )
    args_schema: type[BaseModel] = StatuteSearchArgs
    # Runtime config — not LLM-visible
    collection_name: str
    sparse_artifact_path: Path

    # Lazy-initialized state
    _dense: Any = None
    _sparse: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _ensure_encoders(self) -> None:
        if self._dense is None:
            object.__setattr__(self, "_dense", DenseEncoder())
        if self._sparse is None:
            object.__setattr__(
                self, "_sparse", SparseEncoder.load(self.sparse_artifact_path)
            )

    async def call(self, args: StatuteSearchArgs, recorder: Recorder) -> ToolResult:
        self._ensure_encoders()
        client = get_qdrant_client()

        dense_vec = self._dense.encode_one(args.query).tolist()
        sparse_vec = self._sparse.encode(args.query)

        # Build optional filter
        query_filter = None
        if args.law_short:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_short",
                        match=models.MatchValue(value=args.law_short),
                    )
                ]
            )

        # Native hybrid via prefetch + RRF
        result = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=max(args.k * 2, 20),
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using="sparse",
                    limit=max(args.k * 2, 20),
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=args.k,
            with_payload=True,
        )

        evidences: list[dict] = []
        for point in result.points:
            payload = point.payload or {}
            ev = Evidence(
                doc_id=payload.get("doc_id", ""),
                law_name=payload.get("law_name", ""),
                law_short=payload.get("law_short", ""),
                article_no=payload.get("article_no", ""),
                text=payload.get("text", ""),
                score=float(point.score) if point.score is not None else 0.0,
                retriever="hybrid",
                metadata={
                    "book": payload.get("book", ""),
                    "chapter": payload.get("chapter", ""),
                    "concepts": payload.get("concepts", []),
                },
            )
            evidences.append(ev.model_dump())

        return ToolResult(
            tool_use_id="",            # filled by BaseAgent._dispatch_tool
            payload={"evidences": evidences, "count": len(evidences)},
        )
