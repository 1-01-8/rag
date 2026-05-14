"""Search ma_cases collection (laws_data Q&A pairs).

Returns Evidence whose .text combines question + answer for downstream
agents to ingest as 'similar case' context.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


class CaseSearchArgs(BaseModel):
    query: str
    k: int = 5
    cause: str | None = None


class CaseSearchTool(Tool):
    name: str = "case_search"
    description: str = (
        "Search past legal Q&A cases. Returns similar real-world cases "
        "(question + lawyer answer + extracted citations). Optional cause filter."
    )
    args_schema: type[BaseModel] = CaseSearchArgs
    collection_name: str
    sparse_artifact_path: Path

    _dense: Any = None
    _sparse: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _ensure_encoders(self) -> None:
        if self._dense is None:
            object.__setattr__(self, "_dense", DenseEncoder())
        if self._sparse is None:
            object.__setattr__(self, "_sparse", SparseEncoder.load(self.sparse_artifact_path))

    async def call(self, args: CaseSearchArgs, recorder: Recorder) -> ToolResult:
        self._ensure_encoders()
        client = get_qdrant_client()
        dense_vec = self._dense.encode_one(args.query).tolist()
        sparse_vec = self._sparse.encode(args.query)

        query_filter = None
        if args.cause:
            query_filter = models.Filter(must=[
                models.FieldCondition(
                    key="cause", match=models.MatchValue(value=args.cause),
                )
            ])

        result = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense",
                                limit=max(args.k * 2, 20), filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values,
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
            text = f"[问题] {payload.get('question', '')}\n[律师答复] {payload.get('answer', '')}"
            ev = Evidence(
                doc_id=payload.get("case_id", ""),
                law_name="(case)",
                law_short="",
                article_no=payload.get("case_id", ""),
                text=text,
                score=float(point.score) if point.score is not None else 0.0,
                retriever="case",
                metadata={
                    "cause": payload.get("cause", ""),
                    "extracted_cite_ids": payload.get("extracted_cite_ids", []),
                    "extraction_confidence": payload.get("extraction_confidence", 0.0),
                },
            )
            evidences.append(ev.model_dump())

        return ToolResult(
            tool_use_id="",
            payload={"evidences": evidences, "count": len(evidences)},
        )
