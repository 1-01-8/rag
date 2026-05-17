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
    k: int = 5                       # Phase 6b: 默认 5 (从 10 减半), 上限内部强制 max 5
    law_short: str | None = None     # filter: only this law (e.g. "民法典")


# Phase 6b: 单条法条文本截断长度, 防止第二次 LLM 输入爆大
_MAX_EVIDENCE_TEXT_CHARS = 500
_MAX_K = 5


class StatuteSearchTool(Tool):
    name: str = "statute_search"
    description: str = (
        "Search Chinese statutes using hybrid retrieval "
        "(dense BAAI/bge-m3 + sparse jieba+IDF, fused via RRF). "
        "Returns up to k Evidence objects (k<=5, text truncated to 500 chars). "
        "Optional law_short filter."
    )
    args_schema: type[BaseModel] = StatuteSearchArgs
    # Runtime config — not LLM-visible
    collection_name: str
    sparse_artifact_path: Path

    # Lazy-initialized state — Phase 6d 加 dense_encoder/sparse_encoder 注入字段
    _dense: Any = None
    _sparse: Any = None
    # 注入 (可选): 外部已构造好的 encoder, 避免重复加载 bge-m3
    dense_encoder: Any = None
    sparse_encoder: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _ensure_encoders(self) -> None:
        if self._dense is None:
            # Phase 6d: 优先用外部注入的, 否则新建一个
            object.__setattr__(self, "_dense",
                              self.dense_encoder if self.dense_encoder is not None else DenseEncoder())
        if self._sparse is None:
            object.__setattr__(self, "_sparse",
                              self.sparse_encoder if self.sparse_encoder is not None
                              else SparseEncoder.load(self.sparse_artifact_path))

    async def call(self, args: StatuteSearchArgs, recorder: Recorder) -> ToolResult:
        self._ensure_encoders()
        client = get_qdrant_client()

        # Phase 6b: 服务端硬性上限 k, 不管 LLM 传多大
        effective_k = min(args.k, _MAX_K)

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
                    limit=max(effective_k * 2, 20),
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using="sparse",
                    limit=max(effective_k * 2, 20),
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=effective_k,
            with_payload=True,
        )

        # Phase 6b: 去重 doc_id + 截断 text, 减小返给 LLM 的 payload
        seen_doc_ids: set[str] = set()
        evidences: list[dict] = []
        for point in result.points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id", "")
            if doc_id and doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            full_text = payload.get("text", "") or ""
            text = full_text[:_MAX_EVIDENCE_TEXT_CHARS]
            if len(full_text) > _MAX_EVIDENCE_TEXT_CHARS:
                text += "..."
            ev = Evidence(
                doc_id=doc_id,
                law_name=payload.get("law_name", ""),
                law_short=payload.get("law_short", ""),
                article_no=payload.get("article_no", ""),
                text=text,
                score=float(point.score) if point.score is not None else 0.0,
                retriever="hybrid",
                metadata={
                    # Phase 6b: 去掉非必要 metadata 减小 payload (book/chapter/concepts 当前 LLM 用不上)
                },
            )
            evidences.append(ev.model_dump())

        return ToolResult(
            tool_use_id="",            # filled by BaseAgent._dispatch_tool
            payload={"evidences": evidences, "count": len(evidences)},
        )
