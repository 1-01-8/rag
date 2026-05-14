"""Cross-collection retrieval: statutes + cases, merged via local RRF.

Qdrant 1.12 doesn't support cross-collection prefetch, so we query each
collection separately and merge.
"""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool, StatuteSearchArgs
from multi_agent.tools.retrievers.case_search import CaseSearchTool, CaseSearchArgs


class AllSourcesArgs(BaseModel):
    query: str
    k: int = 8
    law_short: str | None = None
    cause: str | None = None


def _rrf_merge(lists: list[list[Evidence]], k_constant: int = 60, top_k: int = 8) -> list[Evidence]:
    """Reciprocal Rank Fusion across multiple ranked Evidence lists.

    Keeps evidences keyed by doc_id; sums 1/(k_constant + rank) across all
    input lists. Returns top_k by fused score.
    """
    fused: dict[str, tuple[Evidence, float]] = {}
    for lst in lists:
        for rank, ev in enumerate(lst):
            score_contribution = 1.0 / (k_constant + rank)
            if ev.doc_id in fused:
                existing_ev, existing_score = fused[ev.doc_id]
                fused[ev.doc_id] = (existing_ev, existing_score + score_contribution)
            else:
                fused[ev.doc_id] = (ev, score_contribution)
    ranked = sorted(fused.values(), key=lambda x: -x[1])[:top_k]
    return [ev.model_copy(update={"score": float(score)}) for ev, score in ranked]


class AllSourcesSearchTool(Tool):
    name: str = "all_sources_search"
    description: str = (
        "Search across BOTH statutes and case law (Q&A pairs) simultaneously. "
        "Results are fused via reciprocal rank fusion. Optional filters: "
        "law_short (limits statute results), cause (limits case results)."
    )
    args_schema: type[BaseModel] = AllSourcesArgs
    statutes_collection: str
    statutes_sparse: Path
    cases_collection: str
    cases_sparse: Path

    async def call(self, args: AllSourcesArgs, recorder: Recorder) -> ToolResult:
        statute_tool = StatuteSearchTool(
            collection_name=self.statutes_collection,
            sparse_artifact_path=self.statutes_sparse,
        )
        case_tool = CaseSearchTool(
            collection_name=self.cases_collection,
            sparse_artifact_path=self.cases_sparse,
        )

        stat_result = await statute_tool.call(
            StatuteSearchArgs(query=args.query, k=args.k, law_short=args.law_short),
            recorder,
        )
        case_result = await case_tool.call(
            CaseSearchArgs(query=args.query, k=args.k, cause=args.cause),
            recorder,
        )

        stat_evs = [Evidence.model_validate(e) for e in (stat_result.payload or {}).get("evidences", [])]
        case_evs = [Evidence.model_validate(e) for e in (case_result.payload or {}).get("evidences", [])]

        fused = _rrf_merge([stat_evs, case_evs], top_k=args.k)
        return ToolResult(
            tool_use_id="",
            payload={
                "evidences": [e.model_dump() for e in fused],
                "count": len(fused),
                "stats": {"statutes": len(stat_evs), "cases": len(case_evs)},
            },
        )
