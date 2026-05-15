"""HistorySearchTool — dense retrieval over ma_user_history (Phase 3d §3.2).

Dense-only (no sparse, no RRF) since semantic turn matching does not benefit
from keyword recall. Filter by session_id by default; pass scope="all_sessions"
for cross-session search.
"""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder  # noqa: F401 (type hint only)


class HistorySearchArgs(BaseModel):
    query: str
    # Optional: if omitted, the tool's default_session_id is used. Allows the
    # Lawyer to call this tool without knowing/passing the session_id every time.
    session_id: str | None = None
    k: int = 5
    scope: Literal["session", "all_sessions"] = "session"


class HistorySearchTool(Tool):
    """Dense-retrieval tool over the ma_user_history collection.

    Pydantic field `dense_encoder` holds the bge-m3 encoder; arbitrary_types_allowed
    is already set on the Tool base class model_config.

    `default_session_id` may be set at construction so the agent can call this
    tool without passing session_id in every invocation.
    """

    name: str = "history_search"
    description: str = (
        "Search semantically-similar past Q&A turns in the user's history. "
        "Default scope='session' restricts to the current session; "
        "scope='all_sessions' searches across all sessions."
    )
    args_schema: type[BaseModel] = HistorySearchArgs
    collection_name: str
    # Any so MagicMock works in unit tests; accepts real DenseEncoder at runtime
    dense_encoder: Any
    default_session_id: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: HistorySearchArgs, recorder: Recorder) -> ToolResult:
        # encode_batch returns shape (N, dim); slice row 0 for the single query
        vec = self.dense_encoder.encode_batch([args.query])[0].tolist()

        client = get_qdrant_client()

        # Resolve session_id: explicit arg wins, fall back to construction default
        effective_session_id = args.session_id or self.default_session_id

        # Build filter: session-scoped by default, omit for all_sessions
        query_filter = None
        if args.scope == "session":
            if not effective_session_id:
                return ToolResult(
                    tool_use_id="",
                    payload={"hits": [], "error": "session-scoped search requires session_id"},
                )
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=effective_session_id),
                    )
                ]
            )

        resp = client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="dense",
            limit=args.k,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: list[dict] = []
        for h in resp.points:
            hits.append({
                "score": float(h.score) if h.score is not None else 0.0,
                **(h.payload or {}),
            })

        return ToolResult(tool_use_id="", payload={"hits": hits})
