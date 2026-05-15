"""Phase 3d E2E: index 3 turns → semantic search retrieves the right one.

Requires Qdrant reachable at localhost:6433 and bge-m3 on GPU.
Uses a unique ephemeral collection per run (cleaned up in `finally`).
"""
from __future__ import annotations
import uuid
import pytest
from datetime import datetime, timezone

from multi_agent.schemas.memory import Turn
from multi_agent.tools.retrievers.turn_indexer import TurnIndexer
from multi_agent.tools.retrievers.history_search import HistorySearchTool, HistorySearchArgs
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.qdrant_client import drop_collection, get_qdrant_client
from multi_agent.tracing.recorder import Recorder


def _qdrant_reachable() -> bool:
    try:
        get_qdrant_client().get_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qdrant_reachable(), reason="Qdrant not reachable")


def _turn(n: int, q: str, a: str) -> Turn:
    return Turn(
        turn=n,
        run_id=f"r{n}",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        question=q,
        final_answer=a,
        agents_invoked=["lawyer"],
    )


@pytest.mark.asyncio
async def test_history_index_and_retrieve(tmp_path):
    """Index 3 turns across 2 sessions; session-scoped query returns correct turn."""
    coll = f"test_hist_{uuid.uuid4().hex[:8]}"
    encoder = DenseEncoder()
    indexer = TurnIndexer(collection_name=coll, dense_encoder=encoder)
    try:
        await indexer.index_turn(session_id="s1", turn=_turn(
            1, "房东合同期内涨租 30% 合法吗", "不合法，需协商一致"))
        await indexer.index_turn(session_id="s1", turn=_turn(
            2, "邻居漏水把我家天花板泡了如何索赔", "走侵权赔偿"))
        await indexer.index_turn(session_id="s2", turn=_turn(
            1, "网购到假货怎么退款", "可主张违约责任"))

        tool = HistorySearchTool(collection_name=coll, dense_encoder=encoder)

        # Session-scoped: rental query should match s1/turn_no=1
        rec = Recorder(run_id="r", run_dir=tmp_path / "r")
        result = await tool.call(
            HistorySearchArgs(query="租金调整", session_id="s1", k=2), rec,
        )
        rec.close()

        hits = result.payload["hits"]
        assert len(hits) >= 1, f"Expected hits, got {hits}"
        # turn_no=1 (rental) should rank above turn_no=2 (water leak)
        turn_nos = [h["turn_no"] for h in hits]
        assert 1 in turn_nos, f"turn_no=1 not in hits: {hits}"
        assert hits[0]["turn_no"] == 1, (
            f"Expected turn_no=1 as top hit, got turn_no={hits[0]['turn_no']}. "
            f"Full hits: {hits}"
        )
        assert hits[0]["session_id"] == "s1"

    finally:
        drop_collection(coll)


@pytest.mark.asyncio
async def test_history_all_sessions_retrieval(tmp_path):
    """scope='all_sessions' finds the rental turn even when session_id differs."""
    coll = f"test_hist_{uuid.uuid4().hex[:8]}"
    encoder = DenseEncoder()
    indexer = TurnIndexer(collection_name=coll, dense_encoder=encoder)
    try:
        await indexer.index_turn(session_id="s1", turn=_turn(
            1, "房东合同期内涨租 30% 合法吗", "不合法，需协商一致"))
        await indexer.index_turn(session_id="s1", turn=_turn(
            2, "邻居漏水把我家天花板泡了如何索赔", "走侵权赔偿"))
        await indexer.index_turn(session_id="s2", turn=_turn(
            1, "网购到假货怎么退款", "可主张违约责任"))

        tool = HistorySearchTool(collection_name=coll, dense_encoder=encoder)

        rec = Recorder(run_id="r2", run_dir=tmp_path / "r2")
        result = await tool.call(
            HistorySearchArgs(
                query="租金", session_id="ignored", scope="all_sessions", k=3
            ),
            rec,
        )
        rec.close()

        hits = result.payload["hits"]
        assert len(hits) >= 1, f"Expected hits, got {hits}"
        top = hits[0]
        # Rental turn should rank first across all 3 turns
        assert top["session_id"] == "s1" and top["turn_no"] == 1, (
            f"Expected s1/turn_no=1 at top, got {top}. Full hits: {hits}"
        )

    finally:
        drop_collection(coll)
