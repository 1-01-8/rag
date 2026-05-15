"""Unit tests for TurnIndexer (Phase 3d Task 1)."""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import numpy as np
from multi_agent.tools.retrievers.turn_indexer import TurnIndexer, HISTORY_COLLECTION_PARAMS
from multi_agent.schemas.memory import Turn


def _turn(n: int) -> Turn:
    return Turn(
        turn=n, run_id=f"r{n}",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        question=f"问题 {n}: 房东涨租", final_answer=f"答复 {n}: 不合法",
        agents_invoked=["lawyer"],
    )


@pytest.mark.asyncio
async def test_turn_indexer_upserts_point():
    """TurnIndexer should call dense_encoder + qdrant upsert with the right shape."""
    mock_encoder = MagicMock()
    # DenseEncoder.encode_batch returns shape (N, dim); called with a list of 1 text
    mock_encoder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)
    mock_client = MagicMock()

    with patch("multi_agent.tools.retrievers.turn_indexer.get_qdrant_client", return_value=mock_client), \
         patch("multi_agent.tools.retrievers.turn_indexer.ensure_collection") as mock_ensure:
        indexer = TurnIndexer(collection_name="test_hist", dense_encoder=mock_encoder)
        await indexer.index_turn(session_id="s1", turn=_turn(1))

    mock_ensure.assert_called_once()
    assert mock_client.upsert.call_count == 1
    args = mock_client.upsert.call_args
    points = args.kwargs.get("points") or args.args[1]
    assert len(points) == 1
    p = points[0]
    assert p.payload["session_id"] == "s1"
    assert p.payload["turn_no"] == 1
    assert "房东涨租" in p.payload["question_preview"]
    assert "不合法" in p.payload["answer_preview"]


@pytest.mark.asyncio
async def test_turn_indexer_deterministic_point_id():
    """Same (session_id, turn_no) → same point id (idempotent upsert)."""
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)
    mock_client = MagicMock()
    with patch("multi_agent.tools.retrievers.turn_indexer.get_qdrant_client", return_value=mock_client), \
         patch("multi_agent.tools.retrievers.turn_indexer.ensure_collection"):
        indexer = TurnIndexer(collection_name="test_hist", dense_encoder=mock_encoder)
        await indexer.index_turn(session_id="s1", turn=_turn(7))
        await indexer.index_turn(session_id="s1", turn=_turn(7))
    ids_seen = []
    for call in mock_client.upsert.call_args_list:
        points = call.kwargs.get("points") or call.args[1]
        ids_seen.append(points[0].id)
    assert ids_seen[0] == ids_seen[1]
