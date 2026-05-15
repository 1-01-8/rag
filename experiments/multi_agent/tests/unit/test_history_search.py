"""Unit tests for HistorySearchTool (Phase 3d Task 2).

Mocks DenseEncoder + Qdrant client — no GPU, no network.
"""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from multi_agent.tools.retrievers.history_search import (
    HistorySearchTool, HistorySearchArgs,
)
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_history_search_filters_by_session(tmp_path):
    """scope='session' (default) must add a session_id filter to query_points."""
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)

    fake_hit = MagicMock()
    fake_hit.score = 0.8
    fake_hit.payload = {
        "session_id": "s1", "turn_no": 3, "run_id": "r3",
        "question_preview": "涨租可以吗", "answer_preview": "不合法",
        "started_at": "2026-05-14T00:00:00",
    }
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[fake_hit])

    with patch("multi_agent.tools.retrievers.history_search.get_qdrant_client",
               return_value=mock_client):
        tool = HistorySearchTool(collection_name="test_hist", dense_encoder=mock_encoder)
        rec = Recorder(run_id="r", run_dir=tmp_path / "r")
        result = await tool.call(
            HistorySearchArgs(query="涨租", session_id="s1", k=3), rec,
        )
        rec.close()

    assert result.error is None
    hits = result.payload["hits"]
    assert len(hits) == 1
    assert hits[0]["session_id"] == "s1"
    # Verify session_id filter was applied
    call = mock_client.query_points.call_args
    qf = call.kwargs.get("query_filter")
    assert qf is not None


@pytest.mark.asyncio
async def test_history_search_all_sessions_no_filter(tmp_path):
    """scope='all_sessions' must omit the session_id filter (query_filter=None)."""
    mock_encoder = MagicMock()
    mock_encoder.encode_batch.return_value = np.zeros((1, 1024), dtype=np.float32)
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[])

    with patch("multi_agent.tools.retrievers.history_search.get_qdrant_client",
               return_value=mock_client):
        tool = HistorySearchTool(collection_name="test_hist", dense_encoder=mock_encoder)
        rec = Recorder(run_id="r", run_dir=tmp_path / "r")
        await tool.call(
            HistorySearchArgs(query="q", session_id="s1", scope="all_sessions"), rec,
        )
        rec.close()

    call = mock_client.query_points.call_args
    qf = call.kwargs.get("query_filter")
    assert qf is None    # no filter when scope=all_sessions
