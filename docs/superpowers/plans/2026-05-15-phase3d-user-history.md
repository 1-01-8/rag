# Phase 3d — `ma_user_history` Collection + HistorySearchTool

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Spec §4.2 / §3.2 long-term-memory layer. Embed each completed `Turn` into a Qdrant `ma_user_history` collection (dense-only). Expose a `HistorySearchTool` that retrieves semantically-similar past turns — keyed by `session_id` (default) or across all sessions (opt-in via flag). Useful for: (a) detecting follow-ups, (b) showing Lawyer past discussion threads on similar topics.

**Phase 3c starting point:** Tag `phase3c-compaction`. ~225 unit tests + 1 skipped + integration.

---

## Out of scope (Phase 3e / later)

- Receptionist/Lawyer prompt integration (the tool is wired and tested; agent prompts that consume it are a follow-up)
- Cross-user history retrieval (would need user_id discriminator)
- Re-embedding on turn compaction (compaction summary not embedded)
- Hybrid sparse+dense over turns (dense-only suffices for semantic turn match)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   └── tools/
│       └── retrievers/
│           ├── turn_indexer.py        # TurnIndexer.index_turn(turn) → embed+upsert
│           └── history_search.py      # HistorySearchTool — dense retrieval, session-scoped
└── tests/
    ├── unit/
    │   ├── test_turn_indexer.py
    │   └── test_history_search.py
    └── integration/
        └── test_history_retrieval_e2e.py   # real Qdrant + bge-m3
```

---

## Task 1: TurnIndexer

**Files:**
- Create: `multi_agent/tools/retrievers/turn_indexer.py`
- Create: `tests/unit/test_turn_indexer.py`

`TurnIndexer.index_turn(session_id, turn)` builds a string like `f"{turn.question}\n{turn.final_answer}"`, dense-encodes via bge-m3, upserts a Qdrant point with payload `{session_id, turn_no, run_id, question_preview, answer_preview, started_at}`.

Point ID: use `f"{session_id}:{turn.turn}"` mapped through `uuid.uuid5` for deterministic deduplication (re-indexing same turn just overwrites).

### Step 1: Failing test

```python
# tests/unit/test_turn_indexer.py
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
    mock_encoder.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
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
    mock_encoder.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
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
```

### Step 2: Implement

```python
"""TurnIndexer — embed Turn into ma_user_history (Phase 3d §4.2)."""
from __future__ import annotations
import uuid
from qdrant_client import models

from multi_agent.schemas.memory import Turn
from multi_agent.tools.retrievers.qdrant_client import (
    get_qdrant_client, ensure_collection, CollectionParams,
)
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


HISTORY_COLLECTION_PARAMS = CollectionParams(dense_dim=1024)

_NAMESPACE = uuid.UUID("e6bf57c0-3a8e-4d22-9b8f-5ac4f6e3b7d1")   # arbitrary fixed namespace


def _point_id(session_id: str, turn_no: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{session_id}:{turn_no}"))


class TurnIndexer:
    def __init__(self, *, collection_name: str, dense_encoder: DenseEncoder):
        self.collection_name = collection_name
        self.dense_encoder = dense_encoder
        ensure_collection(collection_name, HISTORY_COLLECTION_PARAMS)

    async def index_turn(self, *, session_id: str, turn: Turn) -> None:
        text = f"{turn.question}\n{turn.final_answer or ''}".strip()
        # encode returns shape (1, dim) for a single string in a list
        vec = self.dense_encoder.encode([text])[0]
        client = get_qdrant_client()
        client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=_point_id(session_id, turn.turn),
                    vector={"dense": vec.tolist()},
                    payload={
                        "session_id": session_id,
                        "turn_no": turn.turn,
                        "run_id": turn.run_id,
                        "question_preview": (turn.question or "")[:200],
                        "answer_preview": (turn.final_answer or "")[:300],
                        "started_at": turn.started_at.isoformat() if turn.started_at else None,
                    },
                ),
            ],
        )
```

Adapt:
- `DenseEncoder.encode` signature — read `multi_agent/tools/retrievers/dense_encoder.py` first. If it expects different input/output shape, adapt.
- `Turn.final_answer` was confirmed in Phase 3c (commit 2f7cd44). `Turn.question` and `Turn.started_at` should exist — verify before writing.

### Step 3: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_turn_indexer.py -v"
```

Expected: 2 tests pass.

### Step 4: Commit

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/turn_indexer.py experiments/multi_agent/tests/unit/test_turn_indexer.py
git commit -m "phase3d(memory): TurnIndexer — embed Turn into ma_user_history"
```

---

## Task 2: HistorySearchTool

**Files:**
- Create: `multi_agent/tools/retrievers/history_search.py`
- Create: `tests/unit/test_history_search.py`

Dense-only retrieval (no sparse, no RRF) over `ma_user_history`. Filter by `session_id` by default; pass `scope="all_sessions"` for cross-session search.

### Step 1: Failing test

```python
# tests/unit/test_history_search.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
from qdrant_client import models
from multi_agent.tools.retrievers.history_search import (
    HistorySearchTool, HistorySearchArgs,
)
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_history_search_filters_by_session(tmp_path):
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.zeros((1, 1024), dtype=np.float32)

    fake_hit = MagicMock()
    fake_hit.score = 0.8
    fake_hit.payload = {
        "session_id": "s1", "turn_no": 3, "run_id": "r3",
        "question_preview": "涨租可以吗", "answer_preview": "不合法",
        "started_at": "2026-05-14T00:00:00",
    }
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[fake_hit])

    with patch("multi_agent.tools.retrievers.history_search.get_qdrant_client", return_value=mock_client):
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
    mock_encoder = MagicMock()
    mock_encoder.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
    mock_client = MagicMock()
    mock_client.query_points.return_value = MagicMock(points=[])

    with patch("multi_agent.tools.retrievers.history_search.get_qdrant_client", return_value=mock_client):
        tool = HistorySearchTool(collection_name="test_hist", dense_encoder=mock_encoder)
        rec = Recorder(run_id="r", run_dir=tmp_path / "r")
        await tool.call(
            HistorySearchArgs(query="q", session_id="s1", scope="all_sessions"), rec,
        )
        rec.close()

    call = mock_client.query_points.call_args
    qf = call.kwargs.get("query_filter")
    assert qf is None    # no filter when scope=all_sessions
```

### Step 2: Implement

```python
"""HistorySearchTool — dense retrieval over ma_user_history (Phase 3d §3.2)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


class HistorySearchArgs(BaseModel):
    query: str
    session_id: str
    k: int = 5
    scope: Literal["session", "all_sessions"] = "session"


class HistorySearchTool(Tool):
    name: str = "history_search"
    description: str = (
        "Search semantically-similar past Q&A turns in the user's history. "
        "Default scope='session' restricts to the current session_id; "
        "scope='all_sessions' searches across all sessions."
    )
    args_schema: type[BaseModel] = HistorySearchArgs
    collection_name: str
    dense_encoder: DenseEncoder

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: HistorySearchArgs, recorder: Recorder) -> ToolResult:
        vec = self.dense_encoder.encode([args.query])[0].tolist()
        client = get_qdrant_client()
        qf = None
        if args.scope == "session":
            qf = models.Filter(must=[
                models.FieldCondition(key="session_id",
                                      match=models.MatchValue(value=args.session_id)),
            ])
        resp = client.query_points(
            collection_name=self.collection_name,
            query=vec,
            using="dense",
            limit=args.k,
            query_filter=qf,
            with_payload=True,
        )
        hits = []
        for h in resp.points:
            hits.append({
                "score": float(h.score),
                **(h.payload or {}),
            })
        return ToolResult(tool_use_id="", payload={"hits": hits})
```

Adapt:
- `Tool` base — read `multi_agent/tools/base.py` to confirm exact subclass contract (Pydantic field declarations vs class attrs).
- `query_points` shape — mirror what `statute_search.py` does. The `using="dense"` keyword may need to be inside a different parameter name.

### Step 3: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_history_search.py -v"
```

Expected: 2 tests pass.

### Step 4: Real-Qdrant integration test

```python
# tests/integration/test_history_retrieval_e2e.py
"""Phase 3d E2E: index 3 turns → semantic search retrieves the right one."""
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
        turn=n, run_id=f"r{n}",
        started_at=datetime.now(timezone.utc), finished_at=datetime.now(timezone.utc),
        question=q, final_answer=a, agents_invoked=["lawyer"],
    )


@pytest.mark.asyncio
async def test_history_index_and_retrieve(tmp_path):
    coll = f"test_hist_{uuid.uuid4().hex[:8]}"
    encoder = DenseEncoder()
    indexer = TurnIndexer(collection_name=coll, dense_encoder=encoder)
    try:
        await indexer.index_turn(session_id="s1", turn=_turn(
            1, "房东合同期内涨租 30% 合法吗", "不合法,需协商一致"))
        await indexer.index_turn(session_id="s1", turn=_turn(
            2, "邻居漏水把我家天花板泡了如何索赔", "走侵权赔偿"))
        await indexer.index_turn(session_id="s2", turn=_turn(
            1, "网购到假货怎么退款", "可主张违约责任"))

        tool = HistorySearchTool(collection_name=coll, dense_encoder=encoder)
        rec = Recorder(run_id="r", run_dir=tmp_path / "r")

        # Session-scoped: rental query should match turn 1
        result = await tool.call(
            HistorySearchArgs(query="租金调整", session_id="s1", k=2), rec,
        )
        rec.close()
        hits = result.payload["hits"]
        assert len(hits) >= 1
        assert hits[0]["turn_no"] == 1
        assert hits[0]["session_id"] == "s1"

        # All-sessions: rental query crosses sessions but rental turn ranks top
        rec2 = Recorder(run_id="r2", run_dir=tmp_path / "r2")
        result2 = await tool.call(
            HistorySearchArgs(query="租金", session_id="ignored",
                             scope="all_sessions", k=3), rec2,
        )
        rec2.close()
        hits2 = result2.payload["hits"]
        assert len(hits2) >= 1
        top = hits2[0]
        assert top["session_id"] == "s1" and top["turn_no"] == 1
    finally:
        drop_collection(coll)
```

### Step 5: Run

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_history_search.py tests/integration/test_history_retrieval_e2e.py -v"
```

Expected: 4 tests pass (2 unit + 2 integration).

### Step 6: Commit + tag

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/history_search.py \
        experiments/multi_agent/tests/unit/test_history_search.py \
        experiments/multi_agent/tests/integration/test_history_retrieval_e2e.py
git commit -m "phase3d(memory): HistorySearchTool — dense session-scoped turn retrieval"
git tag -a phase3d-user-history -m "Phase 3d: ma_user_history collection + TurnIndexer + HistorySearchTool"
git tag -l "phase*"
```

---

## Acceptance Criteria

1. 4 unit tests pass (2 indexer + 2 search)
2. Integration test passes when Qdrant reachable (E2E retrieval correctness)
3. `TurnIndexer.index_turn` is idempotent for same (session_id, turn_no)
4. `HistorySearchTool` honors `scope` flag for session-vs-all
5. Tag `phase3d-user-history` exists

## Out of Scope (Phase 3e+)

- Agent prompt integration (Receptionist auto-detects follow-ups by querying history_search; Lawyer reads similar past resolutions)
- Cross-user isolation (no user_id; assumes single user)
- `AllSourcesSearchTool` extension to include user_history as a third source
- Re-indexing turn when compaction overwrites history_summary (not the same data — turns are unchanged)
