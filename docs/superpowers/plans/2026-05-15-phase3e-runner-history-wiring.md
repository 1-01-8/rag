# Phase 3e — Runner ↔ TurnIndexer Wiring

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Close the long-term-memory loop. `run_query` already persists turns to MD; add an optional `turn_indexer` kwarg so the same Turn is also embedded into `ma_user_history`. Backward-compatible (default `None` = current behavior).

**Phase 3d starting point:** Tag `phase3d-user-history`. 229 unit tests + 1 skipped + integrations.

---

## Out of scope

- Agent prompt changes (Lawyer/Receptionist using HistorySearchTool — Phase 3f)
- Bulk-reindex of existing turns into a new collection (one-off script if needed)
- Sync hook variants (we keep `turn_indexer` async since `index_turn` is async)

---

## Single Task

**Files:**
- Modify: `multi_agent/runner.py` — add `turn_indexer=None` kwarg, call `await turn_indexer.index_turn(...)` after `append_turn`
- Create: `tests/unit/test_runner_history_indexing.py` — 2 tests (indexer called when set; not called when None)

### Step 1: Failing test

```python
# tests/unit/test_runner_history_indexing.py
import pytest
from pathlib import Path
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock
from multi_agent.runner import run_query
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


class _Out(BaseModel):
    answer: str


class _Lawyer(BaseAgent):
    def system_prompt(self) -> str: return "test"
    def output_schema(self): return _Out


@pytest.mark.asyncio
async def test_run_query_indexes_turn_when_indexer_supplied(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    indexer = MagicMock()
    indexer.index_turn = AsyncMock()
    result = await run_query(
        query="测试问题",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        session_id="s1",
        memory_store=store,
        turn_indexer=indexer,
    )
    assert result["status"] == "ok"
    indexer.index_turn.assert_awaited_once()
    call_kwargs = indexer.index_turn.await_args.kwargs
    assert call_kwargs["session_id"] == "s1"
    turn = call_kwargs["turn"]
    assert turn.question == "测试问题"
    assert turn.turn == 1


@pytest.mark.asyncio
async def test_run_query_does_not_index_without_indexer(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path / "mem")
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    # No indexer supplied — must not raise
    result = await run_query(
        query="测试",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        session_id="s1",
        memory_store=store,
        # turn_indexer omitted
    )
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_indexer_only_called_when_session_id_supplied(tmp_path):
    """If no session_id, no turn is appended → no indexing."""
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    indexer = MagicMock()
    indexer.index_turn = AsyncMock()
    await run_query(
        query="q",
        agent_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
        # session_id omitted, memory_store omitted
        turn_indexer=indexer,
    )
    indexer.index_turn.assert_not_awaited()
```

### Step 2: Modify runner.py

Add `turn_indexer=None` parameter. After the existing `memory_store.append_turn(...)` block, if `turn_indexer is not None`, await `turn_indexer.index_turn(session_id=session_id, turn=<the appended turn>)`.

The trick: the current code constructs the Turn inline inside `append_turn(...)` argument. Refactor to:
```python
turn = Turn(turn=next_turn_no, run_id=run_id, ...)
memory_store.append_turn(session_id, turn)
if turn_indexer is not None:
    await turn_indexer.index_turn(session_id=session_id, turn=turn)
```

Keep `memory_store.write_sticky(sticky)` after the indexing.

### Step 3: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_runner_history_indexing.py -v"
```

Expected: 3 tests pass. Then a quick check that existing runner tests still pass:

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/ -q -k 'runner or run_with' 2>&1 | tail -10"
```

Expected: all green (no regressions).

### Step 4: Commit + tag

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/runner.py experiments/multi_agent/tests/unit/test_runner_history_indexing.py
git commit -m "phase3e(runner): optional turn_indexer hook indexes Turns into ma_user_history"
git tag -a phase3e-history-wiring -m "Phase 3e: run_query optionally indexes each turn into ma_user_history"
git tag -l "phase*"
```

---

## Acceptance Criteria

1. 3 unit tests pass
2. Existing runner/run_with_supervisor tests still pass (no regressions)
3. `turn_indexer=None` (default) preserves prior behavior exactly
4. Tag `phase3e-history-wiring` exists
