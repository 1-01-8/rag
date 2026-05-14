# Phase 3c — Cross-Turn Compaction Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Spec §5.4.1 — `memory/compaction.py` with `maybe_compact()` that compresses old turns into `StickyContext.history_summary` when a session exceeds `COMPACTION_THRESHOLD` turns. Without this, long sessions explode in prompt tokens.

**Phase 5e starting point:** Tag `phase5e-latency`. ~221 unit tests passing.

---

## Out of scope (Phase 3d / later)

- `ma_user_history` Qdrant collection + turn embedding (Phase 3d)
- `HistorySearchTool` for semantic retrieval over historical turns (Phase 3d)
- Receptionist/Lawyer integration to consume `history_summary` in prompts (existing prompts already accept sticky context — should pick up automatically; verify in integration test)
- Compaction across sessions (only within a session)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   └── memory/
│       └── compaction.py      # maybe_compact + COMPACT_SYSTEM_PROMPT + helpers
└── tests/
    └── unit/
        └── test_compaction.py
```

---

## Task 1: maybe_compact

**Files:**
- Create: `multi_agent/memory/compaction.py`
- Create: `tests/unit/test_compaction.py`

The function `maybe_compact(session_id, store, provider, model)` checks `len(store.recent_turns(session_id, n=999))`; if ≥ `COMPACTION_THRESHOLD`, takes the **oldest** `len - KEEP_RECENT_TURNS` turns, asks the LLM to compress them into a short prose summary, and writes that summary into `StickyContext.history_summary` via `store.write_sticky()`. Old turn files are NOT deleted (traceability).

Configurable constants: `COMPACTION_THRESHOLD=5`, `KEEP_RECENT_TURNS=3`.

### Step 1: Failing test

```python
# tests/unit/test_compaction.py
import pytest
from pathlib import Path
from datetime import datetime, timezone
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.memory.compaction import maybe_compact, COMPACTION_THRESHOLD
from multi_agent.schemas.memory import StickyContext, Turn
from multi_agent.providers.stub import StubProvider, ScriptedResponse


def _new_turn(n: int) -> Turn:
    return Turn(
        turn=n,
        run_id=f"r{n}",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        question=f"Q{n}",
        answer=f"A{n}",
        agents_invoked=["lawyer"],
    )


@pytest.mark.asyncio
async def test_no_compaction_when_below_threshold(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_short"
    store.write_sticky(StickyContext(session_id=sid, created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, COMPACTION_THRESHOLD):    # 4 turns < threshold 5
        store.append_turn(sid, _new_turn(i))

    p = StubProvider(responses=[])    # would error if called
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is False
    sticky = store.read_sticky(sid)
    assert sticky is not None
    assert sticky.history_summary == ""


@pytest.mark.asyncio
async def test_compaction_summarizes_old_turns(tmp_path):
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid, created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, 8):   # 7 turns > threshold 5
        store.append_turn(sid, _new_turn(i))

    p = StubProvider(responses=[
        ScriptedResponse(text="第 1-4 轮: 用户就 X 进行咨询,Lawyer 引用 Y 法条建议 Z。",
                        finish_reason="end_turn"),
    ])
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is True
    sticky = store.read_sticky(sid)
    assert sticky.history_summary.startswith("第 1-4 轮")


@pytest.mark.asyncio
async def test_compaction_keeps_recent_turn_files(tmp_path):
    """Old turn files are NOT deleted — only summarized into sticky."""
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid, created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc)))
    for i in range(1, 8):
        store.append_turn(sid, _new_turn(i))

    turn_files_before = sorted((tmp_path / "sessions" / sid / "turns").glob("*.md"))
    assert len(turn_files_before) == 7

    p = StubProvider(responses=[
        ScriptedResponse(text="summary", finish_reason="end_turn"),
    ])
    await maybe_compact(sid, store, provider=p, model="stub")

    turn_files_after = sorted((tmp_path / "sessions" / sid / "turns").glob("*.md"))
    assert len(turn_files_after) == 7    # nothing deleted


@pytest.mark.asyncio
async def test_idempotent_when_already_compacted(tmp_path):
    """Calling maybe_compact twice doesn't re-summarize (history_summary already set)."""
    store = MarkdownMemoryStore(root=tmp_path)
    sid = "s_long"
    store.write_sticky(StickyContext(session_id=sid, created_at=datetime.now(timezone.utc),
                                     updated_at=datetime.now(timezone.utc),
                                     history_summary="already compacted"))
    for i in range(1, 8):
        store.append_turn(sid, _new_turn(i))
    p = StubProvider(responses=[])   # would error if called
    compacted = await maybe_compact(sid, store, provider=p, model="stub")
    assert compacted is False
    sticky = store.read_sticky(sid)
    assert sticky.history_summary == "already compacted"
```

### Step 2: Failure → ImportError. Then implement.

```python
"""Cross-turn compaction (Spec §5.4.1)."""
from __future__ import annotations
from datetime import datetime, timezone
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import Turn
from multi_agent.providers.base import LLMProvider
# AgentMessage import based on what providers.complete() requires — adapt if needed.


COMPACTION_THRESHOLD = 5
KEEP_RECENT_TURNS = 3


COMPACT_SYSTEM_PROMPT = """你是法律咨询会话压缩器。
给你一段历史 turn 流水,请压缩成 200 字以内的中文 prose summary,
保留:
1. 用户问题主题与演进
2. Lawyer 提及的关键法条(law_short-article)
3. 已达成的结论或方向
不要列表式,不要标题。直接输出 summary。"""


def _format_turns(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        lines.append(f"## Turn {t.turn}")
        lines.append(f"Q: {t.question}")
        lines.append(f"A: {t.answer[:500]}")
    return "\n".join(lines)


async def maybe_compact(
    session_id: str,
    store: MarkdownMemoryStore,
    *,
    provider: LLMProvider,
    model: str,
) -> bool:
    """If session has > COMPACTION_THRESHOLD turns AND no existing summary,
    compress oldest turns into history_summary. Returns True if a compaction ran."""
    sticky = store.read_sticky(session_id)
    if sticky is None:
        return False
    if sticky.history_summary and sticky.history_summary.strip():
        return False    # already compacted; idempotent

    turns = store.recent_turns(session_id, n=999)
    if len(turns) <= COMPACTION_THRESHOLD:
        return False

    to_compact = turns[:-KEEP_RECENT_TURNS]
    body = _format_turns(to_compact)

    # Adapt the messages= shape to whatever LLMProvider.complete() expects.
    # See multi_agent/eval/judges/base.py for the pattern (you may need to create
    # AgentMessage objects with role="system"/"user" and a fresh Recorder + agent_name).
    from multi_agent.providers.base import AgentMessage  # adjust if path differs
    from multi_agent.tracing.recorder import Recorder
    import tempfile
    from pathlib import Path

    # Lightweight throwaway recorder
    tmp = Path(tempfile.mkdtemp(prefix="compact_"))
    rec = Recorder(run_id=f"compact-{session_id}", run_dir=tmp)
    try:
        resp = await provider.complete(
            model=model,
            messages=[
                AgentMessage(role="system", content=COMPACT_SYSTEM_PROMPT),
                AgentMessage(role="user", content=body),
            ],
            tools=None,
            temperature=0.0,
            max_tokens=512,
            recorder=rec,
            agent_name="compactor",
        )
    finally:
        rec.close()

    summary = (resp.text or "").strip()
    if not summary:
        return False

    sticky.history_summary = summary
    sticky.updated_at = datetime.now(timezone.utc)
    store.write_sticky(sticky)
    return True
```

**IMPORTANT for implementer:** the import names + provider.complete() signature must match what already exists in this project. Read `multi_agent/eval/judges/base.py` (commit a250be1) which already calls `provider.complete()` correctly — mirror its pattern verbatim including the throwaway-Recorder dance.

### Step 3: Verify

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_compaction.py -v"
```

Expected: 4 tests pass.

### Step 4: Commit + tag

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/memory/compaction.py experiments/multi_agent/tests/unit/test_compaction.py
git commit -m "phase3c(memory): cross-turn compaction (spec §5.4.1)"
git tag -a phase3c-compaction -m "Phase 3c: maybe_compact compresses old turns into StickyContext.history_summary"
git tag -l "phase*"
```

---

## Acceptance Criteria

1. 4 unit tests pass
2. `maybe_compact` is a no-op when turns ≤ threshold
3. `maybe_compact` is idempotent when history_summary already set
4. Old turn files preserved on disk after compaction
5. Tag `phase3c-compaction` exists

## Out of Scope (Phase 3d+)

- `ma_user_history` Qdrant collection + semantic turn retrieval
- Receptionist/Lawyer prompt changes (existing prompts already read history_summary from sticky — verify)
- Re-compaction on a session that grew further after first compaction (one-shot for now)
- Compaction in the run_query flow (caller invokes maybe_compact after run completes)
