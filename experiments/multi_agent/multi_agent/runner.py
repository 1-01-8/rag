from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Callable, Any
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id
from multi_agent.schemas.events import RunStarted, RunFinished
from multi_agent.providers.base import LLMProvider
from multi_agent.agents.base import BaseAgent, AgentInput


async def run_query(
    *,
    query: str,
    agent_factory: Callable[[LLMProvider, Recorder], BaseAgent],
    provider: LLMProvider,
    runs_root: Path,
    config: dict[str, Any] | None = None,
    session_id: str | None = None,
    memory_store=None,
    turn_indexer=None,
    compaction_provider: LLMProvider | None = None,
    compaction_model: str | None = None,
) -> dict:
    """Top-level entry. Guarantees a RunFinished event regardless of outcome.

    Returns a small dict {run_id, status, final_answer?}.

    Optional params:
        session_id: if provided alongside memory_store, appends a Turn and
                    updates StickyContext.linked_runs after a successful run.
        memory_store: a MarkdownMemoryStore instance (or compatible duck-type).
        compaction_provider: if set (together with compaction_model), triggers
                    maybe_compact() after write_sticky() per spec §5.4.1.
                    Intentionally separate from `provider` so a cheaper model
                    can be used for compaction.
        compaction_model: model name passed to maybe_compact().
    """
    started_at = datetime.now()
    run_id = fresh_run_id()
    run_dir = Path(runs_root) / run_id
    recorder = Recorder(run_id=run_id, run_dir=run_dir)
    recorder.set_meta(query=query, config=(config or {}))

    final_answer: str | None = None
    status = "ok"
    error: str | None = None
    agent: BaseAgent | None = None

    try:
        recorder.emit(RunStarted(
            event_id=recorder.fresh_event_id(), run_id=run_id,
            timestamp=recorder.now(), parent_id=None,
            query=query, config=(config or {}),
        ))
        agent = agent_factory(provider, recorder)
        output = await agent.run(AgentInput(payload={"query": query}))
        final_answer = output.payload.model_dump_json()
    except Exception as e:
        status = "error"
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        # Spec §5.4.2: persist WorkingMemory snapshot to artifacts/ alongside trace.
        # Best-effort: ignore failures (agent may not exist, or WM may be empty).
        if agent is not None:
            wm = getattr(agent, "working_memory", None)
            if wm is not None:
                try:
                    artifacts_dir = run_dir / "artifacts"
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    (artifacts_dir / "working_memory.json").write_text(
                        wm.model_dump_json(indent=2), encoding="utf-8",
                    )
                except Exception:
                    pass
        try:
            recorder.emit(RunFinished(
                event_id=recorder.fresh_event_id(), run_id=run_id,
                timestamp=recorder.now(), parent_id=None,
                status=status, final_answer=final_answer, error=error,
            ))
        finally:
            recorder.close()

    # Expose evidence_pool from the agent's WorkingMemory (if any) so callers
    # like ExperimentRunner can pass it to LLM judges. Empty list when the
    # agent has no WorkingMemory (e.g. Receptionist).
    evidence_pool: list[dict] = []
    if agent is not None:
        wm = getattr(agent, "working_memory", None)
        if wm is not None:
            evidence_pool = [ev.model_dump() for ev in getattr(wm, "retrieved_evidence", [])]

    result = {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "evidence_pool": evidence_pool,
    }

    # Memory integration: persist Turn and update StickyContext after success.
    if session_id and memory_store is not None and status == "ok":
        from multi_agent.schemas.memory import StickyContext, Turn
        sticky = memory_store.read_sticky(session_id) or StickyContext(session_id=session_id)
        if run_id not in sticky.linked_runs:
            sticky.linked_runs.append(run_id)
        existing_turns = memory_store.recent_turns(session_id, n=999)
        next_turn_no = max((t.turn for t in existing_turns), default=0) + 1
        turn = Turn(
            turn=next_turn_no,
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(),
            question=query,
            final_answer=final_answer or "",
            agents_invoked=[agent.name] if agent is not None else [],
        )
        memory_store.append_turn(session_id, turn)
        memory_store.write_sticky(sticky)

        # Phase 6e: 索引到 ma_user_history 在后台跑, 不阻塞返回
        # (bge-m3 encode + Qdrant upsert 约 1-3 秒, 用户不应该等)
        if turn_indexer is not None:
            import asyncio
            async def _bg_index():
                try:
                    await turn_indexer.index_turn(session_id=session_id, turn=turn)
                except Exception:
                    # Best-effort: 索引失败不影响主流程, 也不抛
                    pass
            # fire-and-forget; 在 Python 3.11+ 上是 detached task
            # 用 ensure_future 兼容更广; loop.create_task 也行
            asyncio.ensure_future(_bg_index())

        # Spec §5.4.1: auto-compact when session exceeds threshold.
        # Only runs when caller opts in with both compaction_provider + compaction_model.
        if compaction_provider is not None and compaction_model is not None:
            from multi_agent.memory.compaction import maybe_compact
            await maybe_compact(
                session_id,
                memory_store,
                provider=compaction_provider,
                model=compaction_model,
            )

    return result
