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
) -> dict:
    """Top-level entry. Guarantees a RunFinished event regardless of outcome.

    Returns a small dict {run_id, status, final_answer?}.

    Optional params:
        session_id: if provided alongside memory_store, appends a Turn and
                    updates StickyContext.linked_runs after a successful run.
        memory_store: a MarkdownMemoryStore instance (or compatible duck-type).
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
        try:
            recorder.emit(RunFinished(
                event_id=recorder.fresh_event_id(), run_id=run_id,
                timestamp=recorder.now(), parent_id=None,
                status=status, final_answer=final_answer, error=error,
            ))
        finally:
            recorder.close()

    result = {"run_id": run_id, "status": status, "final_answer": final_answer}

    # Memory integration: persist Turn and update StickyContext after success.
    if session_id and memory_store is not None and status == "ok":
        from multi_agent.schemas.memory import StickyContext, Turn
        sticky = memory_store.read_sticky(session_id) or StickyContext(session_id=session_id)
        if run_id not in sticky.linked_runs:
            sticky.linked_runs.append(run_id)
        existing_turns = memory_store.recent_turns(session_id, n=999)
        next_turn_no = max((t.turn for t in existing_turns), default=0) + 1
        memory_store.append_turn(session_id, Turn(
            turn=next_turn_no,
            run_id=run_id,
            started_at=started_at,
            finished_at=datetime.now(),
            question=query,
            final_answer=final_answer or "",
            agents_invoked=[agent.name] if agent is not None else [],
        ))
        memory_store.write_sticky(sticky)

    return result
