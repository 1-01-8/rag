from __future__ import annotations
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
) -> dict:
    """Top-level entry. Guarantees a RunFinished event regardless of outcome.

    Returns a small dict {run_id, status, final_answer?}.
    """
    run_id = fresh_run_id()
    run_dir = Path(runs_root) / run_id
    recorder = Recorder(run_id=run_id, run_dir=run_dir)
    recorder.set_meta(query=query, config=(config or {}))

    final_answer: str | None = None
    status = "ok"
    error: str | None = None

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

    return {"run_id": run_id, "status": status, "final_answer": final_answer}
