"""Lawyer + Supervisor orchestration (Phase 5a)."""
from __future__ import annotations
import json as _json
from pathlib import Path
from typing import Callable, Any

from multi_agent.providers.base import LLMProvider
from multi_agent.runner import run_query
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


async def run_with_supervisor(
    *,
    query: str,
    lawyer_factory: Callable,
    supervisor_factory: Callable,
    lawyer_provider: LLMProvider,
    supervisor_provider: LLMProvider,
    runs_root: Path,
    lawyer_config: dict | None = None,
    session_id: str | None = None,
    memory_store=None,
) -> dict[str, Any]:
    """Run Lawyer via run_query, then run Supervisor on the lawyer output.

    Returns a combined dict:
        {
            "lawyer_run_id": str,
            "supervisor_run_id": str,
            "lawyer_result": {status, run_id, final_answer},
            "supervisor_verdict": {verdict, confidence, issues, ...},
        }

    The Supervisor receives the Lawyer's parsed output and the evidence pool
    captured from the Lawyer's WorkingMemory.
    """
    # Wrap lawyer_factory to capture the agent instance after construction.
    # run_query calls agent_factory(provider, recorder) synchronously, so
    # captured["agent"] is populated before any await boundary.
    captured: dict[str, Any] = {"agent": None}

    def _lawyer_factory_wrapped(p, r):
        agent = lawyer_factory(p, r)
        captured["agent"] = agent
        return agent

    lawyer_result = await run_query(
        query=query,
        agent_factory=_lawyer_factory_wrapped,
        provider=lawyer_provider,
        runs_root=runs_root,
        config=lawyer_config or {},
        session_id=session_id,
        memory_store=memory_store,
    )

    # Extract Lawyer's evidence pool from WorkingMemory.
    lawyer_agent = captured["agent"]
    evidence_pool: list[dict] = []
    if lawyer_agent is not None and getattr(lawyer_agent, "working_memory", None):
        evidence_pool = [
            ev.model_dump()
            for ev in lawyer_agent.working_memory.retrieved_evidence
        ]

    # Parse Lawyer's final_answer JSON string into a dict.
    try:
        lawyer_out_dict = _json.loads(lawyer_result.get("final_answer") or "{}")
    except Exception:
        lawyer_out_dict = {}

    # Run Supervisor in its own isolated run/recorder.
    sup_run_id = fresh_run_id()
    sup_recorder = Recorder(run_id=sup_run_id, run_dir=Path(runs_root) / sup_run_id)
    supervisor = supervisor_factory(supervisor_provider, sup_recorder)
    sup_output = await supervisor.run(AgentInput(payload={
        "user_query": query,
        "lawyer_output": lawyer_out_dict,
        "evidence_pool": evidence_pool,
    }))
    sup_recorder.close()

    return {
        "lawyer_run_id": lawyer_result["run_id"],
        "supervisor_run_id": sup_run_id,
        "lawyer_result": lawyer_result,
        "supervisor_verdict": sup_output.payload.model_dump(),
    }
