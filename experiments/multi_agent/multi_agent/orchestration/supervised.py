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

_NOTE_SYSTEM_PROMPT = """\
你是失败模式归档员。给你一段律师答复 + Supervisor 拒绝理由,生成一个简短的 agent_note 用于以后避免类似错误。

输出 JSON:
{
  "name": "lawyer-<slug>-<key-issue>",
  "description": "<一句话总结失败模式>",
  "body": "<200 字以内的 markdown 描述错误 + 建议改进>"
}

只输出 JSON。"""


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
    note_provider: LLMProvider | None = None,
    note_model: str | None = None,
    agent_input_extra: dict[str, Any] | None = None,  # Phase 6h: 上下文透传
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
        agent_input_extra=agent_input_extra,  # Phase 6h 透传
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

    # Short-circuit: clarification mode needs no Supervisor review.
    if lawyer_out_dict.get("mode") == "clarification":
        sup_run_id = fresh_run_id()
        verdict_dict = {
            "verdict": "pass",
            "confidence": 1.0,
            "issues": [],
            "suggested_fix": None,
            "citation_checks": [],
            "groundedness": None,
        }
        return {
            "lawyer_run_id": lawyer_result["run_id"],
            "supervisor_run_id": sup_run_id,
            "lawyer_result": lawyer_result,
            "supervisor_verdict": verdict_dict,
        }

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

    verdict_dict = sup_output.payload.model_dump()

    # §5.6 — on reject, summarize failure into an AgentNote via cheap local LLM.
    if (
        verdict_dict.get("verdict") == "reject"
        and memory_store is not None
        and note_provider is not None
        and note_model is not None
    ):
        try:
            from multi_agent.schemas.messages import AgentMessage
            from multi_agent.providers.json_robust import parse_json_robust
            from multi_agent.schemas.memory import AgentNote

            user_ctx = _json.dumps({
                "lawyer_output": lawyer_out_dict,
                "supervisor_issues": verdict_dict.get("issues", []),
                "supervisor_verdict": verdict_dict,
            }, ensure_ascii=False)

            note_run_id = fresh_run_id()
            note_recorder = Recorder(
                run_id=note_run_id,
                run_dir=Path(runs_root) / note_run_id,
            )
            try:
                note_resp = await note_provider.complete(
                    [
                        AgentMessage(role="system", content=_NOTE_SYSTEM_PROMPT),
                        AgentMessage(role="user", content=user_ctx),
                    ],
                    model=note_model,
                    recorder=note_recorder,
                    agent_name="note-generator",
                )
            finally:
                note_recorder.close()

            parsed = parse_json_robust(note_resp.text)
            note = AgentNote(
                name=parsed.get("name", f"lawyer-reject-{note_run_id[:8]}"),
                description=parsed.get("description", ""),
                body=parsed.get("body", ""),
                produced_by="note-generator",
                about_agent="lawyer",
                verdict_that_triggered="reject",
                triggered_by_run=lawyer_result["run_id"],
            )
            memory_store.write_note(note)
        except Exception:
            pass  # note-generation failure must not break the main return

    return {
        "lawyer_run_id": lawyer_result["run_id"],
        "supervisor_run_id": sup_run_id,
        "lawyer_result": lawyer_result,
        "supervisor_verdict": verdict_dict,
    }
