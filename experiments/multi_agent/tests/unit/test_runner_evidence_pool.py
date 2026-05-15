"""Phase 5p: run_query exposes evidence_pool from agent's WorkingMemory."""
from __future__ import annotations
import pytest
from pydantic import BaseModel
from multi_agent.runner import run_query
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.evidence import Evidence
from multi_agent.schemas.working_memory import WorkingMemory


class _Out(BaseModel):
    answer: str


class _LawyerWithEvidence(BaseAgent):
    """Stub Lawyer that pre-populates working_memory.retrieved_evidence."""

    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _Out

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        self.working_memory = WorkingMemory(retrieved_evidence=[
            Evidence(
                doc_id="民法典-703", law_name="民法典", law_short="民法典",
                article_no="703", text="租赁合同...", score=0.9, retriever="hybrid",
            ),
            Evidence(
                doc_id="民法典-510", law_name="民法典", law_short="民法典",
                article_no="510", text="合同补充内容...", score=0.8, retriever="hybrid",
            ),
        ])


class _LawyerNoEvidence(BaseAgent):
    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_run_query_returns_evidence_pool_from_working_memory(tmp_path):
    """When the agent has retrieved_evidence in working_memory, run_query exposes it."""
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    result = await run_query(
        query="q",
        agent_factory=lambda p, r: _LawyerWithEvidence(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
    )
    assert result["status"] == "ok"
    ep = result.get("evidence_pool")
    assert isinstance(ep, list)
    assert len(ep) == 2
    doc_ids = {e["doc_id"] for e in ep}
    assert doc_ids == {"民法典-703", "民法典-510"}


@pytest.mark.asyncio
async def test_run_query_evidence_pool_empty_without_working_memory(tmp_path):
    """Agents without WorkingMemory expose an empty evidence_pool (not None)."""
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    result = await run_query(
        query="q",
        agent_factory=lambda p, r: _LawyerNoEvidence(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
    )
    assert result["status"] == "ok"
    assert result["evidence_pool"] == []
