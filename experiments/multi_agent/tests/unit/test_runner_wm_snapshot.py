"""Phase 5r: WorkingMemory snapshot persisted to artifacts/working_memory.json (spec §5.4.2)."""
from __future__ import annotations
import json
import pytest
from pydantic import BaseModel
from multi_agent.runner import run_query
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.evidence import Evidence
from multi_agent.schemas.working_memory import WorkingMemory


class _Out(BaseModel):
    answer: str


class _LawyerWithWM(BaseAgent):
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
        ])


class _LawyerNoWM(BaseAgent):
    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_working_memory_snapshot_written(tmp_path):
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    result = await run_query(
        query="q",
        agent_factory=lambda p, r: _LawyerWithWM(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
    )
    assert result["status"] == "ok"
    wm_path = tmp_path / "runs" / result["run_id"] / "artifacts" / "working_memory.json"
    assert wm_path.exists(), f"Expected snapshot at {wm_path}"
    data = json.loads(wm_path.read_text(encoding="utf-8"))
    assert "retrieved_evidence" in data
    assert len(data["retrieved_evidence"]) == 1
    assert data["retrieved_evidence"][0]["doc_id"] == "民法典-703"


@pytest.mark.asyncio
async def test_empty_working_memory_still_written(tmp_path):
    """BaseAgent always creates a fresh WorkingMemory (per Phase 3b design),
    so the snapshot exists but is empty."""
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    result = await run_query(
        query="q",
        agent_factory=lambda p, r: _LawyerNoWM(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=tmp_path / "runs",
    )
    wm_path = tmp_path / "runs" / result["run_id"] / "artifacts" / "working_memory.json"
    assert wm_path.exists()
    data = json.loads(wm_path.read_text(encoding="utf-8"))
    assert data.get("retrieved_evidence") == []
