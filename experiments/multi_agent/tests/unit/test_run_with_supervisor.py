"""Tests for run_with_supervisor orchestrator (Phase 5a Task 4)."""
import json
import pytest
from pydantic import BaseModel
from multi_agent.orchestration.supervised import run_with_supervisor
from multi_agent.agents.base import BaseAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


class _LawyerOut(BaseModel):
    mode: str
    primary_answer: str


class _Lawyer(BaseAgent):
    def system_prompt(self) -> str:
        return "test lawyer"

    def output_schema(self):
        return _LawyerOut


@pytest.mark.asyncio
async def test_run_with_supervisor_returns_both_results(tmp_path):
    runs_root = tmp_path / "runs"
    lawyer_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode": "consultation", "primary_answer": "测试答复"}',
            finish_reason="end_turn",
        ),
    ])
    supervisor_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"verdict": "pass", "confidence": 0.9, "issues": []}',
            finish_reason="end_turn",
        ),
    ])
    result = await run_with_supervisor(
        query="测试问题",
        lawyer_factory=lambda p, r: _Lawyer(
            name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
        ),
        supervisor_factory=lambda p, r: SupervisorAgent(
            name="supervisor", role="qa", provider=p, recorder=r, model="stub-1",
            max_pre_tool_rejections=10,
        ),
        lawyer_provider=lawyer_provider,
        supervisor_provider=supervisor_provider,
        runs_root=runs_root,
    )
    assert result["lawyer_result"]["status"] == "ok"
    assert result["supervisor_verdict"]["verdict"] == "pass"
    assert "lawyer_run_id" in result
    assert "supervisor_run_id" in result
