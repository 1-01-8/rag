"""Unit tests for LLMJudge base class (Phase 5c Task 1)."""
from __future__ import annotations
import pytest
from pydantic import BaseModel
from multi_agent.eval.judges.base import LLMJudge, JudgeResult
from multi_agent.providers.stub import StubProvider, ScriptedResponse


class _DummyOut(BaseModel):
    score: float
    issues: list[str] = []


class _DummyJudge(LLMJudge[_DummyOut]):
    name = "dummy"
    output_schema = _DummyOut

    def render_prompt(self, *, query, lawyer_output, evidence_pool) -> str:
        return f"judge: {query} -> {lawyer_output}"


@pytest.mark.asyncio
async def test_judge_calls_provider_and_parses(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.8, "issues": ["x"]}', finish_reason="end_turn"),
    ])
    j = _DummyJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(query="Q", lawyer_output={"a": 1}, evidence_pool=[])
    assert isinstance(result, JudgeResult)
    assert result.judge == "dummy"
    assert result.score == 0.8
    assert result.parsed.issues == ["x"]
    assert result.error is None


@pytest.mark.asyncio
async def test_judge_handles_malformed_json(tmp_path):
    p = StubProvider(responses=[
        ScriptedResponse(text='not json', finish_reason="end_turn"),
    ])
    j = _DummyJudge(provider=p, model="stub", judge_run_dir=tmp_path / "judge_run")
    result = await j.judge(query="Q", lawyer_output={}, evidence_pool=[])
    assert result.error is not None
    assert result.score == 0.0
