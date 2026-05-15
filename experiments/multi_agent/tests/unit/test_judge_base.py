"""Unit tests for LLMJudge base class (Phase 5c Task 1)."""
from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path
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


@pytest.mark.asyncio
async def test_judge_no_run_dir_cleans_up_temp():
    """Fix 1: self-created temp dirs must be removed after judge() returns."""
    import glob
    import os
    p = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.5}', finish_reason="end_turn"),
    ])
    j = _DummyJudge(provider=p, model="stub")  # no judge_run_dir
    before = set(glob.glob(str(Path(tempfile.gettempdir()) / "judge_run_*")))
    await j.judge(query="Q", lawyer_output={}, evidence_pool=[])
    after = set(glob.glob(str(Path(tempfile.gettempdir()) / "judge_run_*")))
    leaked = after - before
    assert len(leaked) == 0, f"Temp dir(s) leaked: {leaked}"


@pytest.mark.asyncio
async def test_concurrent_judges_use_separate_subdirs(tmp_path):
    """Fix 2: concurrent judge() calls under the same judge_run_dir must each
    write to their own per-call subdir, not the shared parent dir."""
    responses = [
        ScriptedResponse(text='{"score": 0.9}', finish_reason="end_turn"),
        ScriptedResponse(text='{"score": 0.7}', finish_reason="end_turn"),
    ]
    p = StubProvider(responses=responses)
    shared_dir = tmp_path / "shared_judge"
    j = _DummyJudge(provider=p, model="stub", judge_run_dir=shared_dir)

    r1, r2 = await asyncio.gather(
        j.judge(query="Q1", lawyer_output={}, evidence_pool=[]),
        j.judge(query="Q2", lawyer_output={}, evidence_pool=[]),
    )
    assert r1.error is None
    assert r2.error is None

    # Each call should have created its own subdir
    subdirs = list(shared_dir.iterdir()) if shared_dir.exists() else []
    assert len(subdirs) == 2, f"Expected 2 per-call subdirs, got: {subdirs}"


@pytest.mark.asyncio
async def test_caller_supplied_run_dir_not_deleted(tmp_path):
    """Fix 1: caller-supplied judge_run_dir must NOT be removed by judge()."""
    p = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.3}', finish_reason="end_turn"),
    ])
    caller_dir = tmp_path / "caller_dir"
    caller_dir.mkdir()
    j = _DummyJudge(provider=p, model="stub", judge_run_dir=caller_dir)
    await j.judge(query="Q", lawyer_output={}, evidence_pool=[])
    assert caller_dir.exists(), "Caller-supplied dir must not be deleted by judge()"
