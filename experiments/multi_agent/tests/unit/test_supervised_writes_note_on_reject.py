"""Tests for Phase 5u: Supervisor reject → write AgentNote (spec §5.6)."""
import pytest
from pydantic import BaseModel
from pathlib import Path

from multi_agent.orchestration.supervised import run_with_supervisor
from multi_agent.agents.base import BaseAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.tracing.recorder import Recorder


class _LawyerOut(BaseModel):
    mode: str
    primary_answer: str


class _Lawyer(BaseAgent):
    def system_prompt(self) -> str:
        return "test lawyer"

    def output_schema(self):
        return _LawyerOut


def _make_lawyer_factory():
    return lambda p, r: _Lawyer(
        name="lawyer", role="advisor", provider=p, recorder=r, model="stub-1",
    )


def _make_supervisor_factory():
    return lambda p, r: SupervisorAgent(
        name="supervisor", role="qa", provider=p, recorder=r, model="stub-1",
        max_pre_tool_rejections=10,
    )


@pytest.mark.asyncio
async def test_reject_with_memory_store_writes_note(tmp_path):
    """verdict='reject' + memory_store + note_provider → AgentNote file appears."""
    runs_root = tmp_path / "runs"
    memory_store = MarkdownMemoryStore(tmp_path / "memory")

    lawyer_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode": "consultation", "primary_answer": "不完整的答复"}',
            finish_reason="end_turn",
        ),
    ])
    supervisor_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"verdict": "reject", "confidence": 0.85, "issues": ["答复缺少法律依据"]}',
            finish_reason="end_turn",
        ),
    ])
    note_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"name": "lawyer-reject-missing-basis", "description": "律师答复缺乏法律依据", "body": "## 问题\\n律师未引用具体法条。\\n\\n## 建议\\n确保答复包含相关法律条文的引用。"}',
            finish_reason="end_turn",
        ),
    ])

    result = await run_with_supervisor(
        query="劳动合同纠纷问题",
        lawyer_factory=_make_lawyer_factory(),
        supervisor_factory=_make_supervisor_factory(),
        lawyer_provider=lawyer_provider,
        supervisor_provider=supervisor_provider,
        runs_root=runs_root,
        memory_store=memory_store,
        note_provider=note_provider,
        note_model="stub-model",
    )

    assert result["supervisor_verdict"]["verdict"] == "reject"

    notes_dir = tmp_path / "memory" / "agent_notes"
    note_files = list(notes_dir.glob("*.md"))
    assert len(note_files) == 1, f"Expected 1 note file, found: {note_files}"

    note_content = note_files[0].read_text(encoding="utf-8")
    assert "lawyer-reject-missing-basis" in note_content
    assert "律师答复缺乏法律依据" in note_content


@pytest.mark.asyncio
async def test_pass_verdict_does_not_write_note(tmp_path):
    """verdict='pass' → no AgentNote written even when memory_store is provided."""
    runs_root = tmp_path / "runs"
    memory_store = MarkdownMemoryStore(tmp_path / "memory")

    lawyer_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode": "consultation", "primary_answer": "完整的答复，含法律依据"}',
            finish_reason="end_turn",
        ),
    ])
    supervisor_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"verdict": "pass", "confidence": 0.95, "issues": []}',
            finish_reason="end_turn",
        ),
    ])
    # note_provider not set — but even if it were, pass should not trigger it
    note_provider = StubProvider(responses=[])

    result = await run_with_supervisor(
        query="劳动合同问题",
        lawyer_factory=_make_lawyer_factory(),
        supervisor_factory=_make_supervisor_factory(),
        lawyer_provider=lawyer_provider,
        supervisor_provider=supervisor_provider,
        runs_root=runs_root,
        memory_store=memory_store,
        note_provider=note_provider,
        note_model="stub-model",
    )

    assert result["supervisor_verdict"]["verdict"] == "pass"

    notes_dir = tmp_path / "memory" / "agent_notes"
    note_files = list(notes_dir.glob("*.md"))
    assert len(note_files) == 0, f"Expected no note files for 'pass', found: {note_files}"


@pytest.mark.asyncio
async def test_reject_without_memory_store_no_exception(tmp_path):
    """verdict='reject' but memory_store=None → no exception raised, no note written."""
    runs_root = tmp_path / "runs"

    lawyer_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode": "consultation", "primary_answer": "不完整的答复"}',
            finish_reason="end_turn",
        ),
    ])
    supervisor_provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"verdict": "reject", "confidence": 0.8, "issues": ["missing law refs"]}',
            finish_reason="end_turn",
        ),
    ])

    # memory_store=None (default) — no note_provider needed either
    result = await run_with_supervisor(
        query="劳动合同问题",
        lawyer_factory=_make_lawyer_factory(),
        supervisor_factory=_make_supervisor_factory(),
        lawyer_provider=lawyer_provider,
        supervisor_provider=supervisor_provider,
        runs_root=runs_root,
        # memory_store not provided → defaults to None
    )

    assert result["supervisor_verdict"]["verdict"] == "reject"
    # No exception was raised — test passes by completing without error
