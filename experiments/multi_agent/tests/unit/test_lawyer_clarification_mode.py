"""Phase 5af: LawyerOutput clarification mode + Supervisor auto-pass."""
from __future__ import annotations
import pytest
from multi_agent.schemas.lawyer import LawyerOutput, FiveSection


def test_lawyer_output_clarification_mode_validates():
    out = LawyerOutput(
        mode="clarification",
        primary_answer="需要更多信息",
        clarifying_questions=["你是出手方?", "有受伤吗?"],
    )
    assert out.mode == "clarification"
    assert out.five_section is None
    assert out.citations == []
    assert len(out.clarifying_questions) == 2


def test_lawyer_output_consultation_mode_still_works():
    out = LawyerOutput(
        mode="consultation",
        primary_answer="不合法",
        citations=[],
        five_section=FiveSection(
            dispute_analysis="x",
            applicable_laws="x",
            similar_cases="x",
            remedy_suggestions="x",
            risk_assessment="x",
        ),
    )
    assert out.mode == "consultation"
    assert out.five_section is not None


def test_lawyer_output_default_mode_is_consultation():
    """Backward compat: existing code that constructs LawyerOutput without mode."""
    out = LawyerOutput(
        primary_answer="answer",
        five_section=FiveSection(
            dispute_analysis="x",
            applicable_laws="x",
            similar_cases="x",
            remedy_suggestions="x",
            risk_assessment="x",
        ),
    )
    assert out.mode == "consultation"


def test_lawyer_output_clarification_empty_questions():
    """clarification mode with no questions is still valid (prompt enforces >=1)."""
    out = LawyerOutput(
        mode="clarification",
        primary_answer="信息不足",
    )
    assert out.mode == "clarification"
    assert out.clarifying_questions == []
    assert out.five_section is None


def test_lawyer_output_rejects_unknown_mode():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LawyerOutput(mode="bogus_mode", primary_answer="")


def test_clarifying_questions_default_empty_for_consultation():
    """consultation mode: clarifying_questions defaults to empty list."""
    out = LawyerOutput(
        mode="consultation",
        primary_answer="ok",
        five_section=FiveSection(
            dispute_analysis="x", applicable_laws="x", similar_cases="x",
            remedy_suggestions="x", risk_assessment="x",
        ),
    )
    assert out.clarifying_questions == []


@pytest.mark.asyncio
async def test_supervisor_skips_llm_on_clarification(tmp_path):
    """run_with_supervisor short-circuits when lawyer outputs clarification mode."""
    import json
    from unittest.mock import AsyncMock, MagicMock
    from multi_agent.orchestration.supervised import run_with_supervisor

    clarification_json = json.dumps({
        "mode": "clarification",
        "primary_answer": "需要更多信息",
        "citations": [],
        "five_section": None,
        "clarifying_questions": ["你是出手方?", "有受伤吗?"],
    })

    # Lawyer provider returns clarification JSON
    lawyer_provider = MagicMock()
    lawyer_provider.complete = AsyncMock()

    # Supervisor provider should never be called
    supervisor_provider = MagicMock()
    supervisor_provider.complete = AsyncMock()

    # Minimal agent stubs
    def make_lawyer_factory():
        from multi_agent.agents.base import AgentOutput
        from multi_agent.schemas.messages import AgentMessage

        class _StubLawyer:
            working_memory = None

            async def run(self, inp):
                out = MagicMock()
                out.payload = MagicMock()
                out.payload.model_dump = lambda: {}
                return out

        def factory(p, r):
            agent = _StubLawyer()
            return agent

        return factory

    # We'll patch run_query to return clarification directly
    from unittest.mock import patch

    fake_run_id = "r_test_clarification_0000000001"
    fake_lawyer_result = {
        "run_id": fake_run_id,
        "status": "ok",
        "final_answer": clarification_json,
    }

    with patch(
        "multi_agent.orchestration.supervised.run_query",
        new=AsyncMock(return_value=fake_lawyer_result),
    ):
        result = await run_with_supervisor(
            query="我和别人打架了",
            lawyer_factory=make_lawyer_factory(),
            supervisor_factory=lambda p, r: None,  # must never be called
            lawyer_provider=lawyer_provider,
            supervisor_provider=supervisor_provider,
            runs_root=tmp_path,
        )

    # Supervisor LLM was never invoked
    supervisor_provider.complete.assert_not_called()

    # Result structure is correct
    assert result["supervisor_verdict"]["verdict"] == "pass"
    assert result["supervisor_verdict"]["confidence"] == 1.0
    assert result["supervisor_verdict"]["issues"] == []
    assert result["lawyer_run_id"] == fake_run_id
