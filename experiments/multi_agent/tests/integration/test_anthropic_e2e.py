# tests/integration/test_anthropic_e2e.py
"""Phase 2b optional acceptance test: real Anthropic API.
Skipped unless ANTHROPIC_API_KEY env var is set.
"""
import json
import os
import pytest
from pydantic import BaseModel

from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.agents.base import BaseAgent
from multi_agent.runner import run_query


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class _Out(BaseModel):
    answer: str


class _SimpleAgent(BaseAgent):
    def system_prompt(self) -> str:
        return 'Answer in JSON: {"answer": "<text>"}. Be very brief.'
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_real_anthropic_simple_completion(tmp_path):
    runs_root = tmp_path / "runs"
    provider = AnthropicProvider()
    result = await run_query(
        query="What is 1+1? Just the number.",
        agent_factory=lambda p, r: _SimpleAgent(
            name="claude_test", role="t",
            provider=p, recorder=r,
            model="claude-haiku-4-5-20251001",  # cheapest Claude
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "anthropic-haiku-smoke"},
    )
    assert result["status"] == "ok"
    final = json.loads(result["final_answer"])
    assert "2" in final["answer"]
