"""run_stream / stream_one_turn smoke tests against StubProvider (no network).
Real-provider streaming is exercised in tests/integration/test_qwen_e2e.py.
"""
import pytest
from multi_agent.agents.base import BaseAgent, AgentInput, StreamEvent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from pydantic import BaseModel


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_run_stream_yields_final_answer(tmp_run_dir):
    """Phase 1 contract preserved: run_stream yields agent_start, final_answer, agent_end."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text='{"answer": "ok"}')])
    agent = _Agent(name="a", role="t", provider=p, recorder=rec, model="stub-1")
    events = []
    async for ev in agent.run_stream(AgentInput(payload={"query": "hi"})):
        events.append(ev)
    rec.close()
    kinds = [e.kind for e in events]
    assert "agent_start" in kinds
    assert "final_answer" in kinds
    assert "agent_end" in kinds


@pytest.mark.asyncio
async def test_stream_one_turn_yields_tokens(tmp_run_dir):
    """stream_one_turn calls provider.complete_stream and yields llm_token events.
    StubProvider yields one StreamChunk per character of the scripted text."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="hello")])
    agent = _Agent(name="a", role="t", provider=p, recorder=rec, model="stub-1")
    events = []
    async for ev in agent.stream_one_turn("say hi"):
        events.append(ev)
    rec.close()
    tokens = [e.content for e in events if e.kind == "llm_token"]
    assert "".join(tokens) == "hello"
