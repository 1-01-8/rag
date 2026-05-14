import pytest
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.base import LLMResponse, Usage
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def provider():
    return OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        default_model="qwen3.5-9b",
    )


@pytest.mark.asyncio
async def test_complete_returns_llm_response(provider, tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="say ping (one word)")],
        model="qwen3.5-9b",
        max_tokens=8,
        temperature=0,
        recorder=rec,
        agent_name="tester",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert isinstance(resp.text, str)
    assert len(resp.text) > 0
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    assert resp.finish_reason in ("end_turn", "max_tokens")


@pytest.mark.asyncio
async def test_complete_emits_llm_events(provider, tmp_run_dir):
    import json as _j
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    await provider.complete(
        messages=[AgentMessage(role="user", content="say one word")],
        model="qwen3.5-9b", max_tokens=8, temperature=0,
        recorder=rec, agent_name="tester",
    )
    rec.close()
    types = [_j.loads(l)["event_type"]
             for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    assert "LLMRequested" in types
    assert "LLMResponded" in types
