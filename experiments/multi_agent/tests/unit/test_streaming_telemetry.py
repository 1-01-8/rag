"""Streaming must capture usage telemetry — required by Phase 3 budget gates."""
import json as _j
import pytest
import respx
import httpx
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="test-key")


_BASE_URL = "https://api.anthropic.com/v1/messages"


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_streaming_records_usage(anthropic_provider, tmp_run_dir):
    """The LLMResponded event from streaming must record token usage."""
    sse_body = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        'event: content_block_stop\n'
        'data: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    )
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode(),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    chunks = []
    async for ch in anthropic_provider.complete_stream(
        messages=[AgentMessage(role="user", content="hi")],
        model="claude-sonnet-4-6", recorder=rec, agent_name="t",
    ):
        chunks.append(ch)
    rec.close()

    events = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(responded) == 1
    usage = responded[0]["usage"]
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 4


@pytest.mark.asyncio
async def test_openai_compat_streaming_records_usage(tmp_run_dir):
    """Real Qwen — streaming should fetch usage via stream_options."""
    import httpx as _httpx
    try:
        with _httpx.Client(timeout=2.0) as c:
            assert c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        pytest.skip("vLLM not running")

    provider = OpenAICompatibleProvider()
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    async for _ in provider.complete_stream(
        messages=[AgentMessage(role="user", content="say one word")],
        model="qwen3.5-9b", max_tokens=8, temperature=0,
        recorder=rec, agent_name="t",
    ):
        pass
    rec.close()

    events = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(responded) == 1
    usage = responded[0]["usage"]
    assert usage.get("input_tokens", 0) > 0
    assert usage.get("output_tokens", 0) >= 1
