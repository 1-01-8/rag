import pytest
import respx
import httpx
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.base import LLMResponse, ToolSpec
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key", default_model="claude-sonnet-4-6")


_BASE_URL = "https://api.anthropic.com/v1/messages"


def _mock_message_response(content_blocks: list[dict], stop_reason: str = "end_turn"):
    """Build a fake Anthropic /v1/messages response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": "claude-sonnet-4-6",
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_complete_text_response(provider, tmp_run_dir):
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_mock_message_response(
                [{"type": "text", "text": "Hello from Claude"}], "end_turn",
            ),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="hi")],
        model="claude-sonnet-4-6",
        recorder=rec, agent_name="tester",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert resp.text == "Hello from Claude"
    assert resp.finish_reason == "end_turn"
    assert resp.usage.input_tokens == 10


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_complete_tool_use(provider, tmp_run_dir):
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_mock_message_response(
                [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "statute_search",
                        "input": {"query": "民法典 510"},
                    },
                ],
                "tool_use",
            ),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="search")],
        model="claude-sonnet-4-6",
        tools=[ToolSpec(
            name="statute_search", description="search",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )],
        recorder=rec, agent_name="tester",
    )
    rec.close()
    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "statute_search"
    assert resp.tool_calls[0].args == {"query": "民法典 510"}
    assert resp.tool_calls[0].tool_use_id == "toolu_01"


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_marks_system_for_cache(provider, tmp_run_dir):
    """The system message should be sent with cache_control: ephemeral."""
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json as _j
        captured["body"] = _j.loads(request.content)
        return httpx.Response(200, json=_mock_message_response(
            [{"type": "text", "text": "ok"}], "end_turn",
        ))

    respx.post(_BASE_URL).mock(side_effect=_capture)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    await provider.complete(
        messages=[
            AgentMessage(role="system", content="You are a legal assistant."),
            AgentMessage(role="user", content="hi"),
        ],
        model="claude-sonnet-4-6",
        recorder=rec, agent_name="tester",
    )
    rec.close()

    sys_field = captured["body"]["system"]
    assert isinstance(sys_field, list)
    assert sys_field[-1].get("cache_control") == {"type": "ephemeral"}
