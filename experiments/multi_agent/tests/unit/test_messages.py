from multi_agent.schemas.messages import (
    AgentMessage, ToolCallRequest, ToolResult,
)


def test_agent_message_basic():
    m = AgentMessage(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_agent_message_with_tool_calls():
    m = AgentMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCallRequest(tool_use_id="t1", tool_name="search", args={"q": "x"})
        ],
    )
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].tool_name == "search"


def test_tool_result_payload_or_error():
    ok = ToolResult(tool_use_id="t1", payload={"hits": 3}, error=None)
    err = ToolResult(tool_use_id="t2", payload=None, error="boom")
    assert ok.payload == {"hits": 3}
    assert err.error == "boom"
