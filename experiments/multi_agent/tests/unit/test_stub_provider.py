import pytest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.providers.base import LLMResponse, Usage
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_stub_returns_scripted_text(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[
        ScriptedResponse(text="hello world", finish_reason="end_turn"),
    ])
    resp = await p.complete(
        messages=[AgentMessage(role="user", content="hi")],
        model="stub-1", recorder=rec, agent_name="lawyer",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"
    assert resp.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_stub_emits_llm_events(tmp_run_dir):
    import json as _json
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="x")])
    await p.complete(messages=[AgentMessage(role="user", content="hi")],
                     model="stub-1", recorder=rec, agent_name="lawyer")
    rec.close()
    types = [_json.loads(l)["event_type"]
             for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    assert "LLMRequested" in types
    assert "LLMResponded" in types


@pytest.mark.asyncio
async def test_stub_returns_tool_calls(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(
        text="",
        tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "hi"})],
        finish_reason="tool_use",
    )])
    resp = await p.complete(messages=[AgentMessage(role="user", content="x")],
                            model="stub-1", recorder=rec, agent_name="lawyer")
    rec.close()
    assert resp.finish_reason == "tool_use"
    assert resp.tool_calls[0].tool_name == "echo"


@pytest.mark.asyncio
async def test_stub_exhausted_raises(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="x")])
    await p.complete(messages=[], model="m", recorder=rec, agent_name="a")
    with pytest.raises(RuntimeError, match="exhausted"):
        await p.complete(messages=[], model="m", recorder=rec, agent_name="a")
    rec.close()
