import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput, AgentOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


class _EchoOutput(BaseModel):
    answer: str


class _DummyAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "you are a test agent"

    def output_schema(self):
        return _EchoOutput


@pytest.mark.asyncio
async def test_agent_construction(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text='{"answer": "hi"}')])
    agent = _DummyAgent(
        name="dummy", role="test",
        provider=p, recorder=rec,
    )
    rec.close()
    assert agent.name == "dummy"
    assert agent.max_steps == 10  # default


from pydantic import BaseModel as _BM
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.messages import ToolCallRequest, ToolResult
from multi_agent.tools.base import Tool


class _EchoArgs(_BM):
    msg: str


class _EchoTool(Tool):
    name: str = "echo"
    description: str = "echo a message"
    args_schema: type = _EchoArgs

    async def call(self, args, recorder):
        return ToolResult(tool_use_id="x", payload={"echo": args.msg})


@pytest.mark.asyncio
async def test_agent_runs_to_final_answer(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "done"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider, recorder=rec)
    out = await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert isinstance(out.payload, _EchoOutput)
    assert out.payload.answer == "done"
    assert out.steps_used == 1


@pytest.mark.asyncio
async def test_agent_dispatches_tool_then_answers(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "x"})],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "after tool"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider,
                        recorder=rec, tools=[_EchoTool()])
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert out.payload.answer == "after tool"
    assert out.steps_used == 2


@pytest.mark.asyncio
async def test_fan_out_parallel_tools(tmp_run_dir):
    """Two tool calls in one LLM response are dispatched concurrently."""
    import json as _j
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[
                ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "a"}),
                ToolCallRequest(tool_use_id="t2", tool_name="echo", args={"msg": "b"}),
            ],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider,
                        recorder=rec, tools=[_EchoTool()])
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    # Two ToolCalled events should share the same parent span
    lines = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    tool_calls = [l for l in lines if l["event_type"] == "ToolCalled"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["parent_id"] == tool_calls[1]["parent_id"]


from multi_agent.errors import BudgetExceeded


@pytest.mark.asyncio
async def test_exceeding_max_steps_raises_budget(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    # All responses request more tool calls — agent never finalizes
    provider = StubProvider(responses=[
        ScriptedResponse(
            tool_calls=[ToolCallRequest(tool_use_id=f"t{i}", tool_name="echo", args={"msg": "x"})],
            finish_reason="tool_use",
        )
        for i in range(5)
    ])
    agent = _DummyAgent(name="dummy", role="t",
                        provider=provider, recorder=rec,
                        tools=[_EchoTool()], max_steps=3)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert exc.value.budget == "max_steps"
    assert exc.value.limit == 3


@pytest.mark.asyncio
async def test_exceeding_max_tool_calls_raises_budget(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    # Single LLM response with 5 tool calls; agent allows only 3
    provider = StubProvider(responses=[
        ScriptedResponse(
            tool_calls=[
                ToolCallRequest(tool_use_id=f"t{i}", tool_name="echo", args={"msg": str(i)})
                for i in range(5)
            ],
            finish_reason="tool_use",
        ),
    ])
    agent = _DummyAgent(name="dummy", role="t",
                        provider=provider, recorder=rec,
                        tools=[_EchoTool()], max_tool_calls=3)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert exc.value.budget == "max_tool_calls"


from multi_agent.agents.base import StreamEvent


@pytest.mark.asyncio
async def test_run_stream_yields_tokens_and_final(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "done"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider, recorder=rec)

    collected: list[StreamEvent] = []
    async for ev in agent.run_stream(AgentInput(payload={"query": "hi"})):
        collected.append(ev)
    rec.close()

    kinds = [e.kind for e in collected]
    assert "agent_start" in kinds
    assert "agent_end" in kinds
    assert "final_answer" in kinds
