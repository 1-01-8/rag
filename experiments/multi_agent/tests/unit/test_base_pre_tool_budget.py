import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tools.base import Tool
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import BudgetExceeded


class _Args(BaseModel):
    q: str


class _Tool(Tool):
    name: str = "echo"
    description: str = "echo"
    args_schema: type[BaseModel] = _Args

    async def call(self, args, recorder):
        return ToolResult(tool_use_id="x", payload={"echo": args.q})


class _Out(BaseModel):
    a: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_pre_tool_rejection_budget_fires(tmp_run_dir):
    """If model keeps answering without calling tools, BudgetExceeded should fire
    on max_pre_tool_rejections — not silently loop until max_steps."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"a": "fake"}', finish_reason="end_turn")
        for _ in range(5)
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   tools=[_Tool()], max_pre_tool_rejections=2, max_steps=10)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert exc.value.budget == "max_pre_tool_rejections"
    assert exc.value.limit == 2


@pytest.mark.asyncio
async def test_no_tools_no_rejection_budget(tmp_run_dir):
    """Agent with no tools accepts direct answer (no budget applies)."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"a": "ok"}', finish_reason="end_turn"),
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   max_pre_tool_rejections=2)
    out = await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert out.payload.a == "ok"
