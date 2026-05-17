"""Phase 6g: BaseAgent tool-first 给 clarification mode 留出路.

实测场景: 用户问 "我要根据这个法律起诉他", 信息严重不足. 模型应该走
clarification, 而不是被 tool-first 反复 reject 直到 BudgetExceeded.
"""
from __future__ import annotations
import pytest
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal

from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.base import ToolSpec
from multi_agent.tools.base import Tool
from multi_agent.schemas.messages import ToolResult


class _Out(BaseModel):
    mode: Literal["consultation", "clarification"] = "consultation"
    primary_answer: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)


class _DummyArgs(BaseModel):
    query: str


class _DummyTool(Tool):
    name: str = "dummy_search"
    description: str = "test"
    args_schema: type[BaseModel] = _DummyArgs

    async def call(self, args, recorder):
        return ToolResult(tool_use_id="", payload={"evidences": []})


class _TestAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"

    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_clarification_bypasses_tool_first_enforcement(tmp_path):
    """模型第 1 轮不调 tool, 但输出合法 clarification JSON → 直接通过.

    之前会被 reject 然后再次反问, 撞 max_pre_tool_rejections.
    """
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    provider = StubProvider(responses=[
        ScriptedResponse(
            text='{"mode": "clarification", "primary_answer": "信息不足",'
                 '"clarifying_questions": ["你要起诉什么?", "对方是谁?"]}',
            finish_reason="end_turn",
        ),
    ])
    agent = _TestAgent(
        name="lawyer", role="advisor",
        provider=provider, recorder=rec,
        tools=[_DummyTool()],   # 有 tool 但模型没调
        model="stub-1",
        max_pre_tool_rejections=2,
    )
    output = await agent.run(AgentInput(payload={"query": "我要起诉"}))
    rec.close()

    # 1 步就通过, 没触发 reject
    assert output.steps_used == 1
    assert output.payload.mode == "clarification"
    assert len(output.payload.clarifying_questions) == 2


@pytest.mark.asyncio
async def test_empty_clarifying_questions_does_not_bypass(tmp_path):
    """clarification 模式但 clarifying_questions 为空 → 不算合法, 走 reject 路径."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    provider = StubProvider(responses=[
        # 第 1 次: 空 questions 数组, 想钻空子
        ScriptedResponse(
            text='{"mode": "clarification", "clarifying_questions": []}',
            finish_reason="end_turn",
        ),
        # 第 2 次: 又是空, 没意义
        ScriptedResponse(
            text='{"mode": "clarification", "clarifying_questions": []}',
            finish_reason="end_turn",
        ),
        # 第 3 次: 还是空
        ScriptedResponse(
            text='{"mode": "clarification", "clarifying_questions": []}',
            finish_reason="end_turn",
        ),
    ])
    agent = _TestAgent(
        name="lawyer", role="advisor",
        provider=provider, recorder=rec,
        tools=[_DummyTool()],
        model="stub-1",
        max_pre_tool_rejections=2,
    )
    from multi_agent.errors import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        await agent.run(AgentInput(payload={"query": "q"}))
    rec.close()


@pytest.mark.asyncio
async def test_plain_text_refusal_still_triggers_reject(tmp_path):
    """模型输出纯文本反问 (非 JSON) → tool-first 仍 reject. 防回归."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    # 3 次都是纯文本反问 (像我们 trace 里实测的)
    provider = StubProvider(responses=[
        ScriptedResponse(text="请告诉我具体起诉什么...", finish_reason="end_turn"),
        ScriptedResponse(text="请提供更多信息...", finish_reason="end_turn"),
        ScriptedResponse(text="我需要先了解...", finish_reason="end_turn"),
    ])
    agent = _TestAgent(
        name="lawyer", role="advisor",
        provider=provider, recorder=rec,
        tools=[_DummyTool()],
        model="stub-1",
        max_pre_tool_rejections=2,
    )
    from multi_agent.errors import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        await agent.run(AgentInput(payload={"query": "q"}))
    rec.close()
