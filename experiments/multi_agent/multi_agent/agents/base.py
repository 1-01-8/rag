from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal
from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    kind: Literal["agent_start", "agent_end", "llm_token", "tool_start", "tool_end", "final_answer", "error"]
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

from multi_agent.providers.base import LLMProvider
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.schemas.working_memory import WorkingMemory


class AgentInput(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    payload: BaseModel
    steps_used: int

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent(BaseModel, ABC):
    """Template-method base. Subclasses only override system_prompt() and output_schema()."""
    name: str
    role: str
    provider: LLMProvider
    recorder: Recorder
    max_steps: int = 10
    max_total_tokens: int = 20_000
    max_tool_calls: int = 8
    max_pre_tool_rejections: int = 2
    timeout_seconds: int = 60
    tools: list[Tool] = Field(default_factory=list)
    model: str = ""    # set explicitly by ProviderProfile; falls back to provider default
    working_memory: WorkingMemory | None = None

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def output_schema(self) -> type[BaseModel]: ...

    async def run(self, input: AgentInput) -> AgentOutput:
        """Template method. Subclasses do not override."""
        with self.recorder.span(
            "agent_invoke", agent_name=self.name, role=self.role,
        ) as span:
            span.set_input(input.payload)
            output = await self._react_loop(input)
            span.set_output({"steps_used": output.steps_used})
            return output

    async def _react_loop(self, input: AgentInput) -> AgentOutput:
        from multi_agent.schemas.messages import AgentMessage
        from multi_agent.providers.json_robust import parse_json_robust
        from multi_agent.schemas.evidence import Evidence
        import asyncio

        if self.working_memory is None:
            object.__setattr__(self, "working_memory", WorkingMemory())

        tools_by_name = {t.name: t for t in self.tools}
        messages: list[AgentMessage] = [
            AgentMessage(role="system", content=self.system_prompt()),
            AgentMessage(role="user", content=self._render_input(input)),
        ]

        tool_specs = [t.to_spec() for t in self.tools] if self.tools else None
        tool_calls_made: int = 0  # track how many tool calls have occurred
        pre_tool_rejections: int = 0  # track how many times model skipped tool-first
        post_tool_reminder_added: bool = False  # inject citation-only reminder once

        for step in range(1, self.max_steps + 1):
            # Phase 6c: 单次 LLM 调用加 timeout — SiliconFlow 偶尔挂 100s+, 不让它无限等
            try:
                response = await asyncio.wait_for(
                    self.provider.complete(
                        messages=messages,
                        model=self.model or getattr(self.provider, "default_model", "stub-1"),
                        tools=tool_specs,
                        response_format=self.output_schema(),
                        recorder=self.recorder,
                        agent_name=self.name,
                    ),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError as e:
                from multi_agent.errors import ProviderUnavailable
                raise ProviderUnavailable(
                    f"LLM call timeout > {self.timeout_seconds}s (step {step})"
                ) from e

            if response.tool_calls:
                # Phase 6a fix: max_tool_calls 是 RUN 内累计, 不是单次响应
                # 防止模型每个 step 重复 search 直到 max_steps 超时
                next_total = tool_calls_made + len(response.tool_calls)
                if next_total > self.max_tool_calls:
                    from multi_agent.errors import BudgetExceeded
                    raise BudgetExceeded(self.name, "max_tool_calls", self.max_tool_calls)
                # Fan-out: dispatch all tool calls concurrently
                results = await asyncio.gather(*[
                    self._dispatch_tool(tc, tools_by_name) for tc in response.tool_calls
                ], return_exceptions=True)
                for tc, result in zip(response.tool_calls, results):
                    if isinstance(result, Exception):
                        result = self._wrap_tool_exception(tc, result)
                    # Harvest Evidences into working_memory (Phase 3b)
                    if result.payload:
                        evs = result.payload.get("evidences")
                        if isinstance(evs, list):
                            for ev_dict in evs:
                                try:
                                    self.working_memory.add_evidence(Evidence.model_validate(ev_dict))
                                except Exception:
                                    pass
                        # exact_read returns single evidence
                        ev_single = result.payload.get("evidence")
                        if isinstance(ev_single, dict):
                            try:
                                self.working_memory.add_evidence(Evidence.model_validate(ev_single))
                            except Exception:
                                pass
                    messages.append(self._tool_result_message(tc, result))
                tool_calls_made += len(response.tool_calls)
                # After the first batch of tool results, inject a one-time reminder
                # to restrict citations strictly to what the tools returned.
                if not post_tool_reminder_added:
                    messages.append(AgentMessage(
                        role="user",
                        content=(
                            "检索完成。现在请根据以上工具返回的结果撰写五段式 JSON 答案。\n"
                            "⚠️ citations 数组中只允许出现工具结果中出现的法条号，"
                            "严禁添加任何工具未返回的法条号。"
                        ),
                    ))
                    post_tool_reminder_added = True
                continue

            # No tool calls → 两步处理:
            #   1. 先看模型输出是否能 parse 成合法的 clarification mode JSON
            #      (Phase 5af schema): 信息不足时用户允许 Lawyer 反问澄清.
            #      若是 → 通过, 不算 tool-first violation.
            #   2. 否则走 tool-first enforcement (Phase 2d): reject 重定向.
            # Phase 6g 修复: 之前 BaseAgent 一刀切, 模型纯文本反问会被反复 reject
            # 直到 max_pre_tool_rejections 用完, 实测真实用户场景下损坏体验.
            if tool_specs and tool_calls_made == 0:
                # 尝试解析: 是 clarification JSON 吗?
                schema = self.output_schema()
                try:
                    parsed_dict = parse_json_robust(response.text)
                    parsed = schema.model_validate(parsed_dict)
                    mode = getattr(parsed, "mode", None)
                    clarifying_q = getattr(parsed, "clarifying_questions", [])
                    # clarification 通过条件: mode 字段是 "clarification" 且真有问题列表,
                    # 不能空数组钻空子;
                    if mode == "clarification" and clarifying_q:
                        return AgentOutput(payload=parsed, steps_used=step)
                except Exception:
                    pass  # parse 失败, 走 reject 路径

                pre_tool_rejections += 1
                if pre_tool_rejections > self.max_pre_tool_rejections:
                    from multi_agent.errors import BudgetExceeded
                    raise BudgetExceeded(self.name, "max_pre_tool_rejections", self.max_pre_tool_rejections)
                first_tool_name = self.tools[0].name
                messages.append(AgentMessage(
                    role="user",
                    content=(
                        "⚠️ 错误: 你的回答未通过检验, 已被丢弃.\n"
                        "你必须在两条路径中选一条:\n"
                        f"  (A) 信息充足 → 立即调用 {first_tool_name} 工具检索, 然后根据"
                        "工具返回结果撰写 mode=\"consultation\" 的 JSON.\n"
                        "  (B) 信息不足 → 输出 mode=\"clarification\" 的 JSON, 字段含"
                        "clarifying_questions (1-4 个具体澄清问题), citations=[], "
                        "five_section=null. 不要输出纯文本反问.\n"
                        "禁止使用训练记忆里的任何法条号. 必须立即在 (A) 或 (B) 中选择."
                    ),
                ))
                continue

            # Tool calls have been made (or no tools available) → accept final answer
            schema = self.output_schema()
            parsed_dict = parse_json_robust(response.text)
            parsed = schema.model_validate(parsed_dict)
            return AgentOutput(payload=parsed, steps_used=step)

        from multi_agent.errors import BudgetExceeded
        raise BudgetExceeded(self.name, "max_steps", self.max_steps)

    def _render_input(self, input: "AgentInput") -> str:
        """Render the AgentInput's payload into the user-message text.

        Default: return payload["query"] as string. Subclasses override
        to inject extra context (e.g. sub_cases for multi-issue).
        """
        return str(input.payload.get("query", input.payload))

    async def _dispatch_tool(self, tc, tools_by_name):
        from multi_agent.schemas.messages import ToolResult
        tool = tools_by_name.get(tc.tool_name)
        if tool is None:
            return ToolResult(tool_use_id=tc.tool_use_id, payload=None,
                              error=f"unknown tool: {tc.tool_name}")
        try:
            args = tool.args_schema.model_validate(tc.args)
        except Exception as e:
            return ToolResult(tool_use_id=tc.tool_use_id, payload=None,
                              error=f"args validation failed: {e}")
        with self.recorder.span(
            "tool_call", tool_name=tc.tool_name, args=tc.args, agent_name=self.name,
        ) as span:
            try:
                result = await tool.call(args, self.recorder)
                span.set_output(result.payload or {"error": result.error})
                # Force the tool_use_id from the LLM (Tool.call may have set its own)
                return result.model_copy(update={"tool_use_id": tc.tool_use_id})
            except Exception as e:
                return ToolResult(tool_use_id=tc.tool_use_id, payload=None, error=str(e))

    def _wrap_tool_exception(self, tc, exc):
        from multi_agent.schemas.messages import ToolResult
        return ToolResult(tool_use_id=tc.tool_use_id, payload=None, error=str(exc))

    def _tool_result_message(self, tc, result):
        from multi_agent.schemas.messages import AgentMessage
        import json as _j
        payload = result.payload if result.error is None else {"error": result.error}
        return AgentMessage(
            role="tool", content=_j.dumps(payload, ensure_ascii=False),
            tool_use_id=tc.tool_use_id,
        )

    async def run_stream(self, input: AgentInput) -> AsyncGenerator[StreamEvent, None]:
        """Stream version of run(). Yields high-level events for CLI/Web progress display.

        Phase 1: minimal — only agent_start / agent_end / final_answer / error.
        Token-level streaming requires provider.complete_stream() integration (Phase 2).
        """
        yield StreamEvent(kind="agent_start", content=self.name)
        try:
            output = await self.run(input)
            yield StreamEvent(
                kind="final_answer",
                content=output.payload.model_dump_json(),
                metadata={"steps_used": output.steps_used},
            )
        except Exception as e:
            yield StreamEvent(kind="error", content=str(e),
                              metadata={"type": type(e).__name__})
            raise
        finally:
            yield StreamEvent(kind="agent_end", content=self.name)

    async def stream_one_turn(
        self, user_input: str,
    ) -> AsyncGenerator["StreamEvent", None]:
        """Stream a single LLM turn without tool dispatch.

        Useful for CLI/SSE consumers that want token-level output but don't need
        the full ReAct loop. Tool dispatch + multi-turn ReAct still uses run() /
        run_stream() (which are non-streaming under the hood for now — full
        streaming-with-tools is a Phase 2c+ enhancement).
        """
        from multi_agent.schemas.messages import AgentMessage
        messages = [
            AgentMessage(role="system", content=self.system_prompt()),
            AgentMessage(role="user", content=user_input),
        ]
        model = self.model or getattr(self.provider, "default_model", "stub-1")
        async for chunk in self.provider.complete_stream(
            messages=messages, model=model,
            recorder=self.recorder, agent_name=self.name,
        ):
            if chunk.kind == "token":
                yield StreamEvent(kind="llm_token", content=chunk.content)
            elif chunk.kind == "end_turn":
                yield StreamEvent(kind="agent_end", content=self.name)
            elif chunk.kind == "error":
                yield StreamEvent(kind="error", content=chunk.content)
