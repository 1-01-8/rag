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
    timeout_seconds: int = 60
    tools: list[Tool] = Field(default_factory=list)

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
        import asyncio

        tools_by_name = {t.name: t for t in self.tools}
        messages: list[AgentMessage] = [
            AgentMessage(role="system", content=self.system_prompt()),
            AgentMessage(role="user", content=str(input.payload.get("query", input.payload))),
        ]

        tool_specs = [t.to_spec() for t in self.tools] if self.tools else None

        for step in range(1, self.max_steps + 1):
            response = await self.provider.complete(
                messages=messages,
                model=getattr(self.provider, "default_model", "stub-1"),
                tools=tool_specs,
                response_format=self.output_schema(),
                recorder=self.recorder,
                agent_name=self.name,
            )

            if response.tool_calls:
                if len(response.tool_calls) > self.max_tool_calls:
                    from multi_agent.errors import BudgetExceeded
                    raise BudgetExceeded(self.name, "max_tool_calls", self.max_tool_calls)
                # Fan-out: dispatch all tool calls concurrently
                results = await asyncio.gather(*[
                    self._dispatch_tool(tc, tools_by_name) for tc in response.tool_calls
                ], return_exceptions=True)
                for tc, result in zip(response.tool_calls, results):
                    if isinstance(result, Exception):
                        result = self._wrap_tool_exception(tc, result)
                    messages.append(self._tool_result_message(tc, result))
                continue

            # No tool calls → expect final answer
            schema = self.output_schema()
            parsed_dict = parse_json_robust(response.text)
            parsed = schema.model_validate(parsed_dict)
            return AgentOutput(payload=parsed, steps_used=step)

        from multi_agent.errors import BudgetExceeded
        raise BudgetExceeded(self.name, "max_steps", self.max_steps)

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
