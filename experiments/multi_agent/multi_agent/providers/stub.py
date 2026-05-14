from __future__ import annotations
from typing import AsyncGenerator
from pydantic import BaseModel, Field

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


class ScriptedResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = "end_turn"
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0


class StubProvider(LLMProvider):
    """Returns pre-scripted responses for testing. No network."""

    def __init__(self, responses: list[ScriptedResponse]):
        self._responses = list(responses)
        self._idx = 0

    async def complete(
        self, messages, *, model, tools=None, response_format=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> LLMResponse:
        if self._idx >= len(self._responses):
            raise RuntimeError("StubProvider scripted responses exhausted")
        scripted = self._responses[self._idx]
        self._idx += 1

        with recorder.span(
            "llm_call",
            provider="stub", model=model, agent_name=agent_name,
            messages=[m.model_dump() for m in messages],
            params={"max_tokens": max_tokens, "temperature": temperature},
        ) as span:
            resp = LLMResponse(
                text=scripted.text,
                tool_calls=scripted.tool_calls,
                usage=Usage(input_tokens=scripted.usage_input_tokens,
                            output_tokens=scripted.usage_output_tokens),
                duration_ms=0,
                finish_reason=scripted.finish_reason,  # type: ignore[arg-type]
                raw={"scripted": True},
            )
            span.set_output({"raw": scripted.text,
                             "usage": resp.usage.model_dump(),
                             "finish_reason": scripted.finish_reason})
            return resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        if self._idx >= len(self._responses):
            raise RuntimeError("StubProvider scripted responses exhausted")
        scripted = self._responses[self._idx]
        self._idx += 1
        for ch in scripted.text:
            yield StreamChunk(kind="token", content=ch)
        for tc in scripted.tool_calls:
            yield StreamChunk(kind="tool_call_start",
                              tool_use_id=tc.tool_use_id, tool_name=tc.tool_name)
        yield StreamChunk(kind="end_turn")
