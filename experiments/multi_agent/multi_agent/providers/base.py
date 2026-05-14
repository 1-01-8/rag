from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal
from pydantic import BaseModel, Field

from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


class ToolSpec(BaseModel):
    """Lightweight tool definition for LLM tool-use APIs."""
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    parsed: Any | None = None  # Pydantic model instance if response_format used
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal"] = "end_turn"


class StreamChunk(BaseModel):
    kind: Literal["token", "tool_call_start", "tool_call_args", "end_turn", "error"]
    content: str = ""
    tool_use_id: str | None = None
    tool_name: str | None = None


class LLMProvider(ABC):
    """All concrete providers (Anthropic, OpenAI-compatible) implement this."""

    @abstractmethod
    async def complete(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_stream(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]: ...
