from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """LLM-issued request to invoke a tool."""
    tool_use_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of dispatching a tool. Exactly one of payload/error is set."""
    tool_use_id: str
    payload: dict[str, Any] | None = None
    error: str | None = None


class AgentMessage(BaseModel):
    """A single message in the agent's conversation context."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_use_id: str | None = None  # set when role == "tool"
