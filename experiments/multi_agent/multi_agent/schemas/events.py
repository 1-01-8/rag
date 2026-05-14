from __future__ import annotations
from datetime import datetime
from typing import Literal, Any
from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Common fields for every trace event."""
    event_id: str
    run_id: str
    timestamp: datetime
    parent_id: str | None = None
    event_type: str  # subclasses override with Literal

    model_config = {"frozen": False, "extra": "forbid"}


class RunStarted(BaseEvent):
    event_type: Literal["RunStarted"] = "RunStarted"
    query: str
    config: dict[str, Any] = Field(default_factory=dict)


class RunFinished(BaseEvent):
    event_type: Literal["RunFinished"] = "RunFinished"
    status: Literal["ok", "error", "timeout"]
    final_answer: str | None = None
    error: str | None = None


# --- Agent events ---

class AgentInvoked(BaseEvent):
    event_type: Literal["AgentInvoked"] = "AgentInvoked"
    agent_name: str
    role: str
    input: dict[str, Any] = Field(default_factory=dict)


class AgentResponded(BaseEvent):
    event_type: Literal["AgentResponded"] = "AgentResponded"
    agent_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int


# --- LLM events ---

class LLMRequested(BaseEvent):
    event_type: Literal["LLMRequested"] = "LLMRequested"
    provider: str
    model: str
    messages: list[dict[str, Any]]
    params: dict[str, Any] = Field(default_factory=dict)


class LLMResponded(BaseEvent):
    event_type: Literal["LLMResponded"] = "LLMResponded"
    raw_response: str
    usage: dict[str, int] = Field(default_factory=dict)
    duration_ms: int
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal"]


# --- Tool events ---

class ToolCalled(BaseEvent):
    event_type: Literal["ToolCalled"] = "ToolCalled"
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    agent_name: str


class ToolReturned(BaseEvent):
    event_type: Literal["ToolReturned"] = "ToolReturned"
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int


# --- Memory events ---

class MemoryRead(BaseEvent):
    event_type: Literal["MemoryRead"] = "MemoryRead"
    target: Literal["sticky", "turn", "agent_notes", "user_history"]
    query: dict[str, Any] = Field(default_factory=dict)
    hits: list[dict[str, Any]] = Field(default_factory=list)
    agent_name: str


class MemoryWritten(BaseEvent):
    event_type: Literal["MemoryWritten"] = "MemoryWritten"
    target: str
    payload: dict[str, Any]
    path: str
    agent_name: str


# --- Supervisor ---

class SupervisorVerdict(BaseEvent):
    event_type: Literal["SupervisorVerdict"] = "SupervisorVerdict"
    verdict: Literal["pass", "revise", "reject"]
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None


from typing import Annotated, Union
from pydantic import TypeAdapter, Field as PydField


AnyEvent = Annotated[
    Union[
        RunStarted, RunFinished,
        AgentInvoked, AgentResponded,
        LLMRequested, LLMResponded,
        ToolCalled, ToolReturned,
        MemoryRead, MemoryWritten,
        SupervisorVerdict,
    ],
    PydField(discriminator="event_type"),
]

_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def event_from_dict(raw: dict) -> AnyEvent:
    """Parse a dict into the correct event subclass via event_type discriminator."""
    return _event_adapter.validate_python(raw)
