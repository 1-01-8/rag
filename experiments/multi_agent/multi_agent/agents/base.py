from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

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
