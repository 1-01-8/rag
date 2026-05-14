from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder


class ToolSpec(BaseModel):
    """JSON-schema-style tool description shown to LLM."""
    name: str
    description: str
    input_schema: dict


class Tool(BaseModel, ABC):
    name: str
    description: str
    args_schema: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    async def call(self, args: BaseModel, recorder: Recorder) -> ToolResult: ...

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_schema.model_json_schema(),
        )
