"""SecretaryAgent — research + business tools delegated by Lawyer.

Wrapped via SecretaryAsTool for Agent-as-Tool dispatch (ADR-05).
"""
from __future__ import annotations
from importlib.resources import files
from typing import Any
from pydantic import BaseModel, Field

from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.schemas.evidence import Evidence
from multi_agent.schemas.messages import ToolResult
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class SecretaryResponse(BaseModel):
    summary: str
    evidences: list[Evidence] = Field(default_factory=list)
    notes: str = ""
    confidence: float = 0.0


class SecretaryAgent(BaseAgent):
    """Research agent. Uses retrievers + business tools."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.secretary").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[SecretaryResponse]:
        return SecretaryResponse


class SecretaryRequest(BaseModel):
    task: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SecretaryAsTool(Tool):
    name: str = "ask_secretary"
    description: str = (
        "Ask the secretary to do research (statute/case retrieval) or "
        "business work (contract review / doc generation / doc interpret). "
        "Pass a task description and any relevant context in payload."
    )
    args_schema: type[BaseModel] = SecretaryRequest

    secretary_agent: Any                # SecretaryAgent

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: SecretaryRequest, recorder: Recorder) -> ToolResult:
        try:
            input = AgentInput(payload={
                "query": args.task,
                **args.payload,
            })
            output = await self.secretary_agent.run(input)
            return ToolResult(
                tool_use_id="",
                payload=output.payload.model_dump(),
            )
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=str(e))
