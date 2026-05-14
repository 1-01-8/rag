from __future__ import annotations
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent


class EchoStubOutput(BaseModel):
    echoed: str


class EchoStubAgent(BaseAgent):
    """Minimal concrete agent for E2E walking-skeleton test.

    Expects provider to return a JSON object with key 'echoed'.
    No tools. No multi-step reasoning. Just shape-checks the full pipeline.
    """

    def system_prompt(self) -> str:
        return (
            "You are an echo agent. Echo the user's message back inside "
            'a JSON object: {"echoed": "<message>"}. Do not add anything else.'
        )

    def output_schema(self):
        return EchoStubOutput
