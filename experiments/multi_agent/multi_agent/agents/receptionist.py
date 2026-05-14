"""ReceptionistAgent — triage + multi-issue decomposition (spec §3.5)."""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.receptionist import ReceptionistOutput


class ReceptionistAgent(BaseAgent):
    """Tool-less classifier. Reads user query, outputs ReceptionistOutput."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.receptionist").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[ReceptionistOutput]:
        return ReceptionistOutput
