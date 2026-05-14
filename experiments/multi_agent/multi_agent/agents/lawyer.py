"""LawyerAgent — real consultation agent with five-section prompt.

One class, runtime-selected specialty. The system prompt is built from
the shared skeleton + specialty markdown file.
"""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.lawyer import LawyerOutput


_VALID_SPECIALTIES: tuple[str, ...] = ("通用", "民事", "劳动", "交通", "婚姻", "房产")


class LawyerAgent(BaseAgent):
    """Consultation agent. ReAct over statute_search / read_article tools."""

    specialty: str = "通用"

    def model_post_init(self, __context) -> None:
        if self.specialty not in _VALID_SPECIALTIES:
            raise ValueError(
                f"unknown specialty: {self.specialty!r}. "
                f"Choices: {list(_VALID_SPECIALTIES)}"
            )

    def system_prompt(self) -> str:
        """Concatenate _five_section_skeleton.md + specialty_<name>.md."""
        prompts_pkg = files("multi_agent.prompts.lawyer")
        skeleton = prompts_pkg.joinpath("_five_section_skeleton.md").read_text(encoding="utf-8")
        specialty_md = prompts_pkg.joinpath(f"specialty_{self.specialty}.md").read_text(encoding="utf-8")
        return f"{skeleton}\n\n---\n\n{specialty_md}"

    def output_schema(self) -> type[LawyerOutput]:
        return LawyerOutput

    def _render_input(self, input) -> str:
        """If sub_cases present (multi-issue), inject them as a numbered list."""
        payload = input.payload
        query = str(payload.get("query", ""))
        sub_cases = payload.get("sub_cases", [])
        if not sub_cases:
            return query
        lines = [f"用户咨询: {query}", "", "本案包含以下独立子议题(请逐一回答):"]
        for i, sc in enumerate(sub_cases, 1):
            issue = sc.get("issue", "") if isinstance(sc, dict) else sc.issue
            specialty = sc.get("specialty", "") if isinstance(sc, dict) else sc.specialty
            lines.append(f"{i}. [{specialty}] {issue}")
        return "\n".join(lines)
