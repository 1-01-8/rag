"""ReceptionistAgent — triage + multi-issue decomposition (spec §3.5)."""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.receptionist import ReceptionistOutput


class ReceptionistAgent(BaseAgent):
    """Tool-less classifier. Reads user query (+ optional sticky_context)."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.receptionist").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[ReceptionistOutput]:
        return ReceptionistOutput

    def _render_input(self, input) -> str:
        payload = input.payload
        query = str(payload.get("query", ""))
        sticky = payload.get("sticky_context")
        if not sticky:
            return query

        lines = ["# 上一轮主题(供参考,不要直接复述给用户)"]
        if sticky.get("case_type"):
            lines.append(f"- 案件类型: {sticky['case_type']}")
        if sticky.get("legal_domain"):
            lines.append(f"- 法律领域: {sticky['legal_domain']}")
        if sticky.get("last_law_name"):
            lines.append(f"- 上轮主要法律: {sticky['last_law_name']}")
        mentioned = sticky.get("mentioned_laws") or []
        if mentioned:
            lines.append(f"- 提到过的法律: {', '.join(mentioned)}")
        es = sticky.get("entity_state") or {}
        facts = [f.get("fact", "") if isinstance(f, dict) else str(f) for f in (es.get("key_facts") or [])]
        if facts:
            lines.append(f"- 已知事实: {'; '.join(facts)}")
        lines.append("")
        lines.append(f"# 用户本轮提问\n{query}")
        return "\n".join(lines)
