"""SupervisorAgent — post-hoc QA on Lawyer output."""
from __future__ import annotations
import json as _json
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.supervisor import SupervisorVerdict


class SupervisorAgent(BaseAgent):
    """Reviews Lawyer output. Tools: verify_citation."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.supervisor").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[SupervisorVerdict]:
        return SupervisorVerdict

    def _render_input(self, input) -> str:
        payload = input.payload
        user_q = payload.get("user_query", "")
        lawyer_out = payload.get("lawyer_output", {})
        evidence_pool = payload.get("evidence_pool", [])
        return (
            f"# 用户原始问题\n{user_q}\n\n"
            f"# 律师答复(LawyerOutput)\n```json\n{_json.dumps(lawyer_out, ensure_ascii=False, indent=2)}\n```\n\n"
            f"# 律师检索到的证据池\n```json\n{_json.dumps(evidence_pool, ensure_ascii=False, indent=2)}\n```\n\n"
            "请审核并输出 SupervisorVerdict JSON。"
        )
