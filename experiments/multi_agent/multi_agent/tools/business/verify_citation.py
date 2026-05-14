"""Programmatic citation verification tool (Phase 5a)."""
from __future__ import annotations
from pydantic import BaseModel, Field

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.lawyer import Citation
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class VerifyCitationArgs(BaseModel):
    citation: Citation
    evidences: list[Evidence] = Field(default_factory=list)


class VerifyCitationTool(Tool):
    name: str = "verify_citation"
    description: str = (
        "Verify a Citation is grounded in the retrieved Evidence pool. "
        "Returns {valid: bool, reason: str}."
    )
    args_schema: type[BaseModel] = VerifyCitationArgs

    async def call(self, args: VerifyCitationArgs, recorder: Recorder) -> ToolResult:
        target = f"{args.citation.law_short}-{args.citation.article_no}"
        match = None
        for ev in args.evidences:
            if ev.doc_id == target:
                match = ev
                break
            if ev.law_short == args.citation.law_short and ev.article_no == args.citation.article_no:
                match = ev
                break
        if match is None:
            return ToolResult(tool_use_id="", payload={
                "valid": False,
                "reason": f"Citation {target} not in retrieved evidence",
            })

        if args.citation.excerpt and args.citation.excerpt.strip():
            ex = args.citation.excerpt.strip()
            if ex not in match.text:
                normalized_text = match.text.replace(",", ",").replace("。", ".")
                normalized_ex = ex.replace(",", ",").replace("。", ".")
                if normalized_ex not in normalized_text:
                    return ToolResult(tool_use_id="", payload={
                        "valid": False,
                        "reason": f"Excerpt not found in Evidence.text for {target}",
                    })

        return ToolResult(tool_use_id="", payload={"valid": True, "reason": "matches"})
