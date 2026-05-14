"""Document interpretation business tool (Phase 4)."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel

from multi_agent.schemas.messages import AgentMessage, ToolResult
from multi_agent.schemas.doc_interpret import DocInterpretRequest
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.json_robust import parse_json_robust


DOC_INTERPRET_PROMPT_TEMPLATE = """你是法律文书解读专家。请把以下文书翻译成通俗语言,并提取关键条款、权利义务、风险点。

文书原文:
{doc_text}

输出 JSON:
```json
{{
  "key_clauses": [{{"clause": "<条款编号或标题>", "summary": "<一句话摘要>"}}],
  "rights_obligations": "<权利义务概览>",
  "risks": ["<风险点1>", "<风险点2>"],
  "plain_language_summary": "<通俗语言全文摘要>"
}}
```

只输出 JSON。
"""


class DocInterpretTool(Tool):
    name: str = "doc_interpret"
    description: str = (
        "Interpret a legal document into plain language, "
        "extracting key clauses, rights/obligations, and risks."
    )
    args_schema: type[BaseModel] = DocInterpretRequest

    provider: Any
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: DocInterpretRequest, recorder: Recorder) -> ToolResult:
        prompt = DOC_INTERPRET_PROMPT_TEMPLATE.format(doc_text=args.doc_text)
        try:
            resp = await self.provider.complete(
                messages=[AgentMessage(role="user", content=prompt)],
                model=self.model, max_tokens=2048, temperature=0,
                recorder=recorder, agent_name="doc_interpret_tool",
            )
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=str(e))

        try:
            parsed = parse_json_robust(resp.text)
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=f"JSON parse failed: {e}")

        return ToolResult(tool_use_id="", payload=parsed)
