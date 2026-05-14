"""Document generation business tool (Phase 4)."""
from __future__ import annotations
import json as _json
from typing import Any
from pydantic import BaseModel

from multi_agent.schemas.messages import AgentMessage, ToolResult
from multi_agent.schemas.doc_generation import DocGenRequest
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.json_robust import parse_json_robust


DOC_GEN_PROMPT_TEMPLATE = """你是法律文书起草专家。根据下面信息起草 {doc_type}。

# 案件事实
{case_facts}

# 当事人
{parties}

# 补充信息
{extra_context}

输出 JSON:
```json
{{
  "doc_type": "{doc_type}",
  "content": "<完整文书内容,可含换行>",
  "placeholders_filled": {{"key": "value"}},
  "meta": {{}}
}}
```

只输出 JSON。
"""


class DocGenerationTool(Tool):
    name: str = "doc_generation"
    description: str = (
        "Generate a legal document (e.g. 民事起诉状, 律师函, 离婚协议) "
        "from case facts and parties information."
    )
    args_schema: type[BaseModel] = DocGenRequest

    provider: Any
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: DocGenRequest, recorder: Recorder) -> ToolResult:
        prompt = DOC_GEN_PROMPT_TEMPLATE.format(
            doc_type=args.doc_type,
            case_facts=args.case_facts,
            parties=_json.dumps(args.parties, ensure_ascii=False),
            extra_context=args.extra_context or "(无)",
        )
        try:
            resp = await self.provider.complete(
                messages=[AgentMessage(role="user", content=prompt)],
                model=self.model, max_tokens=2048, temperature=0,
                recorder=recorder, agent_name="doc_generation_tool",
            )
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=str(e))

        try:
            parsed = parse_json_robust(resp.text)
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=f"JSON parse failed: {e}")

        return ToolResult(tool_use_id="", payload=parsed)
