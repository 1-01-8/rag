"""Contract review business tool (Phase 4).

LLM-driven: takes a contract text, returns risk_items + missing_clauses +
score (0-100) + summary. Output schema = ContractReviewResult.
"""
from __future__ import annotations
from pydantic import BaseModel
from typing import Any

from multi_agent.schemas.messages import AgentMessage, ToolResult
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.json_robust import parse_json_robust


CONTRACT_REVIEW_PROMPT = """你是合同审查专家。请审查下面的合同文本,识别风险条款和缺失条款。

输出 JSON 格式:
```json
{{
  "risk_items": [
    {{"level": "high|medium|low", "clause": "<原条款>", "reason": "<风险原因>", "suggestion": "<修改建议>"}}
  ],
  "missing_clauses": ["<必要但缺失的条款名>"],
  "summary": "<总体评估>",
  "score": <0-100 整数>
}}
```

# 评分标准
- 90-100: 几乎无问题
- 70-89: 少量风险
- 50-69: 中度风险
- 30-49: 重大风险
- 0-29: 严重不合规

# 输出约束
- 只输出 JSON,不输出其他文字

合同文本:
{contract_text}
"""


class ContractReviewArgs(BaseModel):
    contract_text: str


class ContractReviewTool(Tool):
    name: str = "contract_review"
    description: str = (
        "Review a contract for risk clauses and missing standard clauses. "
        "Returns structured risk items, missing clauses, summary, and a 0-100 score."
    )
    args_schema: type[BaseModel] = ContractReviewArgs

    provider: Any
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: ContractReviewArgs, recorder: Recorder) -> ToolResult:
        prompt = CONTRACT_REVIEW_PROMPT.format(contract_text=args.contract_text)
        try:
            resp = await self.provider.complete(
                messages=[AgentMessage(role="user", content=prompt)],
                model=self.model,
                max_tokens=1024,
                temperature=0,
                recorder=recorder,
                agent_name="contract_review_tool",
            )
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=str(e))

        try:
            parsed = parse_json_robust(resp.text)
        except Exception as e:
            return ToolResult(
                tool_use_id="", payload=None,
                error=f"JSON parse failed: {e}",
            )

        return ToolResult(tool_use_id="", payload=parsed)
