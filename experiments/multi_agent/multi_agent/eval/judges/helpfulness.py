"""HelpfulnessJudge — is the answer actually helpful to the user? (Phase 5c §7.7)"""
from __future__ import annotations
import json as _json

from pydantic import BaseModel, Field

from multi_agent.eval.judges.base import LLMJudge

_PROMPT = """你是法律答复"有用性"审核员。请从用户实际需求角度评判下面的答复质量。

# 用户问题
{query}

# 律师答复
```json
{lawyer_output}
```

# 证据池(律师可见的检索结果)
```json
{evidence_pool}
```

# 任务
请从以下三个维度评判答复:
1. **直接性(directness)**：答复是否直接回答了用户的问题,没有回避或绕圈子
2. **可操作性(actionability)**：答复是否包含用户可以立即执行的下一步建议
3. **完整性(completeness)**：答复是否覆盖了用户最关心的主要方面

输出 JSON:

```json
{{
  "score": 0.0-1.0,
  "missing_aspects": ["用户仍需追问的方面1", ...],
  "rationale": "简要理由"
}}
```

只输出 JSON。score 是主观整体有用性评分,0 表示完全无用,1 表示极其有用。
missing_aspects 列出用户看完答复后仍需追问的方面(若无则为空列表)。
"""


class HelpfulnessOutput(BaseModel):
    score: float
    missing_aspects: list[str] = Field(default_factory=list)
    rationale: str = ""


class HelpfulnessJudge(LLMJudge[HelpfulnessOutput]):
    name = "helpfulness"
    output_schema = HelpfulnessOutput

    def render_prompt(self, *, query: str, lawyer_output: dict, evidence_pool: list[dict]) -> str:
        return _PROMPT.format(
            query=query,
            lawyer_output=_json.dumps(lawyer_output, ensure_ascii=False, indent=2),
            evidence_pool=_json.dumps(evidence_pool, ensure_ascii=False, indent=2),
        )
