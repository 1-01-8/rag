"""GroundednessJudge — does answer trace to evidence? (Phase 5c §7.7)"""
from __future__ import annotations
import json as _json

from pydantic import BaseModel, Field

from multi_agent.eval.judges.base import LLMJudge

_PROMPT = """你是法律答复"溯源性"审核员。请判断下面的答复每个陈述是否有 evidence 支持。

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
1. 提取答复中所有事实性陈述(法条引用、数字、条件、结论等)
2. 对每条陈述,判断是否在 evidence 中可溯源
3. 输出 JSON:

```json
{{
  "score": 0.0-1.0,
  "ungrounded_claims": ["陈述1", ...],
  "rationale": "简要理由"
}}
```

只输出 JSON。score 是 grounded_claims/total_claims 的比例。
"""


class GroundednessOutput(BaseModel):
    score: float
    ungrounded_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


class GroundednessJudge(LLMJudge[GroundednessOutput]):
    name = "groundedness"
    output_schema = GroundednessOutput

    def render_prompt(self, *, query: str, lawyer_output: dict, evidence_pool: list[dict]) -> str:
        return _PROMPT.format(
            query=query,
            lawyer_output=_json.dumps(lawyer_output, ensure_ascii=False, indent=2),
            evidence_pool=_json.dumps(evidence_pool, ensure_ascii=False, indent=2),
        )
