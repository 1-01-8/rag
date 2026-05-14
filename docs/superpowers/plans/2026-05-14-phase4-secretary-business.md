# Phase 4 — Secretary Agent + Business Tools Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Introduce `SecretaryAgent` as a separate agent (wrapped as a tool for Lawyer via Agent-as-Tool per ADR-05) + three business tools the Lawyer can dispatch through Secretary: `contract_review`, `doc_generation`, `doc_interpret` (per LexAI inspiration, ADR-21/§3.5.2). Lawyer no longer calls retrievers directly — it asks Secretary for research.

**Architecture:** `SecretaryAgent` is a tool-using agent. Its tools = retrievers (statute_search, case_search, exact_read, all_sources_search) + business tools (contract_review, doc_generation, doc_interpret). When wrapped as a Tool for the Lawyer, the Lawyer's tool list becomes `[ask_secretary, ask_user_clarify]` (drastically simplified surface). The Secretary handles all "research" and reports back with structured Evidence + summary.

For Phase 4 V1, business tools are LLM-driven (Secretary calls them which means another LLM round inside Secretary's ReAct). They produce structured outputs (RiskItem[], GeneratedDoc, InterpretResult) suitable for Lawyer's LawyerOutput non-consultation modes.

**Phase 3b starting point:** Tag `phase3b-memory-integration`. 175 tests pass + 1 skipped.

---

## Out of scope (Phase 5)

- Supervisor agent (审核 闸门)
- Eval framework (QuerySet / Runner / Comparator / Judges / Ablation)
- Latency Profiler
- `ma_user_history` collection + `history_search` tool (Phase 3c)
- Cross-turn compression (Phase 3c)
- Sequential fan-out across sub_cases (Lawyer dispatches multiple Secretary instances) — Phase 4b enhancement

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   ├── schemas/
│   │   ├── contract_review.py            # NEW: ContractReviewResult, RiskItem (already in lawyer.py — reuse)
│   │   ├── doc_generation.py             # NEW: GeneratedDoc + DocGenRequest
│   │   └── doc_interpret.py              # NEW: InterpretResult + DocInterpretRequest
│   ├── agents/
│   │   ├── secretary.py                  # NEW: SecretaryAgent + SecretaryAsTool wrapper
│   │   └── lawyer.py                     # MODIFY: optional — accept ask_secretary instead of raw retrievers
│   ├── tools/
│   │   └── business/                     # NEW
│   │       ├── __init__.py
│   │       ├── contract_review.py        # NEW: ContractReviewTool (LLM-driven)
│   │       ├── doc_generation.py         # NEW
│   │       └── doc_interpret.py          # NEW
│   └── prompts/secretary/
│       ├── __init__.py
│       └── system.md                     # NEW
└── tests/
    ├── unit/
    │   ├── test_business_schemas.py
    │   ├── test_contract_review.py
    │   ├── test_doc_generation.py
    │   ├── test_doc_interpret.py
    │   └── test_secretary.py
    └── integration/
        └── test_lawyer_via_secretary_e2e.py
```

All tests in `conda run -n qwen35`.

---

## Task 1: Business Schemas

**Files:**
- Create: `multi_agent/schemas/contract_review.py`
- Create: `multi_agent/schemas/doc_generation.py`
- Create: `multi_agent/schemas/doc_interpret.py`
- Create: `tests/unit/test_business_schemas.py`

`RiskItem` already in `schemas/lawyer.py` (Phase 2c). Reuse it. New schemas:

- `ContractReviewResult`: `risk_items[]`, `missing_clauses[]`, `summary`, `score (0-100)`
- `GeneratedDoc`: `doc_type`, `content`, `placeholders_filled`, `meta`
- `InterpretResult`: `key_clauses[]`, `rights_obligations`, `risks`, `plain_language_summary`

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_business_schemas.py
import pytest
from multi_agent.schemas.contract_review import ContractReviewResult
from multi_agent.schemas.doc_generation import GeneratedDoc, DocGenRequest
from multi_agent.schemas.doc_interpret import InterpretResult, DocInterpretRequest
from multi_agent.schemas.lawyer import RiskItem


def test_contract_review_result():
    r = ContractReviewResult(
        risk_items=[RiskItem(level="high", clause="第5条", reason="霸王", suggestion="改为...")],
        missing_clauses=["违约金条款", "争议解决条款"],
        summary="合同存在 1 个高风险条款,缺 2 个必要条款",
        score=65,
    )
    assert r.score == 65
    assert len(r.risk_items) == 1


def test_contract_review_score_must_be_in_range():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ContractReviewResult(risk_items=[], missing_clauses=[], summary="x", score=150)


def test_generated_doc():
    g = GeneratedDoc(
        doc_type="离婚协议",
        content="甲方:...\n乙方:...",
        placeholders_filled={"甲方姓名": "张三", "乙方姓名": "李四"},
        meta={"effective_date": "2026-05-14"},
    )
    assert g.doc_type == "离婚协议"
    assert g.placeholders_filled["甲方姓名"] == "张三"


def test_doc_gen_request():
    r = DocGenRequest(
        doc_type="民事起诉状",
        case_facts="原告与被告...",
        parties={"plaintiff": "张三", "defendant": "李四"},
    )
    assert r.doc_type == "民事起诉状"


def test_interpret_result():
    i = InterpretResult(
        key_clauses=[{"clause": "第3条", "summary": "保密条款"}],
        rights_obligations="...",
        risks=["违约风险"],
        plain_language_summary="这份合同主要规定...",
    )
    assert len(i.key_clauses) == 1
    assert "违约风险" in i.risks


def test_doc_interpret_request():
    r = DocInterpretRequest(doc_text="本合同由甲乙双方签订..." )
    assert "甲乙双方" in r.doc_text
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create the 3 schema files**

`multi_agent/schemas/contract_review.py`:

```python
"""Contract review output schema (Phase 4 business tool)."""
from __future__ import annotations
from pydantic import BaseModel, Field, conint
from multi_agent.schemas.lawyer import RiskItem


class ContractReviewResult(BaseModel):
    risk_items: list[RiskItem] = Field(default_factory=list)
    missing_clauses: list[str] = Field(default_factory=list)
    summary: str
    score: conint(ge=0, le=100)            # 0..100 risk score
```

`multi_agent/schemas/doc_generation.py`:

```python
"""Legal document generation schemas (Phase 4 business tool)."""
from __future__ import annotations
from pydantic import BaseModel, Field


class DocGenRequest(BaseModel):
    doc_type: str                          # e.g. "民事起诉状" / "律师函" / "离婚协议"
    case_facts: str
    parties: dict[str, str] = Field(default_factory=dict)
    extra_context: str = ""


class GeneratedDoc(BaseModel):
    doc_type: str
    content: str
    placeholders_filled: dict[str, str] = Field(default_factory=dict)
    meta: dict = Field(default_factory=dict)
```

`multi_agent/schemas/doc_interpret.py`:

```python
"""Legal document interpretation schemas (Phase 4 business tool)."""
from __future__ import annotations
from pydantic import BaseModel, Field


class DocInterpretRequest(BaseModel):
    doc_text: str


class KeyClause(BaseModel):
    clause: str
    summary: str


class InterpretResult(BaseModel):
    key_clauses: list[dict] = Field(default_factory=list)  # tolerant: dict or KeyClause
    rights_obligations: str
    risks: list[str] = Field(default_factory=list)
    plain_language_summary: str
```

- [ ] **Step 4: Verify pass + full suite** → 181 passed + 1 skipped (175 + 6 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/contract_review.py experiments/multi_agent/multi_agent/schemas/doc_generation.py experiments/multi_agent/multi_agent/schemas/doc_interpret.py experiments/multi_agent/tests/unit/test_business_schemas.py
git commit -m "phase4(schemas): business schemas (ContractReview / DocGeneration / DocInterpret)"
```

---

## Task 2: ContractReviewTool

**Files:**
- Create: `multi_agent/tools/business/__init__.py`
- Create: `multi_agent/tools/business/contract_review.py`
- Create: `tests/unit/test_contract_review.py`

LLM-driven tool: takes contract text, asks LLM to identify risks + missing clauses + score. Returns ContractReviewResult.

For testing, we'll use StubProvider with scripted response — tests don't need real LLM.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_contract_review.py
import pytest
from multi_agent.tools.business.contract_review import (
    ContractReviewTool, ContractReviewArgs,
)
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_contract_review_returns_structured_result(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "risk_items": [{"level": "high", "clause": "第5条", "reason": "霸王条款", "suggestion": "改为协商一致"}],
  "missing_clauses": ["违约金条款"],
  "summary": "合同存在高风险条款",
  "score": 60
}
```''', finish_reason="end_turn"),
    ])
    tool = ContractReviewTool(provider=provider, model="stub-1")
    result = await tool.call(
        ContractReviewArgs(contract_text="甲方应无条件接受乙方任何条款..."),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert payload["score"] == 60
    assert len(payload["risk_items"]) == 1
    assert payload["risk_items"][0]["level"] == "high"


@pytest.mark.asyncio
async def test_contract_review_handles_malformed_json(tmp_run_dir):
    """If LLM returns invalid JSON, tool returns error gracefully."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text="this is not JSON", finish_reason="end_turn"),
    ])
    tool = ContractReviewTool(provider=provider, model="stub-1")
    result = await tool.call(ContractReviewArgs(contract_text="..."), rec)
    rec.close()
    assert result.error is not None
    assert "parse" in result.error.lower() or "json" in result.error.lower()
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create files**

`multi_agent/tools/business/__init__.py` (empty)

`multi_agent/tools/business/contract_review.py`:

```python
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
from multi_agent.providers.base import LLMProvider
from multi_agent.providers.json_robust import parse_json_robust


CONTRACT_REVIEW_PROMPT = """你是合同审查专家。请审查下面的合同文本,识别风险条款和缺失条款。

输出 JSON 格式:
```json
{
  "risk_items": [
    {"level": "high|medium|low", "clause": "<原条款>", "reason": "<风险原因>", "suggestion": "<修改建议>"}
  ],
  "missing_clauses": ["<必要但缺失的条款名>"],
  "summary": "<总体评估>",
  "score": <0-100 整数>
}
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

    provider: Any                       # LLMProvider — Any to avoid Pydantic ABC validation
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: ContractReviewArgs, recorder: Recorder) -> ToolResult:
        prompt = CONTRACT_REVIEW_PROMPT.replace("{contract_text}", args.contract_text)
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
```

- [ ] **Step 4: Verify pass + full suite** → 183 passed + 1 skipped (181 + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/business/__init__.py experiments/multi_agent/multi_agent/tools/business/contract_review.py experiments/multi_agent/tests/unit/test_contract_review.py
git commit -m "phase4(tools): ContractReviewTool (LLM-driven business tool)"
```

---

## Task 3: DocGenerationTool + DocInterpretTool

**Files:**
- Create: `multi_agent/tools/business/doc_generation.py`
- Create: `multi_agent/tools/business/doc_interpret.py`
- Create: `tests/unit/test_doc_generation.py`
- Create: `tests/unit/test_doc_interpret.py`

Follow same pattern as ContractReviewTool — LLM prompt + parse_json_robust. Each returns its corresponding result schema.

- [ ] **Step 1: Failing tests**

```python
# tests/unit/test_doc_generation.py
import pytest
from multi_agent.tools.business.doc_generation import DocGenerationTool
from multi_agent.schemas.doc_generation import DocGenRequest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_doc_generation_returns_structured_doc(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "doc_type": "民事起诉状",
  "content": "原告: 张三\\n被告: 李四\\n诉讼请求: ...",
  "placeholders_filled": {"plaintiff": "张三", "defendant": "李四"},
  "meta": {"jurisdiction": "北京市朝阳区人民法院"}
}
```''', finish_reason="end_turn"),
    ])
    tool = DocGenerationTool(provider=provider, model="stub-1")
    result = await tool.call(
        DocGenRequest(
            doc_type="民事起诉状",
            case_facts="原告与被告...",
            parties={"plaintiff": "张三", "defendant": "李四"},
        ),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert payload["doc_type"] == "民事起诉状"
    assert "原告" in payload["content"]
```

```python
# tests/unit/test_doc_interpret.py
import pytest
from multi_agent.tools.business.doc_interpret import DocInterpretTool
from multi_agent.schemas.doc_interpret import DocInterpretRequest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_doc_interpret_returns_plain_language(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "key_clauses": [{"clause": "第三条", "summary": "保密义务"}],
  "rights_obligations": "甲方负保密义务,乙方有权要求审计",
  "risks": ["违约金过高", "管辖条款不利"],
  "plain_language_summary": "这是一份保密协议,主要保护商业秘密..."
}
```''', finish_reason="end_turn"),
    ])
    tool = DocInterpretTool(provider=provider, model="stub-1")
    result = await tool.call(
        DocInterpretRequest(doc_text="第三条 保密义务\n甲方应..."),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert "保密" in payload["plain_language_summary"]
    assert len(payload["risks"]) == 2
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create files**

`multi_agent/tools/business/doc_generation.py`:

```python
"""Document generation business tool (Phase 4)."""
from __future__ import annotations
from typing import Any
from multi_agent.schemas.messages import AgentMessage, ToolResult
from multi_agent.schemas.doc_generation import DocGenRequest
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.json_robust import parse_json_robust


DOC_GEN_PROMPT = """你是法律文书起草专家。根据下面信息起草 {doc_type}。

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
    args_schema: type = DocGenRequest

    provider: Any
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: DocGenRequest, recorder: Recorder) -> ToolResult:
        import json as _j
        prompt = DOC_GEN_PROMPT.format(
            doc_type=args.doc_type, case_facts=args.case_facts,
            parties=_j.dumps(args.parties, ensure_ascii=False),
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
```

`multi_agent/tools/business/doc_interpret.py`:

```python
"""Document interpretation business tool (Phase 4)."""
from __future__ import annotations
from typing import Any
from multi_agent.schemas.messages import AgentMessage, ToolResult
from multi_agent.schemas.doc_interpret import DocInterpretRequest
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.providers.json_robust import parse_json_robust


DOC_INTERPRET_PROMPT = """你是法律文书解读专家。请把以下文书翻译成通俗语言,并提取关键条款、权利义务、风险点。

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
    args_schema: type = DocInterpretRequest

    provider: Any
    model: str = "qwen3.5-9b"

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: DocInterpretRequest, recorder: Recorder) -> ToolResult:
        prompt = DOC_INTERPRET_PROMPT.format(doc_text=args.doc_text)
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
```

- [ ] **Step 4: Verify pass + full suite** → 185 passed + 1 skipped (183 + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/business/doc_generation.py experiments/multi_agent/multi_agent/tools/business/doc_interpret.py experiments/multi_agent/tests/unit/test_doc_generation.py experiments/multi_agent/tests/unit/test_doc_interpret.py
git commit -m "phase4(tools): DocGenerationTool + DocInterpretTool"
```

---

## Task 4: SecretaryAgent + SecretaryAsTool

**Files:**
- Create: `multi_agent/prompts/secretary/__init__.py`
- Create: `multi_agent/prompts/secretary/system.md`
- Create: `multi_agent/agents/secretary.py`
- Create: `tests/unit/test_secretary.py`
- Modify: `pyproject.toml` — add `secretary/*.md` to package-data

SecretaryAgent is a regular BaseAgent that uses retrievers + business tools. Then wrapped via `SecretaryAsTool` so Lawyer can call it.

For Phase 4 V1, we implement BOTH the agent class AND the wrapper Tool class.

- [ ] **Step 1: Create prompt + update pyproject**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/secretary
touch /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/secretary/__init__.py
```

`multi_agent/prompts/secretary/system.md`:

```markdown
你是律师事务所的秘书。你的工作是为律师做研究和事务性工作。

# 可用工具
- statute_search: 检索法条
- read_article: 精确获取某条法律全文
- case_search: 检索过往类案
- contract_review: 合同审查(返回风险条款+评分)
- doc_generation: 起草法律文书
- doc_interpret: 解读法律文书

# 工作流程
1. 理解律师的请求(query)
2. 选择合适的工具(可多次调用,先 statute_search,再 case_search,等等)
3. 汇总结果,以 JSON 输出

# 输出 JSON
```json
{
  "summary": "<一句话总结>",
  "evidences": [],
  "notes": "<给律师的备注>",
  "confidence": 0.8
}
```

只输出 JSON,不输出其他文字。
```

In `pyproject.toml`, find `[tool.setuptools.package-data]` and add secretary:

```toml
[tool.setuptools.package-data]
multi_agent = ["prompts/lawyer/*.md", "prompts/receptionist/*.md", "prompts/secretary/*.md"]
```

Reinstall: `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pip install -e '.[dev]' 2>&1 | tail -3"`.

- [ ] **Step 2: Failing test**

```python
# tests/unit/test_secretary.py
import pytest
from pydantic import BaseModel
from multi_agent.agents.secretary import SecretaryAgent, SecretaryResponse, SecretaryAsTool, SecretaryRequest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from multi_agent.agents.base import AgentInput


def test_secretary_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SecretaryAgent(name="secretary", role="research",
                          provider=p, recorder=rec, model="stub-1")
    prompt = agent.system_prompt()
    assert "秘书" in prompt
    assert "statute_search" in prompt
    rec.close()


def test_secretary_output_schema(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = SecretaryAgent(name="secretary", role="research",
                          provider=p, recorder=rec)
    assert agent.output_schema() is SecretaryResponse
    rec.close()


@pytest.mark.asyncio
async def test_secretary_as_tool_dispatches_to_agent(tmp_path):
    """SecretaryAsTool should run the wrapped SecretaryAgent and return its output."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"summary": "found Article 510", "evidences": [], "notes": "", "confidence": 0.9}',
            finish_reason="end_turn",
        ),
    ])
    secretary = SecretaryAgent(name="secretary", role="research",
                              provider=p, recorder=rec, model="stub-1",
                              max_pre_tool_rejections=10)  # no tools, no enforcement
    tool = SecretaryAsTool(secretary_agent=secretary)
    result = await tool.call(SecretaryRequest(task="search", payload={"query": "民法典 510"}), rec)
    rec.close()
    assert result.error is None
    assert result.payload["summary"] == "found Article 510"
```

- [ ] **Step 3: Verify failure** → ImportError.

- [ ] **Step 4: Create `multi_agent/agents/secretary.py`**

```python
"""SecretaryAgent — research + business tools delegated by Lawyer.

Wrapped via SecretaryAsTool for Agent-as-Tool dispatch (ADR-05).
"""
from __future__ import annotations
from importlib.resources import files
from typing import Any
from pydantic import BaseModel, Field

from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.schemas.evidence import Evidence
from multi_agent.schemas.messages import ToolResult
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class SecretaryResponse(BaseModel):
    summary: str                                        # one-line gist
    evidences: list[Evidence] = Field(default_factory=list)
    notes: str = ""
    confidence: float = 0.0


class SecretaryAgent(BaseAgent):
    """Research agent. Uses retrievers + business tools."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.secretary").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[SecretaryResponse]:
        return SecretaryResponse


# --- Agent-as-Tool wrapper ---

class SecretaryRequest(BaseModel):
    task: str                                           # "search" / "review_contract" / "draft_doc" / ...
    payload: dict[str, Any] = Field(default_factory=dict)


class SecretaryAsTool(Tool):
    name: str = "ask_secretary"
    description: str = (
        "Ask the secretary to do research (statute/case retrieval) or "
        "business work (contract review / doc generation / doc interpret). "
        "Pass a task description and any relevant context in payload."
    )
    args_schema: type[BaseModel] = SecretaryRequest

    secretary_agent: Any                                # SecretaryAgent

    model_config = {"arbitrary_types_allowed": True}

    async def call(self, args: SecretaryRequest, recorder: Recorder) -> ToolResult:
        try:
            input = AgentInput(payload={
                "query": args.task,
                **args.payload,
            })
            output = await self.secretary_agent.run(input)
            return ToolResult(
                tool_use_id="",
                payload=output.payload.model_dump(),
            )
        except Exception as e:
            return ToolResult(tool_use_id="", payload=None, error=str(e))
```

- [ ] **Step 5: Verify pass + full suite** → 188 passed + 1 skipped (185 + 3 new).

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/prompts/secretary/ experiments/multi_agent/multi_agent/agents/secretary.py experiments/multi_agent/tests/unit/test_secretary.py experiments/multi_agent/pyproject.toml
git commit -m "phase4(agents): SecretaryAgent + SecretaryAsTool (Agent-as-Tool wrapper)"
```

---

## Task 5: Lawyer-via-Secretary E2E (Real Qwen)

**Files:**
- Create: `tests/integration/test_lawyer_via_secretary_e2e.py`

Real Qwen-driven test: Lawyer is given `[SecretaryAsTool]` only (not raw retrievers). Lawyer asks Secretary for legal research, Secretary uses statute_search internally, returns to Lawyer.

Expected behavior is more brittle than direct-retrieval — there's a nested ReAct loop (Lawyer's loop calls Secretary which has its own loop). The test tolerates some Qwen flakiness; just verifies the chain doesn't crash and Lawyer produces a valid LawyerOutput.

- [ ] **Step 1: Write test**

```python
# tests/integration/test_lawyer_via_secretary_e2e.py
"""Phase 4 E2E: Lawyer delegates research to Secretary; Secretary uses retrievers."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.secretary import SecretaryAgent, SecretaryAsTool
from multi_agent.runner import run_query
from multi_agent.tracing.recorder import Recorder


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_sec_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
            Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="703",
                  text="租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ],
    )]
    build_index(documents=docs, collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_lawyer_delegates_to_secretary(statute_index, tmp_path):
    """Lawyer gets ONLY [ask_secretary]; Secretary internally has statute_search."""
    runs_root = tmp_path / "runs"
    provider = OpenAICompatibleProvider()

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    def lawyer_factory(p, r):
        # Secretary inside Lawyer — shares same recorder so trace nesting is correct
        secretary = SecretaryAgent(
            name="secretary", role="research",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            max_steps=5, max_tool_calls=8,
        )
        return LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[SecretaryAsTool(secretary_agent=secretary)],
            model="qwen3.5-9b",
            specialty="民事",
            max_steps=8, max_tool_calls=10,
        )

    result = await run_query(
        query="房东合同期内涨租 30% 合法吗?",
        agent_factory=lawyer_factory,
        provider=provider, runs_root=runs_root, config={},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"
    # Verify ask_secretary was called at least once
    events = [json.loads(l) for l in (runs_root / result["run_id"] / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    tool_names = {e["tool_name"] for e in tool_calls}
    assert "ask_secretary" in tool_names, f"Lawyer didn't call ask_secretary; tools called: {tool_names}"
```

- [ ] **Step 2: Run + Step 3: Full suite + Step 4: Commit + tag**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_lawyer_via_secretary_e2e.py -v"
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -10"

cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_lawyer_via_secretary_e2e.py
git commit -m "phase4(integration): Lawyer-via-Secretary E2E with real Qwen"
git tag -a phase4-secretary-business -m "Phase 4 complete: SecretaryAgent + business tools (contract/doc gen/doc interpret)"
git tag -l "phase*"
```

## Acceptance Criteria

Phase 4 complete when:

1. Full pytest passes (~189 tests)
2. Business schemas + 3 LLM-driven tools work (ContractReview/DocGen/DocInterpret)
3. SecretaryAgent + SecretaryAsTool work; can be embedded in Lawyer
4. Real-Qwen E2E proves Lawyer-via-Secretary chain works without crashes
5. Tag `phase4-secretary-business` exists

## Out of Scope (carry forward)

- Phase 5: Supervisor + eval framework + ablations
- Phase 3c: Cross-turn compression + ma_user_history collection
- Real-API integration test for Anthropic with business tools
- HyDE / advanced query rewriting
