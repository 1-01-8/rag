# Phase 2c — Real Lawyer Agent (Five-Section Prompt + ReAct) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stub agents with a real `LawyerAgent` that follows the spec §3.5.1 five-section structure (争议分析 / 适用法规 / 相似类案 / 维权建议 / 风险评估) and does ReAct over `statute_search` / `read_article` tools against the Phase 2a Chinese-Laws corpus, driven by the Phase 2b providers. Close Phase 2b's three "Important" findings: streaming usage telemetry, stream_one_turn tool-chunk handling, and stream tool_call coverage.

**Architecture:** Builds directly on Phase 2a (retrievers) and Phase 2b (providers). New code lives in `agents/lawyer.py` + `prompts/lawyer/*.md`. The lawyer is one class with a runtime-selected specialty prompt; specialties (民事, 劳动, etc.) are markdown files loaded at construction. We do NOT introduce a separate class per specialty.

**Tech Stack:** Python 3.10+, Pydantic 2.x, existing dependencies. No new system deps.

**Spec reference:** `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` §3.5 / §3.5.1 / §3.5.2; ADR-21 (five-section prompt).

**Phase 2b starting point:** Tag `phase2b-real-providers`. 118 tests passing + 1 skipped (Anthropic real-API).

---

## What this plan does NOT cover

Out of scope (later phases):
- **Phase 3**: Receptionist + EntityState/WorkingMemory + multi-issue (sub_cases) handling
- **Phase 4**: Secretary as separate agent + agent-as-tool wrapping + contract_review/doc_generation/doc_interpret business tools
- **Phase 5**: Supervisor + eval framework + ablations
- Streaming-with-tools (full ReAct loop streaming through tool dispatch) — too complex; deferred indefinitely
- Case (`ma_cases`) + user_history collections — Phase 2d

---

## File Structure (Phase 2c additions)

```
experiments/multi_agent/
├── multi_agent/
│   ├── agents/
│   │   └── lawyer.py                           # NEW: LawyerAgent class
│   ├── providers/
│   │   ├── openai_compatible.py                # MODIFY: streaming usage telemetry
│   │   └── anthropic.py                        # MODIFY: streaming usage telemetry
│   └── agents/base.py                          # MODIFY: stream_one_turn handles tool_call kinds
├── prompts/                                    # NEW DIR
│   └── lawyer/
│       ├── _five_section_skeleton.md           # shared 5-section framework
│       ├── specialty_民事.md                    # 民事咨询
│       ├── specialty_劳动.md                    # 劳动纠纷
│       ├── specialty_交通.md                    # 交通事故
│       ├── specialty_婚姻.md                    # 婚姻家庭
│       ├── specialty_房产.md                    # 房产纠纷
│       └── specialty_通用.md                    # fallback
└── tests/
    ├── unit/
    │   ├── test_lawyer.py                      # Lawyer construction, prompt loading, output schema
    │   └── test_streaming_telemetry.py         # streaming usage capture
    └── integration/
        ├── test_lawyer_civil_e2e.py            # real Qwen + Lawyer + statute_search → 涨租 case
        ├── test_lawyer_labor_e2e.py            # real Qwen + Lawyer + 劳动 specialty
        └── test_lawyer_traffic_e2e.py          # real Qwen + Lawyer + 交通 specialty
```

**Working directory for all tasks:** `/home/xxm/rag/experiments/multi_agent/`
**Test command prefix:** `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && <pytest cmd>"`

---

## Task 0: Phase 2b Follow-Up — Streaming Usage Telemetry

**Files:**
- Modify: `multi_agent/providers/openai_compatible.py` — pass `stream_options={"include_usage": True}` and capture final usage chunk
- Modify: `multi_agent/providers/anthropic.py` — capture usage from `message_delta` event
- Create: `tests/unit/test_streaming_telemetry.py`

Phase 2b final review flagged that `complete_stream` discards usage. Phase 3 budget gates need it. Fix now while context is fresh.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_streaming_telemetry.py
"""Streaming must capture usage telemetry — required by Phase 3 budget gates."""
import json as _j
import pytest
import respx
import httpx
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="test-key")


_BASE_URL = "https://api.anthropic.com/v1/messages"


@respx.mock
@pytest.mark.asyncio
async def test_anthropic_streaming_records_usage(anthropic_provider, tmp_run_dir):
    """The LLMResponded event from streaming must record token usage."""
    sse_body = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":7,"output_tokens":0}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        'event: content_block_stop\n'
        'data: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}\n\n'
        'event: message_stop\n'
        'data: {"type":"message_stop"}\n\n'
    )
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body.encode(),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    chunks = []
    async for ch in anthropic_provider.complete_stream(
        messages=[AgentMessage(role="user", content="hi")],
        model="claude-sonnet-4-6", recorder=rec, agent_name="t",
    ):
        chunks.append(ch)
    rec.close()

    events = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(responded) == 1
    usage = responded[0]["usage"]
    assert usage["input_tokens"] == 7    # from message_start
    assert usage["output_tokens"] == 4   # from message_delta


@pytest.mark.asyncio
async def test_openai_compat_streaming_records_usage(tmp_run_dir):
    """Real Qwen — streaming should fetch usage via stream_options."""
    import httpx as _httpx
    try:
        with _httpx.Client(timeout=2.0) as c:
            assert c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        pytest.skip("vLLM not running")

    provider = OpenAICompatibleProvider()
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    async for _ in provider.complete_stream(
        messages=[AgentMessage(role="user", content="say one word")],
        model="qwen3.5-9b", max_tokens=8, temperature=0,
        recorder=rec, agent_name="t",
    ):
        pass
    rec.close()

    events = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(responded) == 1
    usage = responded[0]["usage"]
    # vLLM returns usage when stream_options.include_usage=True
    assert usage.get("input_tokens", 0) > 0
    assert usage.get("output_tokens", 0) >= 1
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_streaming_telemetry.py -v"
```

Expected: both FAIL (usage fields stay at 0 / not set).

- [ ] **Step 3: Update `OpenAICompatibleProvider.complete_stream`**

In `multi_agent/providers/openai_compatible.py`, in `complete_stream` method, modify the `stream` creation and accumulation:

Replace:
```python
                stream = await self._client.chat.completions.create(
                    model=model, messages=oai_messages, tools=oai_tools or None,
                    max_tokens=max_tokens, temperature=temperature, stream=True,
                )
```

With:
```python
                stream = await self._client.chat.completions.create(
                    model=model, messages=oai_messages, tools=oai_tools or None,
                    max_tokens=max_tokens, temperature=temperature, stream=True,
                    stream_options={"include_usage": True},
                )
```

And replace the accumulation loop:
```python
            full_text = ""
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield StreamChunk(kind="token", content=delta.content)
                if event.choices[0].finish_reason is not None:
                    break
            span.set_output({"raw": full_text, "usage": {}, "finish_reason": "end_turn"})
```

With:
```python
            full_text = ""
            captured_usage = {"input_tokens": 0, "output_tokens": 0}
            finish_reason = "end_turn"
            async for event in stream:
                # vLLM/OpenAI emits a final event with usage when include_usage=True;
                # this event has empty choices[].
                if event.usage:
                    captured_usage = {
                        "input_tokens": event.usage.prompt_tokens,
                        "output_tokens": event.usage.completion_tokens,
                    }
                if not event.choices:
                    continue
                delta = event.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    yield StreamChunk(kind="token", content=delta.content)
                if event.choices[0].finish_reason is not None:
                    finish_reason = self._normalize_finish_reason(event.choices[0].finish_reason)
            span.set_output({"raw": full_text, "usage": captured_usage, "finish_reason": finish_reason})
```

- [ ] **Step 4: Update `AnthropicProvider.complete_stream`**

In `multi_agent/providers/anthropic.py`, in `complete_stream` method:

Replace:
```python
            full_text = ""
            try:
                async with self._client.messages.stream(...) as stream:
                    async for text in stream.text_stream:
                        full_text += text
                        yield StreamChunk(kind="token", content=text)
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic stream failed: {e}") from e
            span.set_output({"raw": full_text, "usage": {}, "finish_reason": "end_turn"})
```

With:
```python
            full_text = ""
            usage_in = 0
            usage_out = 0
            stop_reason = "end_turn"
            try:
                async with self._client.messages.stream(
                    model=model,
                    system=cache_friendly_system,
                    messages=anthropic_messages,
                    tools=anthropic_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    async for event in stream:
                        et = getattr(event, "type", None)
                        if et == "message_start":
                            msg = getattr(event, "message", None)
                            u = getattr(msg, "usage", None) if msg else None
                            if u:
                                usage_in = getattr(u, "input_tokens", 0) or 0
                        elif et == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            dt = getattr(delta, "type", None) if delta else None
                            if dt == "text_delta":
                                text = getattr(delta, "text", "") or ""
                                if text:
                                    full_text += text
                                    yield StreamChunk(kind="token", content=text)
                        elif et == "message_delta":
                            delta = getattr(event, "delta", None)
                            sr = getattr(delta, "stop_reason", None) if delta else None
                            if sr:
                                stop_reason = self._normalize_stop_reason(sr)
                            u = getattr(event, "usage", None)
                            if u:
                                usage_out = getattr(u, "output_tokens", 0) or 0
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic stream failed: {e}") from e
            span.set_output({
                "raw": full_text,
                "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
                "finish_reason": stop_reason,
            })
```

The shift from `stream.text_stream` to `async for event in stream` is the key change. We lose the convenience iterator but gain access to typed events with usage.

- [ ] **Step 5: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_streaming_telemetry.py tests/unit/test_openai_compat.py tests/unit/test_anthropic.py -v"
```

Expected: all tests pass. The 2 new streaming-telemetry tests should be green; the existing streaming tests should not regress.

Then full suite:

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -5"
```

Expected: 120 passed + 1 skipped (118 + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/openai_compatible.py experiments/multi_agent/multi_agent/providers/anthropic.py experiments/multi_agent/tests/unit/test_streaming_telemetry.py
git commit -m "phase2c(providers): streaming captures usage for budget gates (resolves 2b review)"
```

---

## Task 1: LawyerOutput Schema (Five-Section Structured Result)

**Files:**
- Create: `multi_agent/schemas/lawyer.py`
- Create: `tests/unit/test_lawyer_output.py`

Define the structured output schema lawyers produce. Per spec §3.5.2, `LawyerOutput.mode` selects between consultation / contract_review / doc_generation / doc_interpret. Phase 2c covers only `consultation` (the other modes are Phase 4 business tools); `mode` field stays so Phase 4 can extend without breaking schema.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_lawyer_output.py
import pytest
from multi_agent.schemas.lawyer import (
    LawyerOutput, FiveSection, Citation, RiskItem,
)


def test_lawyer_output_consultation_mode():
    out = LawyerOutput(
        mode="consultation",
        primary_answer="房东不能单方涨租。",
        citations=[
            Citation(law_short="民法典", article_no="510", excerpt="按照交易习惯..."),
        ],
        five_section=FiveSection(
            dispute_analysis="租赁合同期内房东要求涨租 30%, 用户拒绝。",
            applicable_laws="《民法典》第 510 条规定...",
            similar_cases="（无类案）",
            remedy_suggestions="1. 与房东协商 2. 拒绝缴纳超额租金 3. 必要时仲裁",
            risk_assessment="胜诉可能性较高,因合同未约定涨租条款。",
        ),
    )
    assert out.mode == "consultation"
    assert out.primary_answer.startswith("房东")
    assert len(out.citations) == 1
    assert out.citations[0].article_no == "510"


def test_lawyer_output_other_modes_have_no_five_section():
    """contract_review / doc_generation / doc_interpret are Phase 4 modes
    — five_section may be None for those."""
    out = LawyerOutput(
        mode="contract_review",
        primary_answer="合同存在 2 个风险条款。",
        citations=[],
        risk_items=[
            RiskItem(level="high", clause="第 5 条", reason="霸王条款", suggestion="改为..."),
        ],
    )
    assert out.mode == "contract_review"
    assert out.five_section is None


def test_lawyer_output_rejects_unknown_mode():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        LawyerOutput(mode="bogus", primary_answer="", citations=[])
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_lawyer_output.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/lawyer.py`**

```python
"""Lawyer agent output schema — five-section structured answers per spec §3.5."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A specific article citation with excerpt for grounding."""
    law_short: str          # e.g. "民法典"
    article_no: str         # e.g. "510"
    excerpt: str = ""       # short quote from the article (≤ 200 chars)


class FiveSection(BaseModel):
    """The five-section consultation framework (spec §3.5.1)."""
    dispute_analysis: str   # 【争议分析】
    applicable_laws: str    # 【适用法规】
    similar_cases: str      # 【相似类案】
    remedy_suggestions: str # 【维权建议】
    risk_assessment: str    # 【风险评估】


class RiskItem(BaseModel):
    """Used by contract_review mode (Phase 4)."""
    level: Literal["high", "medium", "low"]
    clause: str
    reason: str
    suggestion: str


class LawyerOutput(BaseModel):
    """Top-level lawyer output. Mode selects sub-fields (spec §3.5.2).

    Phase 2c implements `consultation` mode (with `five_section`).
    `contract_review` / `doc_generation` / `doc_interpret` modes are
    declared here but produced by Phase 4 business tools.
    """
    mode: Literal["consultation", "contract_review", "doc_generation", "doc_interpret"]
    primary_answer: str         # short summary or final answer
    citations: list[Citation] = Field(default_factory=list)
    five_section: FiveSection | None = None         # for consultation
    risk_items: list[RiskItem] | None = None        # for contract_review
    generated_doc: str | None = None                # for doc_generation
    interpretation: dict | None = None              # for doc_interpret
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_lawyer_output.py -v"
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/lawyer.py experiments/multi_agent/tests/unit/test_lawyer_output.py
git commit -m "phase2c(schemas): LawyerOutput + FiveSection + Citation + RiskItem"
```

---

## Task 2: Five-Section Skeleton + Generic Specialty Prompts

**Files:**
- Create: `multi_agent/prompts/lawyer/_five_section_skeleton.md`
- Create: `multi_agent/prompts/lawyer/specialty_通用.md` (fallback)
- Modify: `pyproject.toml` — `[tool.setuptools.package-data]` so prompts ship with the package

Specialty prompts use a shared skeleton (the five-section template + JSON output requirement) and add specialty-specific reminders on top. Task 3 will add 5 more specialties.

- [ ] **Step 1: Create the directory + skeleton**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/lawyer
```

Write `/home/xxm/rag/experiments/multi_agent/multi_agent/prompts/lawyer/_five_section_skeleton.md`:

```markdown
你是一位资深律师。你的工作流程:

1. 用 statute_search 工具检索相关法条(必要时多次检索)
2. 用 read_article 精确获取关键法条全文
3. 综合检索结果按"五段式"输出 JSON

# 五段式产出格式(必须严格遵守)

输出 JSON 格式:
```json
{
  "mode": "consultation",
  "primary_answer": "<一句话核心结论>",
  "citations": [
    {"law_short": "民法典", "article_no": "510", "excerpt": "<原文摘录,≤100字>"}
  ],
  "five_section": {
    "dispute_analysis": "【争议分析】明确争议焦点和法律性质",
    "applicable_laws": "【适用法规】引用具体法律条文,禁止编造",
    "similar_cases": "【相似类案】若有则列出,无则注明'无类案'",
    "remedy_suggestions": "【维权建议】证据收集 / 程序路径 / 时效提醒",
    "risk_assessment": "【风险评估】胜诉可能性 / 替代方案"
  }
}
```

# 强制规则
- citations 中每条法条必须经 statute_search 或 read_article 实际检索得到;**严禁编造法条号或内容**
- excerpt 必须是工具返回原文的真实片段
- 若检索为空,在 dispute_analysis 中如实声明"未检索到直接适用法条"
- 不输出任何额外文字,只输出 JSON
```

Write `/home/xxm/rag/experiments/multi_agent/multi_agent/prompts/lawyer/specialty_通用.md`:

```markdown
# 通用法律咨询

适用于所有未指定具体领域的咨询。重点:
- 优先检索《民法典》总则编与合同编
- 用户描述含具体行业(如医疗/教育/交通)时,提示用户选择更专业的咨询路径
- 时效提醒:民事一般 3 年(《民法典》188 条)
```

- [ ] **Step 2: Ensure markdown ships with the package**

Read current `pyproject.toml`. Find the section `[tool.setuptools.packages.find]` and either:

a) Add a `[tool.setuptools.package-data]` section after it:

```toml
[tool.setuptools.package-data]
multi_agent = ["prompts/lawyer/*.md"]
```

b) Re-install to pick up the new package data:

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pip install -e '.[dev]' 2>&1 | tail -3"
```

- [ ] **Step 3: Smoke test that prompts are loadable**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && python -c \"
from importlib.resources import files
skel = files('multi_agent.prompts.lawyer').joinpath('_five_section_skeleton.md').read_text(encoding='utf-8')
generic = files('multi_agent.prompts.lawyer').joinpath('specialty_通用.md').read_text(encoding='utf-8')
print('skeleton chars:', len(skel))
print('generic chars:', len(generic))
assert '五段式' in skel
assert '通用法律咨询' in generic
print('OK')
\""
```

Expected: prints non-zero char counts + `OK`.

If `importlib.resources.files` doesn't find the package data, the install didn't pick it up. Re-run pip install or add an `__init__.py` to `multi_agent/prompts/` and `multi_agent/prompts/lawyer/` directories.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/prompts/ experiments/multi_agent/pyproject.toml
git commit -m "phase2c(prompts): five-section skeleton + 通用 specialty + package data"
```

---

## Task 3: Five Specialty Prompts (民事 / 劳动 / 交通 / 婚姻 / 房产)

**Files:**
- Create: `multi_agent/prompts/lawyer/specialty_民事.md`
- Create: `multi_agent/prompts/lawyer/specialty_劳动.md`
- Create: `multi_agent/prompts/lawyer/specialty_交通.md`
- Create: `multi_agent/prompts/lawyer/specialty_婚姻.md`
- Create: `multi_agent/prompts/lawyer/specialty_房产.md`

Mirrors the 4 main cause categories from `laws_data` (婚姻家庭, 债权债务, 交通事故, 房产纠纷, 劳动纠纷 — note: 劳动 is supported here even though corpus is incomplete; ADR-15 acceptance) plus 民事 as catch-all.

- [ ] **Step 1: Write the 5 prompts**

Each follows the same pattern: short header, applicable laws to prefer, specialty-specific reminders.

`specialty_民事.md`:
```markdown
# 民事咨询

适用范围:债权债务、合同纠纷、侵权、继承等一般民事问题。

# 优先检索方向
- 《民法典》合同编(第三编)— 合同纠纷
- 《民法典》侵权责任编(第七编)— 侵权
- 《民法典》总则编 第188条 — 诉讼时效(3年)

# 强制提醒清单
- 时效:一般 3 年;最长 20 年
- 证据收集:合同/聊天记录/转账凭证
- 程序:协商 → 调解 → 诉讼;标的额小可申请简易程序
```

`specialty_劳动.md`:
```markdown
# 劳动纠纷咨询

适用范围:工资拖欠、违法辞退、加班费、社保、工伤等。

# 优先检索方向
- 《劳动合同法》— 注意:本 corpus 暂未收录,可能无直接命中
- 《社会保险法》— 已收录
- 《工会法》— 已收录

# 重要提醒(必须包含)
- **仲裁时效 1 年**(《劳动争议调解仲裁法》27条)— 务必告知用户
- 证据:劳动合同/工资流水/考勤记录/工作沟通记录
- 程序:劳动仲裁前置,不服仲裁可起诉
- 经济补偿金(N) vs 赔偿金(2N)区别

# 注意
若工具未返回劳动合同法相关法条,在 similar_cases 中注明"本系统当前 corpus 未收录《劳动合同法》"。
```

`specialty_交通.md`:
```markdown
# 交通事故咨询

适用范围:机动车事故责任、人身/财产损害赔偿、保险理赔。

# 优先检索方向
- 《道路交通安全法》— 已收录
- 《民法典》侵权责任编 — 第1208-1217条机动车交通事故

# 强制提醒清单
- 时效:1 年(人身损害)/ 3 年(财产)
- 证据:事故认定书 / 病历 / 修车发票 / 误工证明
- 程序:交警事故认定 → 保险公司 → 必要时诉讼
- 责任划分:全责/主责/同责/次责/无责
```

`specialty_婚姻.md`:
```markdown
# 婚姻家庭咨询

适用范围:离婚、财产分割、子女抚养、家暴、继承。

# 优先检索方向
- 《民法典》婚姻家庭编(第五编)
- 《反家庭暴力法》— 已收录
- 《妇女权益保障法》— 已收录

# 强制提醒清单
- **冷静期**:协议离婚需 30 天冷静期(《民法典》1077条)
- 财产:婚前个人 vs 婚后共同
- 子女:2 周岁以下原则上随母,8 周岁以上听其意见
- 家暴:可申请人身安全保护令(《反家暴法》23条)
```

`specialty_房产.md`:
```markdown
# 房产纠纷咨询

适用范围:租赁、买卖、物业、相邻权、宅基地。

# 优先检索方向
- 《民法典》物权编(第二编)— 第205-462条
- 《民法典》合同编 第703-734条 — 租赁合同
- 《城市房地产管理法》— 已收录

# 强制提醒清单
- 租赁期内不得单方涨租(除非合同明确约定)
- 房屋买卖:网签 → 过户 → 物权法定生效
- 装修瑕疵:质保期内可主张维修
- 时效:一般 3 年
```

- [ ] **Step 2: Reinstall package data + smoke test**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pip install -e '.[dev]' 2>&1 | tail -3 && python -c \"
from importlib.resources import files
for sp in ['民事','劳动','交通','婚姻','房产']:
    text = files('multi_agent.prompts.lawyer').joinpath(f'specialty_{sp}.md').read_text(encoding='utf-8')
    assert len(text) > 100, f'{sp} too short'
    print(f'{sp}: {len(text)} chars OK')
\""
```

Expected: all 5 specialties print char counts + OK.

- [ ] **Step 3: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/prompts/lawyer/specialty_民事.md experiments/multi_agent/multi_agent/prompts/lawyer/specialty_劳动.md experiments/multi_agent/multi_agent/prompts/lawyer/specialty_交通.md experiments/multi_agent/multi_agent/prompts/lawyer/specialty_婚姻.md experiments/multi_agent/multi_agent/prompts/lawyer/specialty_房产.md
git commit -m "phase2c(prompts): 5 specialty prompts (民事/劳动/交通/婚姻/房产)"
```

---

## Task 4: LawyerAgent Class

**Files:**
- Create: `multi_agent/agents/lawyer.py`
- Create: `tests/unit/test_lawyer.py`

The Lawyer is one class. Its specialty is selected at construction time (defaults to 通用). The class builds its `system_prompt()` by concatenating `_five_section_skeleton.md` + `specialty_<name>.md`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_lawyer.py
import pytest
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


def test_lawyer_default_specialty_is_general(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                        provider=p, recorder=rec)
    prompt = lawyer.system_prompt()
    assert "五段式" in prompt
    assert "通用法律咨询" in prompt
    rec.close()


def test_lawyer_specialty_loads_correct_prompt(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    for sp in ["民事", "劳动", "交通", "婚姻", "房产"]:
        lawyer = LawyerAgent(name="lawyer", role="advisor",
                             provider=p, recorder=rec, specialty=sp)
        prompt = lawyer.system_prompt()
        assert "五段式" in prompt, f"{sp} missing skeleton"
        # Each specialty file mentions its own name in its header
        assert sp in prompt, f"{sp} prompt did not include specialty marker"
    rec.close()


def test_lawyer_output_schema_is_lawyer_output(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                        provider=p, recorder=rec)
    assert lawyer.output_schema() is LawyerOutput
    rec.close()


def test_lawyer_unknown_specialty_raises(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    with pytest.raises(ValueError, match="unknown specialty"):
        LawyerAgent(name="lawyer", role="advisor",
                    provider=p, recorder=rec, specialty="nonexistent")
    rec.close()
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_lawyer.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/agents/lawyer.py`**

```python
"""LawyerAgent — real consultation agent with five-section prompt.

One class, runtime-selected specialty. The system prompt is built from
the shared skeleton + specialty markdown file.
"""
from __future__ import annotations
from importlib.resources import files
from typing import ClassVar

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
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_lawyer.py -v"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/lawyer.py experiments/multi_agent/tests/unit/test_lawyer.py
git commit -m "phase2c(agents): LawyerAgent with specialty-selected prompt loading"
```

---

## Task 5: Real Qwen E2E — 民事 Lawyer + statute_search (Civil Code 510)

**Files:**
- Create: `tests/integration/test_lawyer_civil_e2e.py`

The flagship test: a real LawyerAgent (specialty=民事) handles "房东能不能涨我 30% 房租" using real Qwen + real Qdrant + real statute_search.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_lawyer_civil_e2e.py
"""Phase 2c flagship test: real LawyerAgent (民事 specialty) handles a real
rental-dispute query using real Qwen + real Qdrant statute_search.
Skipped if vLLM not reachable."""
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
from multi_agent.tools.retrievers.exact_read import ExactReadTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(),
    reason="Qwen vLLM not running at http://localhost:8000",
)


@pytest.fixture(scope="module")
def civil_index(tmp_path_factory):
    name = f"test_civil_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照合同相关条款或者交易习惯确定。"),
                Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="703",
                      text="租赁合同是出租人将租赁物交付承租人使用、收益，承租人支付租金的合同。"),
                Chunk(doc_id="民法典-720", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="720",
                      text="在租赁期限内因占有、使用租赁物获得的收益，归承租人所有，但是当事人另有约定的除外。"),
                Chunk(doc_id="民法典-188", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="188",
                      text="向人民法院请求保护民事权利的诉讼时效期间为三年。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_civil_lawyer_handles_rental_dispute(civil_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=civil_index["collection"],
        sparse_artifact_path=civil_index["sparse_path"],
    )
    read_article = ExactReadTool(collection_name=civil_index["collection"])
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="我租的房子合同还没到期,房东突然要涨 30% 房租,合法吗?我应该怎么办?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search, read_article],
            model="qwen3.5-9b",
            specialty="民事",
            max_steps=8,
            max_tool_calls=10,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "civil_rental_dispute"},
    )

    assert result["status"] == "ok"

    # Parse the final structured answer
    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    assert out.mode == "consultation"
    assert out.five_section is not None
    assert len(out.five_section.dispute_analysis) > 20
    assert len(out.five_section.applicable_laws) > 20
    assert len(out.five_section.remedy_suggestions) > 20

    # If lawyer cited any articles, they must be from our index (no fabrication)
    indexed_articles = {"民法典-510", "民法典-703", "民法典-720", "民法典-188"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed_articles, (
            f"Lawyer cited {doc_id} which is NOT in our test index. "
            f"This indicates Qwen fabricated a citation."
        )

    # Verify the lawyer actually called statute_search at least once
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    assert len(tool_calls) >= 1, "Lawyer should have called at least one retrieval tool"
```

- [ ] **Step 2: Run the test**

```bash
docker ps | grep legal-rag-qdrant
curl -s http://localhost:8000/v1/models | head -3
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_lawyer_civil_e2e.py -v -s"
```

Expected: PASSED. Time: 30-120s (multi-turn ReAct with real LLM).

The `-s` flag is included so you can see Qwen's intermediate output if a debug `print` is added.

If the test fails:
1. Inspect `runs/<run_id>/events.jsonl` — what did the LLM produce? Did it call the tool?
2. Common issue: Qwen's tool args have malformed JSON. The `parse_json_robust` fallback should handle most, but if a specific Qwen output is breaking parse, capture the raw and consider improving the fallback.
3. Common issue: Qwen produces a `final_answer` not matching `LawyerOutput` schema. Either:
   - Update the skeleton prompt to be MORE explicit about JSON shape
   - Add a re-prompt-on-validation-failure inside `BaseAgent._react_loop` (Phase 2c+ enhancement)
4. If Qwen fabricates a citation (e.g., 民法典-999), the test correctly FAILS — that's a real correctness issue and the prompt's anti-fabrication clause needs strengthening.

Report DONE if it passes. Report DONE_WITH_CONCERNS noting flake rate if it passes sometimes. Report BLOCKED if it consistently fails for a reason that requires plan-level change.

- [ ] **Step 3: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_lawyer_civil_e2e.py
git commit -m "phase2c(integration): LawyerAgent 民事 specialty handles rental dispute end-to-end"
```

---

## Task 6: Multi-Specialty E2E Smoke Tests (劳动 + 交通)

**Files:**
- Create: `tests/integration/test_lawyer_labor_e2e.py`
- Create: `tests/integration/test_lawyer_traffic_e2e.py`

Two more specialty E2Es to validate the prompt-loading pattern works across specialties. These can be lighter than Task 5 — just verify the agent produces valid `LawyerOutput`, no strict assertion on tool usage.

- [ ] **Step 1: Write 劳动 specialty test**

```python
# tests/integration/test_lawyer_labor_e2e.py
"""LawyerAgent 劳动 specialty smoke test against real Qwen.
Tolerant: 劳动合同法 is NOT in our corpus (ADR-15); test only checks that the
agent runs to a valid LawyerOutput without crashing."""
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
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(), reason="Qwen vLLM not running",
)


@pytest.fixture(scope="module")
def labor_index(tmp_path_factory):
    name = f"test_labor_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    # Use 社会保险法 + 工会法 — both ARE in corpus
    docs = [
        Document(
            law_name="中华人民共和国社会保险法", law_short="社会保险法", source_path="t",
            chunks=[
                Chunk(doc_id="社会保险法-1", law_name="中华人民共和国社会保险法",
                      law_short="社会保险法", article_no="1",
                      text="为了规范社会保险关系,维护公民参加社会保险和享受社会保险待遇的合法权益,制定本法。"),
                Chunk(doc_id="社会保险法-58", law_name="中华人民共和国社会保险法",
                      law_short="社会保险法", article_no="58",
                      text="用人单位应当自用工之日起三十日内为其职工向社会保险经办机构申请办理社会保险登记。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_labor_lawyer_produces_valid_output(labor_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=labor_index["collection"],
        sparse_artifact_path=labor_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="公司没给我交社保,我能怎么办?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty="劳动",
            max_steps=6,
            max_tool_calls=8,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "labor"},
    )
    assert result["status"] == "ok"
    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    # Loose checks — labor corpus is incomplete per ADR-15
    assert out.mode == "consultation"
    assert len(out.primary_answer) > 0
```

- [ ] **Step 2: Write 交通 specialty test**

```python
# tests/integration/test_lawyer_traffic_e2e.py
"""LawyerAgent 交通 specialty E2E. Corpus has 道路交通安全法 — good coverage."""
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
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(), reason="Qwen vLLM not running",
)


@pytest.fixture(scope="module")
def traffic_index(tmp_path_factory):
    name = f"test_traffic_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国道路交通安全法", law_short="道路交通安全法",
            source_path="t",
            chunks=[
                Chunk(doc_id="道路交通安全法-76", law_name="中华人民共和国道路交通安全法",
                      law_short="道路交通安全法", article_no="76",
                      text="机动车发生交通事故造成人身伤亡、财产损失的,由保险公司在机动车第三者责任强制保险责任限额范围内予以赔偿。"),
            ],
        ),
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-1208", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="1208",
                      text="机动车发生交通事故造成损害的,依照道路交通安全法律和本法的有关规定承担赔偿责任。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_traffic_lawyer_produces_valid_output(traffic_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=traffic_index["collection"],
        sparse_artifact_path=traffic_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="我开车不小心撞了人,对方住院了,我要承担什么责任?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty="交通",
            max_steps=6,
            max_tool_calls=8,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "traffic"},
    )
    assert result["status"] == "ok"
    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    assert out.mode == "consultation"
    # Citations (if any) should be from indexed articles only
    indexed = {"道路交通安全法-76", "民法典-1208"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated citation: {doc_id}"
```

- [ ] **Step 3: Run both tests**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_lawyer_labor_e2e.py tests/integration/test_lawyer_traffic_e2e.py -v"
```

Expected: 2 passed (~60-180s total).

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_lawyer_labor_e2e.py experiments/multi_agent/tests/integration/test_lawyer_traffic_e2e.py
git commit -m "phase2c(integration): 劳动 + 交通 specialty Lawyer E2E smoke tests"
```

---

## Task 7: Full Suite + Tag Phase 2c

**Files:** none

- [ ] **Step 1: Run full test suite**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -20"
```

Expected: all tests pass. Phase 1 (63) + Phase 2a (~34) + Phase 2b (~21) + Phase 2c (~14) ≈ 132 tests passing, 1 skipped (anthropic E2E).

- [ ] **Step 2: Tag Phase 2c**

```bash
cd /home/xxm/rag
git tag -a phase2c-real-lawyer -m "Phase 2c complete: LawyerAgent with 5-section prompt + ReAct over statute_search"
```

- [ ] **Step 3: Confirm tag**

```bash
git tag -l "phase*"
```

Expected: lists `phase1-walking-skeleton`, `phase2a-statute-retrieval`, `phase2b-real-providers`, `phase2c-real-lawyer`.

---

## Acceptance Criteria

Phase 2c is complete when:

1. Full pytest suite passes (~132 tests, 1 skipped)
2. `test_civil_lawyer_handles_rental_dispute` produces valid `LawyerOutput` with non-empty five_section AND no fabricated citations
3. 劳动 + 交通 specialty smoke tests both produce valid LawyerOutput
4. Streaming usage telemetry is captured (Phase 2b "Important" #1 resolved)
5. Tag `phase2c-real-lawyer` exists

## Out-of-Scope (Reminder)

- **Phase 3**: Receptionist + EntityState/WorkingMemory + multi-issue handling
- **Phase 4**: Secretary as separate agent + business tools
- **Phase 5**: Supervisor + eval + ablations
- **Streaming with tools**: still single-turn streaming only
- **`response_format` Anthropic structured output**: Phase 2c relies on `parse_json_robust` + prompt instructions; native structured output is a later enhancement

## Notes for Implementing Engineer

- **Prompt files use Chinese filenames**: `specialty_民事.md` etc. This works on Linux with UTF-8 locale; verify your shell can handle them.
- **`importlib.resources.files`** is the modern way to load package data; works for both editable and installed packages.
- **Qwen's `parse_json_robust` fallback** is the safety net for ~10-20% of tool calls per ADR-13. If the test flakes, log the raw tool args and consider strengthening the parser, not the prompt.
- **Don't tighten Qwen prompts pre-emptively**: empirical evidence from Phase 2b shows Qwen 9B handled the simple Phase 2b prompt well; the more verbose 5-section skeleton may degrade it. If tests flake, look at trace before adjusting prompts.
- **Test for fabrication explicitly**: the civil + traffic E2Es check that citations come from indexed articles. This is the most important anti-hallucination gate at this layer.
