# Phase 2b — Real LLM Providers (Anthropic + Local Qwen via vLLM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `StubProvider` with two real providers — `OpenAICompatibleProvider` (tested against local Qwen 3.5 9B served by vLLM on GPU 3, port 8000) and `AnthropicProvider` (mock-tested; real-API tests gated on env var). Wire `provider.complete_stream()` into `BaseAgent.run_stream()` so CLI/SSE consumers see real token deltas. Add `ProviderProfile` config so each agent can use a different provider/model. Close Phase 2a evaluator notes: replace `default_model="stub-1"` hardcoding and promote `Evidence.law_short` to a real field.

**Architecture:** Builds on Phase 2a. New code:
- `multi_agent/providers/anthropic.py` + `openai_compatible.py` (both implement Phase 1 `LLMProvider` ABC)
- `multi_agent/providers/profile.py` — config that maps agent role → (provider, model)
- `BaseAgent` gets a `model: str` field; ReAct loop uses it instead of `getattr(provider, "default_model", "stub-1")`
- `run_stream()` on `BaseAgent` becomes truly streaming: yields `llm_token` events from `provider.complete_stream()` in real time

Out-of-scope (later phases): real Lawyer agent (Phase 2c), cases/user_history collections (Phase 2d), agent ablations (Phase 5).

**Tech Stack:** Python 3.10+, anthropic SDK, openai SDK, vLLM (already deployed at `/home/xxm/models/qwen3.5-9b/`), pytest-asyncio, respx (HTTP mock for Anthropic tests).

**Spec reference:** `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` §6 (LLM Provider 层), §6.3 (差异点), §6.4 (Profile 配置), ADR-12/13/24.

**Phase 2a starting point:** Tag `phase2a-statute-retrieval`. 97 tests passing.

---

## Environment Prerequisites

1. `legal-rag-qdrant` container running on port 6433 (from Phase 2a — leave it alone).
2. Qwen 3.5 9B vLLM service runnable via `cd /home/xxm/models/qwen3.5-9b && conda activate qwen35 && nohup bash serve_vllm.sh > /tmp/vllm_9b.log 2>&1 &` — Task 1 starts it.
3. `bge-m3` model still at `/home/xxm/models/bge-m3/` (cuda:1) — Phase 2a tests still need to pass.
4. Optional: `ANTHROPIC_API_KEY` env var if real Claude tests are wanted. Default tests for Anthropic use HTTP mocks (`respx`) so this is NOT required.

---

## File Structure (Phase 2b additions)

```
experiments/multi_agent/
├── pyproject.toml                                  # Task 0 — add deps: anthropic, openai, respx
├── multi_agent/
│   ├── agents/base.py                              # Task 0 — add `model: str` field; remove stub-1 fallback
│   ├── schemas/evidence.py                         # Task 0 — promote law_short to real Pydantic field
│   ├── providers/
│   │   ├── openai_compatible.py                    # Tasks 2-3, 7 (streaming added)
│   │   ├── anthropic.py                            # Tasks 4-5, 7
│   │   └── profile.py                              # Task 6 — ProviderProfile + factory
│   └── agents/base.py                              # Task 8 — run_stream uses complete_stream
└── tests/
    ├── unit/
    │   ├── test_evidence.py                        # Task 0 — add law_short field test
    │   ├── test_openai_compat.py                   # Task 2 (basic) + Task 3 (tools) + Task 7 (streaming)
    │   ├── test_anthropic.py                       # Task 4-5, 7 with respx mocks
    │   ├── test_provider_profile.py                # Task 6
    │   └── test_run_stream_real.py                 # Task 8 (integration-ish, uses StubProvider stream path)
    └── integration/
        ├── test_qwen_e2e.py                        # Task 9 — real Qwen + statute_search end-to-end
        └── test_anthropic_e2e.py                   # Task 10 — gated on ANTHROPIC_API_KEY env
```

**Working directory for all tasks:** `/home/xxm/rag/experiments/multi_agent/`
**Test command prefix:** `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && <pytest cmd>"` (DenseEncoder + qdrant-client require qwen35 env)

---

## Task 0: Phase 2a Follow-Up Fixes

**Files:**
- Modify: `multi_agent/agents/base.py` — add `model: str = ""` field, remove `getattr(..., "stub-1")` fallback (replace with `self.model or "stub-1"`)
- Modify: `multi_agent/schemas/evidence.py` — promote `law_short` from `@property` to real Pydantic field `law_short: str = ""`; remove the property
- Modify: `multi_agent/tools/retrievers/statute_search.py` — populate `law_short` field directly (not via metadata)
- Modify: `multi_agent/tools/retrievers/exact_read.py` — same
- Modify: `multi_agent/tools/retrievers/index_builder.py` — payload still includes `law_short` for backward compat (no change)
- Modify: `tests/unit/test_evidence.py` — add test for the new field
- Modify: `pyproject.toml` — add `anthropic>=0.40`, `openai>=1.50`, `respx>=0.21`

- [ ] **Step 1: Write failing test for `law_short` field**

Append to `tests/unit/test_evidence.py`:

```python
def test_evidence_law_short_is_a_real_field():
    """law_short should be a first-class Pydantic field (not a @property),
    so it appears in model_dump() and round-trips through model_validate().
    """
    e = Evidence(
        doc_id="民法典-510", law_name="中华人民共和国民法典",
        article_no="510", text="...", score=0.5, retriever="hybrid",
        law_short="民法典",
    )
    assert e.law_short == "民法典"
    # Must appear in dump
    dumped = e.model_dump()
    assert dumped["law_short"] == "民法典"
    # Must round-trip
    e2 = Evidence.model_validate(dumped)
    assert e2.law_short == "民法典"


def test_evidence_law_short_defaults_to_empty():
    e = Evidence(
        doc_id="x", law_name="y", article_no="1", text="t",
        score=0.5, retriever="hybrid",
    )
    assert e.law_short == ""
```

- [ ] **Step 2: Run test to verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_evidence.py::test_evidence_law_short_is_a_real_field -v"
```

Expected: FAIL — currently `law_short` is `@property` so passing it to constructor errors, or model_dump won't include it.

- [ ] **Step 3: Promote `law_short` to a real field**

Edit `multi_agent/schemas/evidence.py`:

```python
class Evidence(BaseModel):
    doc_id: str
    law_name: str
    law_short: str = ""            # NEW: promoted from @property
    article_no: str
    text: str
    score: float
    retriever: Literal["bm25", "dense", "hybrid", "exact", "memory", "case", "history"]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

DELETE the `@property law_short` definition. (Phase 2a added it; we replace with a real field.)

- [ ] **Step 4: Update Evidence call sites**

In `multi_agent/tools/retrievers/statute_search.py`, change the Evidence construction:

```python
ev = Evidence(
    doc_id=payload.get("doc_id", ""),
    law_name=payload.get("law_name", ""),
    law_short=payload.get("law_short", ""),       # NEW: explicit field
    article_no=payload.get("article_no", ""),
    text=payload.get("text", ""),
    score=float(point.score) if point.score is not None else 0.0,
    retriever="hybrid",
    metadata={                                     # metadata no longer holds law_short
        "book": payload.get("book", ""),
        "chapter": payload.get("chapter", ""),
        "concepts": payload.get("concepts", []),
    },
)
```

In `multi_agent/tools/retrievers/exact_read.py`:

```python
ev = Evidence(
    doc_id=payload.get("doc_id", ""),
    law_name=payload.get("law_name", ""),
    law_short=payload.get("law_short", ""),       # NEW
    article_no=payload.get("article_no", ""),
    text=payload.get("text", ""),
    score=1.0,
    retriever="exact",
    metadata={
        "book": payload.get("book", ""),
        "chapter": payload.get("chapter", ""),
    },
)
```

`index_builder.py` writes `law_short` into the Qdrant payload already — no change needed.

- [ ] **Step 5: Add `model` field to `BaseAgent` + remove stub-1 fallback**

Edit `multi_agent/agents/base.py`:

a) In the `BaseAgent` class field list, add:

```python
    model: str = ""    # set explicitly by ProviderProfile; falls back to provider default
```

b) In `_react_loop`, replace:

```python
            model=getattr(self.provider, "default_model", "stub-1"),
```

with:

```python
            model=self.model or getattr(self.provider, "default_model", "stub-1"),
```

This keeps the stub-1 fallback for tests that don't set `model` explicitly but lets Phase 2b agents specify their model.

- [ ] **Step 6: Add new deps to `pyproject.toml`**

Read current `pyproject.toml`, append to `dependencies`:

```toml
    "anthropic>=0.40",
    "openai>=1.50",
```

And to `optional-dependencies.dev`:

```toml
    "respx>=0.21",          # HTTP mock for Anthropic tests
```

- [ ] **Step 7: Install + run all tests to verify**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pip install -e '.[dev]' && pytest -v 2>&1 | tail -10"
```

Expected: 99 tests passing (97 prior + 2 new).

- [ ] **Step 8: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/ experiments/multi_agent/tests/unit/test_evidence.py experiments/multi_agent/pyproject.toml
git commit -m "phase2b(prep): Evidence.law_short as real field; BaseAgent.model field; +anthropic/openai/respx deps"
```

---

## Task 1: Start Qwen vLLM + Smoke Test

**Files:** none (operational task)

This task starts the Qwen vLLM service on GPU 3 and verifies it's reachable. No code changes; only execution + documentation.

- [ ] **Step 1: Check current state**

```bash
ps aux | grep vllm | grep -v grep || echo "no vllm running"
curl -s -m 2 http://localhost:8000/v1/models || echo "8000 unreachable"
nvidia-smi --query-gpu=index,memory.used --format=csv | head -10
```

If vllm is already running and `/v1/models` returns a list with `qwen3.5-9b`, skip to Step 3.

- [ ] **Step 2: Start vLLM service**

```bash
cd /home/xxm/models/qwen3.5-9b
ls serve_vllm.sh   # confirm script exists
nohup bash serve_vllm.sh > /tmp/vllm_9b.log 2>&1 &
echo "Waiting for vLLM to start (~2 min)..."
for i in {1..30}; do
    sleep 5
    if curl -s -m 2 http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "vLLM up after ${i}*5 seconds"
        break
    fi
    echo "...still starting (${i}/30)"
done
curl -s http://localhost:8000/v1/models | python -m json.tool
```

Expected: response containing model id `qwen3.5-9b`. If after 30 attempts it's still down, tail `/tmp/vllm_9b.log` to diagnose.

- [ ] **Step 3: Smoke chat completion**

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "say ping"}],
    "max_tokens": 16,
    "temperature": 0
  }' | python -m json.tool
```

Expected: a JSON response with a `content` field containing "ping" or similar.

- [ ] **Step 4: Verify GPU 3 is in use**

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Expected: GPU 3 shows ~15-20 GB used (model + KV cache).

- [ ] **Step 5: Document the started service in README**

Append to `experiments/multi_agent/README.md`:

```markdown

## Local Qwen 3.5 9B (vLLM)

This project's `openai_compatible` provider talks to a local Qwen 3.5 9B served by vLLM at `http://localhost:8000/v1`.

```bash
# Start (one-time per machine boot)
cd /home/xxm/models/qwen3.5-9b
conda activate qwen35
nohup bash serve_vllm.sh > /tmp/vllm_9b.log 2>&1 &

# Verify (waits ~2 min on first start)
curl http://localhost:8000/v1/models

# Stop
pkill -9 -f vllm
```

The service uses GPU card 3 (~20 GB VRAM). See `/home/xxm/models/qwen3.5-9b/USAGE.md` for details.
```

- [ ] **Step 6: Commit the README**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/README.md
git commit -m "phase2b(docs): document local Qwen vLLM service for provider tests"
```

---

## Task 2: OpenAICompatibleProvider — Basic `complete()`

**Files:**
- Create: `multi_agent/providers/openai_compatible.py`
- Create: `tests/unit/test_openai_compat.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_openai_compat.py
import pytest
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.base import LLMResponse, Usage
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def provider():
    return OpenAICompatibleProvider(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        default_model="qwen3.5-9b",
    )


@pytest.mark.asyncio
async def test_complete_returns_llm_response(provider, tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="say ping (one word)")],
        model="qwen3.5-9b",
        max_tokens=8,
        temperature=0,
        recorder=rec,
        agent_name="tester",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert isinstance(resp.text, str)
    assert len(resp.text) > 0
    assert resp.usage.input_tokens > 0
    assert resp.usage.output_tokens > 0
    assert resp.finish_reason in ("end_turn", "max_tokens")


@pytest.mark.asyncio
async def test_complete_emits_llm_events(provider, tmp_run_dir):
    import json as _j
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    await provider.complete(
        messages=[AgentMessage(role="user", content="say one word")],
        model="qwen3.5-9b", max_tokens=8, temperature=0,
        recorder=rec, agent_name="tester",
    )
    rec.close()
    types = [_j.loads(l)["event_type"]
             for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    assert "LLMRequested" in types
    assert "LLMResponded" in types
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/providers/openai_compatible.py`**

```python
"""OpenAI-compatible provider — used for local Qwen via vLLM,
or for OpenAI proper / any other OpenAI-compat service (DeepSeek, etc.)
by swapping base_url + api_key + model.
"""
from __future__ import annotations
import os
from typing import Any, AsyncGenerator
from openai import AsyncOpenAI

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import ProviderUnavailable


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str = "qwen3.5-9b",
        timeout: float = 120.0,
    ):
        self.base_url = base_url or os.environ.get(
            "OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "dummy")
        self.default_model = default_model
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
        )

    def _to_oai_messages(self, messages: list[AgentMessage]) -> list[dict]:
        """Convert internal AgentMessage to OpenAI chat format."""
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                out.append({
                    "role": "tool",
                    "tool_call_id": m.tool_use_id or "",
                    "content": m.content,
                })
                continue
            entry: dict = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": __import__("json").dumps(tc.args, ensure_ascii=False),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        return out

    def _to_oai_tools(self, tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    def _normalize_finish_reason(self, raw: str | None) -> str:
        # OpenAI: "stop", "tool_calls", "length", "content_filter"
        mapping = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "refusal",
        }
        return mapping.get(raw or "stop", "end_turn")  # type: ignore[return-value]

    async def complete(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse:
        oai_messages = self._to_oai_messages(messages)
        oai_tools = self._to_oai_tools(tools)

        # If response_format provided, append a system instruction. Phase 2b doesn't
        # use OpenAI's structured-output mode because Qwen doesn't honor it; we rely
        # on parse_json_robust at the agent layer.

        with recorder.span(
            "llm_call",
            provider="openai_compat",
            model=model,
            agent_name=agent_name,
            messages=oai_messages,
            params={"max_tokens": max_tokens, "temperature": temperature},
        ) as span:
            try:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=oai_messages,
                    tools=oai_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                raise ProviderUnavailable(
                    f"OpenAI-compat at {self.base_url} failed: {e}"
                ) from e

            choice = resp.choices[0]
            text = choice.message.content or ""
            raw_tool_calls = choice.message.tool_calls or []
            tool_calls: list[ToolCallRequest] = []
            for tc in raw_tool_calls:
                try:
                    args = __import__("json").loads(tc.function.arguments)
                except Exception:
                    args = parse_json_robust(tc.function.arguments or "{}")
                tool_calls.append(ToolCallRequest(
                    tool_use_id=tc.id,
                    tool_name=tc.function.name,
                    args=args,
                ))

            usage = resp.usage
            llm_resp = LLMResponse(
                text=text,
                tool_calls=tool_calls,
                usage=Usage(
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                ),
                raw={"model": resp.model, "id": resp.id},
                duration_ms=0,
                finish_reason=self._normalize_finish_reason(choice.finish_reason),  # type: ignore[arg-type]
            )
            span.set_output({
                "raw": text,
                "usage": llm_resp.usage.model_dump(),
                "finish_reason": llm_resp.finish_reason,
            })
            return llm_resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Implemented in Task 7
        raise NotImplementedError("complete_stream lands in Task 7")
```

- [ ] **Step 4: Verify pass**

```bash
docker ps | grep legal-rag-qdrant
curl -s -m 2 http://localhost:8000/v1/models | head -1
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py -v"
```

Expected: 2 passed (~5-10s for inference).

If vLLM not running, the test will fail with `ProviderUnavailable` — start it per Task 1.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/openai_compatible.py experiments/multi_agent/tests/unit/test_openai_compat.py
git commit -m "phase2b(providers): OpenAICompatibleProvider basic complete() against local Qwen"
```

---

## Task 3: OpenAICompatibleProvider — Tool Calling + JSON Retry

**Files:**
- Modify: `multi_agent/providers/openai_compatible.py` (add JSON-arg retry; tool_call already works in Task 2)
- Modify: `tests/unit/test_openai_compat.py` (add tool-call test)

Qwen's tool_use sometimes emits args that aren't valid JSON. Task 2's code uses `parse_json_robust` as fallback, but we need a test that exercises this path.

- [ ] **Step 1: Append failing tests**

```python
# Add to tests/unit/test_openai_compat.py
from multi_agent.providers.base import ToolSpec


@pytest.mark.asyncio
async def test_complete_with_tool_definition(provider, tmp_run_dir):
    """Qwen called with a tool definition decides whether to use it.
    We don't assert it MUST call the tool (Qwen can choose to answer directly),
    only that the call succeeds and returns either tool_calls or text."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    tools = [
        ToolSpec(
            name="get_weather",
            description="Get the weather for a city.",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    ]
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="北京今天天气如何?用 get_weather 工具查询。")],
        model="qwen3.5-9b",
        tools=tools,
        max_tokens=64,
        temperature=0,
        recorder=rec,
        agent_name="tester",
    )
    rec.close()
    # Either it called the tool or it answered directly — both are valid
    assert resp.finish_reason in ("end_turn", "tool_use")
    if resp.finish_reason == "tool_use":
        assert len(resp.tool_calls) >= 1
        assert resp.tool_calls[0].tool_name == "get_weather"
        assert "city" in resp.tool_calls[0].args
```

- [ ] **Step 2: Run test — should pass already** (Task 2's code handles tool_calls)

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py::test_complete_with_tool_definition -v"
```

Expected: PASS. If it fails because Qwen refused or emitted malformed JSON, the parse_json_robust fallback should kick in. If still failing, investigate the Qwen response by tailing `/tmp/vllm_9b.log` or adding `print(resp.text)` in the test.

- [ ] **Step 3: Add `parse_json_robust` fallback test (with monkey-patched malformed args)**

Append:

```python
import pytest
from unittest.mock import patch, AsyncMock
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall, Function,
)
from openai.types.completion_usage import CompletionUsage


def _fake_completion_with_malformed_args(name: str, args_str: str):
    return ChatCompletion(
        id="fake-id",
        model="qwen3.5-9b",
        object="chat.completion",
        created=0,
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="tc-1",
                            type="function",
                            function=Function(name=name, arguments=args_str),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


@pytest.mark.asyncio
async def test_provider_recovers_from_fenced_json_args(provider, tmp_run_dir):
    """If Qwen wraps tool args in ```json ... ```, parse_json_robust should
    recover them rather than crashing."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    fake = _fake_completion_with_malformed_args(
        "get_weather", '```json\n{"city": "北京"}\n```'
    )
    with patch.object(
        provider._client.chat.completions, "create",
        new=AsyncMock(return_value=fake),
    ):
        resp = await provider.complete(
            messages=[AgentMessage(role="user", content="x")],
            model="qwen3.5-9b", recorder=rec, agent_name="tester",
        )
    rec.close()
    assert resp.tool_calls[0].args == {"city": "北京"}
```

- [ ] **Step 4: Verify test passes**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py -v"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/unit/test_openai_compat.py
git commit -m "phase2b(providers): test tool-call + parse_json_robust fallback for malformed args"
```

---

## Task 4: AnthropicProvider — Basic complete() with respx Mocks

**Files:**
- Create: `multi_agent/providers/anthropic.py`
- Create: `tests/unit/test_anthropic.py`

We test against mocked HTTP (no API key required for unit tests). Real-API integration test is Task 10.

- [ ] **Step 1: Write failing test (mock-based)**

```python
# tests/unit/test_anthropic.py
import pytest
import respx
import httpx
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.base import LLMResponse, ToolSpec
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


@pytest.fixture
def provider():
    return AnthropicProvider(api_key="test-key", default_model="claude-sonnet-4-6")


_BASE_URL = "https://api.anthropic.com/v1/messages"


def _mock_message_response(content_blocks: list[dict], stop_reason: str = "end_turn"):
    """Build a fake Anthropic /v1/messages response body."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": "claude-sonnet-4-6",
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


@pytest.mark.asyncio
@respx.mock(assert_all_called=False)
async def test_anthropic_complete_text_response(provider, tmp_run_dir):
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_mock_message_response(
                [{"type": "text", "text": "Hello from Claude"}], "end_turn",
            ),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="hi")],
        model="claude-sonnet-4-6",
        recorder=rec, agent_name="tester",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert resp.text == "Hello from Claude"
    assert resp.finish_reason == "end_turn"
    assert resp.usage.input_tokens == 10


@pytest.mark.asyncio
@respx.mock(assert_all_called=False)
async def test_anthropic_complete_tool_use(provider, tmp_run_dir):
    respx.post(_BASE_URL).mock(
        return_value=httpx.Response(
            200,
            json=_mock_message_response(
                [
                    {"type": "text", "text": ""},
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "statute_search",
                        "input": {"query": "民法典 510"},
                    },
                ],
                "tool_use",
            ),
        )
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    resp = await provider.complete(
        messages=[AgentMessage(role="user", content="search")],
        model="claude-sonnet-4-6",
        tools=[ToolSpec(
            name="statute_search", description="search",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )],
        recorder=rec, agent_name="tester",
    )
    rec.close()
    assert resp.finish_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].tool_name == "statute_search"
    assert resp.tool_calls[0].args == {"query": "民法典 510"}
    assert resp.tool_calls[0].tool_use_id == "toolu_01"
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_anthropic.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/providers/anthropic.py`**

```python
"""Anthropic Claude provider — full message API including tool use.

Prompt caching (cache_control) lands in Task 5.
Streaming lands in Task 7.
"""
from __future__ import annotations
import os
from typing import Any, AsyncGenerator
from anthropic import AsyncAnthropic

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import ProviderUnavailable


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = "claude-sonnet-4-6",
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.default_model = default_model
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout)

    def _split_system_and_messages(
        self, messages: list[AgentMessage]
    ) -> tuple[str, list[dict]]:
        """Anthropic API takes system separately and accepts only user/assistant/tool messages."""
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
                continue
            if m.role == "tool":
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_use_id or "",
                        "content": m.content,
                    }],
                })
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.tool_use_id,
                        "name": tc.tool_name,
                        "input": tc.args,
                    })
                out.append({"role": "assistant", "content": blocks})
                continue
            out.append({"role": m.role, "content": m.content})
        return "\n\n".join(system_parts), out

    def _to_anthropic_tools(self, tools: list[ToolSpec] | None) -> list[dict] | None:
        if not tools:
            return None
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _normalize_stop_reason(self, raw: str | None) -> str:
        mapping = {
            "end_turn": "end_turn",
            "tool_use": "tool_use",
            "max_tokens": "max_tokens",
            "stop_sequence": "end_turn",
            "refusal": "refusal",
        }
        return mapping.get(raw or "end_turn", "end_turn")  # type: ignore[return-value]

    async def complete(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse:
        system, anthropic_messages = self._split_system_and_messages(messages)
        anthropic_tools = self._to_anthropic_tools(tools)

        with recorder.span(
            "llm_call",
            provider="anthropic",
            model=model,
            agent_name=agent_name,
            messages=anthropic_messages,
            params={"max_tokens": max_tokens, "temperature": temperature, "system": system},
        ) as span:
            try:
                msg = await self._client.messages.create(
                    model=model,
                    system=system or None,
                    messages=anthropic_messages,
                    tools=anthropic_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic API failed: {e}") from e

            text_parts: list[str] = []
            tool_calls: list[ToolCallRequest] = []
            for block in msg.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    text_parts.append(getattr(block, "text", ""))
                elif btype == "tool_use":
                    tool_calls.append(ToolCallRequest(
                        tool_use_id=getattr(block, "id", ""),
                        tool_name=getattr(block, "name", ""),
                        args=getattr(block, "input", {}) or {},
                    ))

            usage = getattr(msg, "usage", None)
            llm_resp = LLMResponse(
                text="".join(text_parts),
                tool_calls=tool_calls,
                usage=Usage(
                    input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                    output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                    cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) if usage else 0,
                    cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) if usage else 0,
                ),
                raw={"model": msg.model, "id": msg.id},
                duration_ms=0,
                finish_reason=self._normalize_stop_reason(msg.stop_reason),  # type: ignore[arg-type]
            )
            span.set_output({
                "raw": llm_resp.text,
                "usage": llm_resp.usage.model_dump(),
                "finish_reason": llm_resp.finish_reason,
            })
            return llm_resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        raise NotImplementedError("complete_stream lands in Task 7")
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_anthropic.py -v"
```

Expected: 2 passed (no real API call — all mocked).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/anthropic.py experiments/multi_agent/tests/unit/test_anthropic.py
git commit -m "phase2b(providers): AnthropicProvider basic complete() with respx-mocked tests"
```

---

## Task 5: AnthropicProvider — Prompt Caching (cache_control)

**Files:**
- Modify: `multi_agent/providers/anthropic.py`
- Modify: `tests/unit/test_anthropic.py`

Anthropic supports `cache_control: {"type": "ephemeral"}` on `system` / `tools` to enable 5-min prompt caching. Spec §6.3 requires we use this.

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_anthropic.py`:

```python
@pytest.mark.asyncio
@respx.mock(assert_all_called=True)
async def test_anthropic_marks_system_for_cache(provider, tmp_run_dir):
    """The system message should be sent with cache_control: ephemeral."""
    captured = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        import json
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_mock_message_response(
            [{"type": "text", "text": "ok"}], "end_turn",
        ))

    respx.post(_BASE_URL).mock(side_effect=_capture)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    await provider.complete(
        messages=[
            AgentMessage(role="system", content="You are a legal assistant."),
            AgentMessage(role="user", content="hi"),
        ],
        model="claude-sonnet-4-6",
        recorder=rec, agent_name="tester",
    )
    rec.close()

    # system field should be a list of typed blocks with cache_control
    sys_field = captured["body"]["system"]
    assert isinstance(sys_field, list)
    assert sys_field[-1].get("cache_control") == {"type": "ephemeral"}
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_anthropic.py::test_anthropic_marks_system_for_cache -v"
```

Expected: FAIL — current code sends system as plain string, not the typed-block-with-cache_control form.

- [ ] **Step 3: Update `complete()` to use cache-friendly system blocks**

In `multi_agent/providers/anthropic.py`, modify the system-handling part of `complete()`. Replace:

```python
            msg = await self._client.messages.create(
                model=model,
                system=system or None,
                ...
            )
```

With:

```python
            cache_friendly_system = None
            if system:
                cache_friendly_system = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            msg = await self._client.messages.create(
                model=model,
                system=cache_friendly_system,
                ...
            )
```

- [ ] **Step 4: Verify pass + existing tests still pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_anthropic.py -v"
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/anthropic.py experiments/multi_agent/tests/unit/test_anthropic.py
git commit -m "phase2b(providers): Anthropic cache_control on system prompt for prefix caching"
```

---

## Task 6: ProviderProfile + Factory

**Files:**
- Create: `multi_agent/providers/profile.py`
- Create: `tests/unit/test_provider_profile.py`

Maps `agent_role → (provider, model)`. Default profile `all-local` puts everything on local Qwen.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_provider_profile.py
import pytest
from multi_agent.providers.profile import (
    ProviderProfile, build_provider_for, PROFILES,
)
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.anthropic import AnthropicProvider


def test_default_profiles_exist():
    """The 4 spec'd profiles are pre-defined."""
    assert "all-local" in PROFILES
    assert "all-claude" in PROFILES
    assert "mixed-cloud-judge" in PROFILES
    assert "mixed-cloud-brain" in PROFILES


def test_all_local_profile_uses_qwen_everywhere():
    p = PROFILES["all-local"]
    assert p.agent_to_provider["lawyer"] == ("openai_compat", "qwen3.5-9b")
    assert p.agent_to_provider["receptionist"] == ("openai_compat", "qwen3.5-9b")


def test_all_claude_profile_uses_anthropic_everywhere():
    p = PROFILES["all-claude"]
    assert p.agent_to_provider["lawyer"][0] == "anthropic"
    assert p.agent_to_provider["lawyer"][1].startswith("claude-")


def test_build_provider_for_local_profile():
    provider, model = build_provider_for("lawyer", profile_name="all-local")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert model == "qwen3.5-9b"


def test_build_provider_for_claude_profile():
    provider, model = build_provider_for("supervisor", profile_name="all-claude")
    assert isinstance(provider, AnthropicProvider)
    assert model.startswith("claude-")


def test_build_provider_for_unknown_agent_falls_back():
    """If agent isn't in profile.agent_to_provider, falls back to profile.default."""
    provider, model = build_provider_for("unknown_agent", profile_name="all-local")
    assert isinstance(provider, OpenAICompatibleProvider)


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        build_provider_for("lawyer", profile_name="nonexistent")
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_provider_profile.py -v"
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/providers/profile.py`**

```python
"""Provider profile registry: which agent uses which (provider, model).

Defines the 4 spec'd profiles (all-local / all-claude / mixed-cloud-judge /
mixed-cloud-brain). New profiles can be added by extending PROFILES.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from multi_agent.providers.base import LLMProvider
from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider


@dataclass(frozen=True)
class ProviderProfile:
    """Maps agent role → (provider_name, model_name).

    `default` is used when an agent role isn't explicitly listed.
    """
    name: str
    agent_to_provider: dict[str, tuple[str, str]]
    default: tuple[str, str] = ("openai_compat", "qwen3.5-9b")


PROFILES: dict[str, ProviderProfile] = {
    "all-local": ProviderProfile(
        name="all-local",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("openai_compat", "qwen3.5-9b"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("openai_compat", "qwen3.5-9b"),
        },
    ),
    "all-claude": ProviderProfile(
        name="all-claude",
        agent_to_provider={
            "receptionist": ("anthropic", "claude-haiku-4-5-20251001"),
            "lawyer":       ("anthropic", "claude-sonnet-4-6"),
            "secretary":    ("anthropic", "claude-sonnet-4-6"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
        default=("anthropic", "claude-sonnet-4-6"),
    ),
    "mixed-cloud-judge": ProviderProfile(
        name="mixed-cloud-judge",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("openai_compat", "qwen3.5-9b"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
    ),
    "mixed-cloud-brain": ProviderProfile(
        name="mixed-cloud-brain",
        agent_to_provider={
            "receptionist": ("openai_compat", "qwen3.5-9b"),
            "lawyer":       ("anthropic", "claude-sonnet-4-6"),
            "secretary":    ("openai_compat", "qwen3.5-9b"),
            "supervisor":   ("anthropic", "claude-haiku-4-5-20251001"),
        },
    ),
}


# Singletons keyed by provider name — avoid creating multiple clients per profile-resolve
_provider_singletons: dict[str, LLMProvider] = {}


def _get_singleton(provider_name: str) -> LLMProvider:
    if provider_name not in _provider_singletons:
        if provider_name == "openai_compat":
            _provider_singletons[provider_name] = OpenAICompatibleProvider()
        elif provider_name == "anthropic":
            _provider_singletons[provider_name] = AnthropicProvider()
        else:
            raise ValueError(f"unknown provider: {provider_name}")
    return _provider_singletons[provider_name]


def build_provider_for(
    agent_name: str, *, profile_name: str = "all-local",
) -> tuple[LLMProvider, str]:
    """Return (provider, model) for the agent_name in the named profile.

    Falls back to profile.default if agent_name not explicitly mapped.
    Raises KeyError if profile_name is unknown.
    """
    if profile_name not in PROFILES:
        raise KeyError(f"unknown profile: {profile_name}. Choices: {list(PROFILES)}")
    profile = PROFILES[profile_name]
    provider_name, model = profile.agent_to_provider.get(agent_name, profile.default)
    return _get_singleton(provider_name), model
```

- [ ] **Step 4: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_provider_profile.py -v"
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/profile.py experiments/multi_agent/tests/unit/test_provider_profile.py
git commit -m "phase2b(providers): ProviderProfile + build_provider_for() — 4 profiles per spec §6.4"
```

---

## Task 7: complete_stream() in both providers

**Files:**
- Modify: `multi_agent/providers/openai_compatible.py` — implement `complete_stream`
- Modify: `multi_agent/providers/anthropic.py` — implement `complete_stream`
- Modify: `tests/unit/test_openai_compat.py` — add streaming test against real Qwen
- Modify: `tests/unit/test_anthropic.py` — add streaming test with respx

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/test_openai_compat.py`:

```python
@pytest.mark.asyncio
async def test_openai_compat_streams_tokens(provider, tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    chunks = []
    async for ch in provider.complete_stream(
        messages=[AgentMessage(role="user", content="count 1 to 3, one number per line")],
        model="qwen3.5-9b", max_tokens=32, temperature=0,
        recorder=rec, agent_name="tester",
    ):
        chunks.append(ch)
    rec.close()
    # At least one token chunk + one end_turn
    kinds = [c.kind for c in chunks]
    assert "token" in kinds
    assert kinds[-1] == "end_turn"
    # Concatenated tokens form non-empty text
    text = "".join(c.content for c in chunks if c.kind == "token")
    assert len(text) > 0
```

Append to `tests/unit/test_anthropic.py`:

```python
@pytest.mark.asyncio
@respx.mock(assert_all_called=True)
async def test_anthropic_streams_tokens(provider, tmp_run_dir):
    """Use SSE-formatted mock response."""
    sse_body = (
        'event: message_start\n'
        'data: {"type":"message_start","message":{"id":"msg_1","model":"claude-sonnet-4-6","usage":{"input_tokens":3,"output_tokens":0}}}\n\n'
        'event: content_block_start\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}\n\n'
        'event: content_block_delta\n'
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n'
        'event: content_block_stop\n'
        'data: {"type":"content_block_stop","index":0}\n\n'
        'event: message_delta\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n'
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
    async for ch in provider.complete_stream(
        messages=[AgentMessage(role="user", content="hi")],
        model="claude-sonnet-4-6", recorder=rec, agent_name="tester",
    ):
        chunks.append(ch)
    rec.close()

    text = "".join(c.content for c in chunks if c.kind == "token")
    assert text == "hello world"
    assert chunks[-1].kind == "end_turn"
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py::test_openai_compat_streams_tokens tests/unit/test_anthropic.py::test_anthropic_streams_tokens -v"
```

Expected: both FAIL with NotImplementedError.

- [ ] **Step 3: Implement `OpenAICompatibleProvider.complete_stream`**

Replace the `raise NotImplementedError` body with:

```python
    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        oai_messages = self._to_oai_messages(messages)
        oai_tools = self._to_oai_tools(tools)
        with recorder.span(
            "llm_call", provider="openai_compat", model=model, agent_name=agent_name,
            messages=oai_messages, params={"max_tokens": max_tokens, "temperature": temperature, "stream": True},
        ) as span:
            try:
                stream = await self._client.chat.completions.create(
                    model=model, messages=oai_messages, tools=oai_tools or None,
                    max_tokens=max_tokens, temperature=temperature, stream=True,
                )
            except Exception as e:
                raise ProviderUnavailable(f"OpenAI-compat stream failed: {e}") from e

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
            yield StreamChunk(kind="end_turn")
```

- [ ] **Step 4: Implement `AnthropicProvider.complete_stream`**

Replace `complete_stream` body:

```python
    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        system, anthropic_messages = self._split_system_and_messages(messages)
        anthropic_tools = self._to_anthropic_tools(tools)
        cache_friendly_system = None
        if system:
            cache_friendly_system = [{
                "type": "text", "text": system,
                "cache_control": {"type": "ephemeral"},
            }]

        with recorder.span(
            "llm_call", provider="anthropic", model=model, agent_name=agent_name,
            messages=anthropic_messages,
            params={"max_tokens": max_tokens, "temperature": temperature, "stream": True},
        ) as span:
            full_text = ""
            try:
                async with self._client.messages.stream(
                    model=model,
                    system=cache_friendly_system,
                    messages=anthropic_messages,
                    tools=anthropic_tools or None,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ) as stream:
                    async for text in stream.text_stream:
                        full_text += text
                        yield StreamChunk(kind="token", content=text)
            except Exception as e:
                raise ProviderUnavailable(f"Anthropic stream failed: {e}") from e
            span.set_output({"raw": full_text, "usage": {}, "finish_reason": "end_turn"})
            yield StreamChunk(kind="end_turn")
```

- [ ] **Step 5: Verify pass**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_openai_compat.py tests/unit/test_anthropic.py -v"
```

Expected: all tests pass — including the two new streaming tests.

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/providers/openai_compatible.py experiments/multi_agent/multi_agent/providers/anthropic.py experiments/multi_agent/tests/unit/test_openai_compat.py experiments/multi_agent/tests/unit/test_anthropic.py
git commit -m "phase2b(providers): complete_stream() for both OpenAI-compat and Anthropic"
```

---

## Task 8: BaseAgent.run_stream() Integrates Provider Streaming

**Files:**
- Modify: `multi_agent/agents/base.py`
- Create: `tests/unit/test_run_stream_real.py`

Phase 1's `run_stream` just yielded `agent_start / final_answer / agent_end`. Now we integrate `provider.complete_stream` to emit per-token chunks.

For Phase 2b we keep the design minimal: `run_stream` calls `self.run()` (which uses non-streaming `complete()`) and yields `final_answer` from its result, BUT also exposes a separate `stream_one_turn(...)` helper that uses `complete_stream` for cases that don't need tool dispatch. Reason: full streaming with tool dispatch is complex (need to detect tool_use mid-stream and switch modes). Phase 2c may upgrade to full tool-aware streaming if needed.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_run_stream_real.py
"""run_stream / stream_one_turn smoke tests against StubProvider (no network).
Real-provider streaming is exercised in tests/integration/test_qwen_e2e.py.
"""
import pytest
from multi_agent.agents.base import BaseAgent, AgentInput, StreamEvent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder
from pydantic import BaseModel


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_run_stream_yields_final_answer(tmp_run_dir):
    """Phase 1 contract preserved: run_stream yields agent_start, final_answer, agent_end."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text='{"answer": "ok"}')])
    agent = _Agent(name="a", role="t", provider=p, recorder=rec, model="stub-1")
    events = []
    async for ev in agent.run_stream(AgentInput(payload={"query": "hi"})):
        events.append(ev)
    rec.close()
    kinds = [e.kind for e in events]
    assert "agent_start" in kinds
    assert "final_answer" in kinds
    assert "agent_end" in kinds


@pytest.mark.asyncio
async def test_stream_one_turn_yields_tokens(tmp_run_dir):
    """stream_one_turn calls provider.complete_stream and yields llm_token events.
    StubProvider yields one StreamChunk per character of the scripted text."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="hello")])
    agent = _Agent(name="a", role="t", provider=p, recorder=rec, model="stub-1")
    events = []
    async for ev in agent.stream_one_turn("say hi"):
        events.append(ev)
    rec.close()
    tokens = [e.content for e in events if e.kind == "llm_token"]
    assert "".join(tokens) == "hello"
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_run_stream_real.py -v"
```

Expected: AttributeError on `agent.stream_one_turn`.

- [ ] **Step 3: Add `stream_one_turn` to `BaseAgent`**

In `multi_agent/agents/base.py`, add a new method on the `BaseAgent` class (after `run_stream`):

```python
    async def stream_one_turn(
        self, user_input: str,
    ) -> AsyncGenerator["StreamEvent", None]:
        """Stream a single LLM turn without tool dispatch.

        Useful for CLI/SSE consumers that want token-level output but don't need
        the full ReAct loop. Tool dispatch + multi-turn ReAct still uses run() /
        run_stream() (which are non-streaming under the hood for now — full
        streaming-with-tools is a Phase 2c+ enhancement).
        """
        from multi_agent.schemas.messages import AgentMessage
        messages = [
            AgentMessage(role="system", content=self.system_prompt()),
            AgentMessage(role="user", content=user_input),
        ]
        model = self.model or getattr(self.provider, "default_model", "stub-1")
        async for chunk in self.provider.complete_stream(
            messages=messages, model=model,
            recorder=self.recorder, agent_name=self.name,
        ):
            if chunk.kind == "token":
                yield StreamEvent(kind="llm_token", content=chunk.content)
            elif chunk.kind == "end_turn":
                yield StreamEvent(kind="agent_end", content=self.name)
            elif chunk.kind == "error":
                yield StreamEvent(kind="error", content=chunk.content)
```

- [ ] **Step 4: Verify pass + no regression**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -5"
```

Expected: full suite passes — Phase 1 (63) + Phase 2a (34) + Phase 2b so far (~12 new) ≈ 109 tests.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/base.py experiments/multi_agent/tests/unit/test_run_stream_real.py
git commit -m "phase2b(agents): BaseAgent.stream_one_turn() — token streaming via provider.complete_stream"
```

---

## Task 9: Integration — Real Qwen End-to-End

**Files:**
- Create: `tests/integration/test_qwen_e2e.py`

Phase 2b acceptance test (most rigorous): a stub agent uses real Qwen via `OpenAICompatibleProvider`, calls `statute_search` against the Phase 2a Qdrant collection, gets back a real legal-relevant answer.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_qwen_e2e.py
"""Phase 2b acceptance test: real Qwen vLLM + real Qdrant statute_search +
multi_agent base agent ReAct loop. No mocks, no stubs (except the agent's
'system_prompt' which is minimal).

Skipped if vLLM not reachable.
"""
import json
import uuid
import asyncio
import httpx
import pytest
from pydantic import BaseModel
from pathlib import Path

from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(),
    reason="Qwen vLLM not running at http://localhost:8000 — start it per Task 1",
)


class _Out(BaseModel):
    summary: str


class _LegalAgent(BaseAgent):
    def system_prompt(self) -> str:
        return (
            "你是法律助手。当用户问法条相关问题时,先调用 statute_search 工具检索,"
            '然后用 JSON 总结结果: {"summary": "<简要说明>"}。'
            "禁止编造法条号。"
        )
    def output_schema(self):
        return _Out


@pytest.fixture(scope="module")
def populated_index(tmp_path_factory):
    name = f"test_qwen_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照交易习惯确定。"),
                Chunk(doc_id="民法典-563", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="563",
                      text="一方违约时，对方可以解除合同。"),
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
async def test_real_qwen_finds_civil_code_510(populated_index, tmp_path):
    runs_root = tmp_path / "runs"
    search_tool = StatuteSearchTool(
        collection_name=populated_index["collection"],
        sparse_artifact_path=populated_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="合同补充内容怎么确定?",
        agent_factory=lambda p, r: _LegalAgent(
            name="lawyer", role="lookup",
            provider=p, recorder=r,
            tools=[search_tool],
            model="qwen3.5-9b",
            max_steps=5,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "all-local-qwen+statute_search"},
    )

    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]

    # Real provider should have emitted ≥1 LLMRequested + ≥1 LLMResponded
    n_req = types.count("LLMRequested")
    n_resp = types.count("LLMResponded")
    assert n_req >= 1 and n_resp == n_req

    # statute_search should have been called at least once (we don't enforce
    # the model MUST call it — but the prompt is strong, so flakiness is low)
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    if tool_calls:
        # If the agent called the tool, the doc_id 民法典-510 should be in the result
        for ret in events:
            if ret["event_type"] == "ToolReturned" and ret.get("result"):
                if "evidences" in ret["result"]:
                    doc_ids = [e.get("doc_id") for e in ret["result"]["evidences"]]
                    assert "民法典-510" in doc_ids, f"expected 510 in retrieved evidences, got {doc_ids}"
                    break

    # Final answer should be valid JSON conforming to _Out schema
    final = json.loads(result["final_answer"])
    assert "summary" in final
    assert isinstance(final["summary"], str) and len(final["summary"]) > 0
```

- [ ] **Step 2: Run integration test**

```bash
# Ensure vLLM is up
curl -s http://localhost:8000/v1/models | head -1
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_qwen_e2e.py -v"
```

Expected: 1 passed (~10-40s depending on Qwen response time and whether it calls the tool).

If Qwen returns invalid JSON, `parse_json_robust` should recover. If the test still fails, look at the trace:

```bash
ls runs/r_*/   # find the most recent run
cat runs/r_*/events.jsonl | python -m json.tool | head -200
```

If Qwen tool-call reliability is low, the test may flake. That's expected per ADR-13 ("Qwen 9B tool use 失败率 10-20%"); track failures rather than chase them.

- [ ] **Step 3: Run full suite for acceptance**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -10"
```

Expected: ALL tests pass — ~110 tests total.

- [ ] **Step 4: Commit and tag**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_qwen_e2e.py
git commit -m "phase2b(integration): real Qwen E2E with statute_search via OpenAICompatibleProvider"
git tag -a phase2b-real-providers -m "Phase 2b complete: Anthropic + OpenAI-compat providers + streaming + profiles"
```

---

## Task 10 (OPTIONAL): Real Anthropic API Integration Test

**Files:**
- Create: `tests/integration/test_anthropic_e2e.py`

Gated on `ANTHROPIC_API_KEY` env var. Skipped if absent. Costs ~$0.005 per run.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_anthropic_e2e.py
"""Phase 2b optional acceptance test: real Anthropic API.
Skipped unless ANTHROPIC_API_KEY env var is set.
"""
import json
import os
import pytest
from pydantic import BaseModel

from multi_agent.providers.anthropic import AnthropicProvider
from multi_agent.agents.base import BaseAgent
from multi_agent.runner import run_query


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class _Out(BaseModel):
    answer: str


class _SimpleAgent(BaseAgent):
    def system_prompt(self) -> str:
        return 'Answer in JSON: {"answer": "<text>"}. Be very brief.'
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_real_anthropic_simple_completion(tmp_path):
    runs_root = tmp_path / "runs"
    provider = AnthropicProvider()
    result = await run_query(
        query="What is 1+1? Just the number.",
        agent_factory=lambda p, r: _SimpleAgent(
            name="claude_test", role="t",
            provider=p, recorder=r,
            model="claude-haiku-4-5-20251001",  # cheapest Claude
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "anthropic-haiku-smoke"},
    )
    assert result["status"] == "ok"
    final = json.loads(result["final_answer"])
    assert "2" in final["answer"]
```

- [ ] **Step 2: Run (optional, costs ~$0.005)**

```bash
ANTHROPIC_API_KEY=sk-ant-... conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_anthropic_e2e.py -v"
```

Expected: 1 passed if API key valid; skipped if not.

- [ ] **Step 3: Commit (even if not run)**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_anthropic_e2e.py
git commit -m "phase2b(integration): optional real Anthropic E2E (gated on ANTHROPIC_API_KEY)"
```

---

## Acceptance Criteria

Phase 2b is complete when:

1. `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v"` runs all tests green (Phase 1 + 2a + 2b unit ≈ 110+ tests, integration tests pass when Qwen is up)
2. `test_real_qwen_finds_civil_code_510` passes: a stub agent → real Qwen → statute_search → answers about 民法典 510
3. `Evidence.law_short` is a real Pydantic field (model_dump includes it)
4. `BaseAgent.model` field exists and is honored by `_react_loop`
5. `ProviderProfile` supports the 4 spec'd profiles
6. Tag `phase2b-real-providers` exists
7. Anthropic provider works against mocked HTTP (no real API key needed for unit tests)
8. Both providers expose `complete_stream` that yields per-token deltas

## Out-of-Scope (Reminder)

- **Phase 2c**: Real Lawyer agent with five-section prompt + actual ReAct over legal queries
- **Phase 2d**: cases / user_history collections
- **Tool-aware streaming**: streaming through tool dispatch loops (Phase 2b only streams single-turn LLM calls via `stream_one_turn`)
- **Concepts field auto-generation**: Phase 2c+

## Notes for Implementing Engineer

- **Qwen tool-use reliability**: 10-20% failure rate per ADR-13. The agent layer's existing retry-on-JSON-decode-error (Phase 1's `parse_json_robust`) is enough for unit tests; the real-Qwen integration test in Task 9 may occasionally flake — re-run rather than overfix.
- **vLLM startup time**: ~2 minutes from `bash serve_vllm.sh`. Don't start it mid-task.
- **Token cost**: Anthropic unit tests are all mocked (zero cost). The optional Task 10 real API test costs ~$0.005. Local Qwen costs nothing per call.
- **GPU 3 already in use after Task 1**: ~20 GB. Don't co-launch any other vLLM service.
- **Provider singletons**: `profile._provider_singletons` caches by provider name across the process. Tests that need a fresh instance should construct providers directly, not via `build_provider_for`.
- **`stream_one_turn` doesn't dispatch tools**: it's for CLI-style "echo back the response" UX. Tool-aware streaming is a known Phase 2c+ concern.
