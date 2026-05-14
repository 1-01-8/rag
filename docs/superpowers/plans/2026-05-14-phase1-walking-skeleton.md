# Phase 1 — Walking Skeleton (Trace + Stub Agent + Asyncio) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable end-to-end walking skeleton — one stub query goes through one stub agent backed by a stub LLM provider, producing a complete trace file. No real LLM, no real retrieval. The point is to lock down all cross-cutting interfaces (trace, schemas, async ReAct, error invariants) before any business logic enters.

**Architecture:** Pure Python + Pydantic. Strict layering: `schemas/` → `tracing/` + `providers/` + `tools/` → `agents/` → `runner`. All execution is async (`asyncio`). All errors emit trace events. Every run produces a `runs/<run_id>/` directory with `events.jsonl + events.db + meta.json`.

**Tech Stack:** Python 3.10+, Pydantic 2.x, python-ulid, aiosqlite, pytest, pytest-asyncio.

**Spec reference:** `docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` §0–§3, §8, §9.

---

## File Structure (Phase 1 only)

```
experiments/multi_agent/
├── pyproject.toml                          # Task 0
├── .gitignore                              # Task 0
├── README.md                               # Task 0
├── multi_agent/                            # Python package
│   ├── __init__.py
│   ├── errors.py                           # Task 1  — shared exceptions
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── events.py                       # Task 2-4 — Pydantic discriminated union
│   │   ├── messages.py                     # Task 5 — AgentMessage / ToolCall / ToolResult
│   │   ├── evidence.py                     # Task 6 — Evidence
│   │   ├── state.py                        # Task 7 — RunState
│   │   └── working_memory.py               # Task 8 — WorkingMemory + Hypothesis
│   ├── tracing/
│   │   ├── __init__.py
│   │   ├── ulid_gen.py                     # Task 9 — fresh_event_id / fresh_run_id
│   │   ├── jsonl_writer.py                 # Task 10 — append-only JSONL
│   │   ├── sqlite_indexer.py               # Task 11 — events.db
│   │   └── recorder.py                     # Task 12-14 — Recorder + span
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── json_robust.py                  # Task 15 — parse_json_robust
│   │   ├── base.py                         # Task 16 — LLMProvider ABC + LLMResponse
│   │   └── stub.py                         # Task 17 — StubProvider for testing
│   ├── tools/
│   │   ├── __init__.py
│   │   └── base.py                         # Task 18 — Tool ABC (async)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                         # Task 19-22 — BaseAgent + ReAct + budgets + stream
│   │   └── stub_echo.py                    # Task 23 — concrete stub agent for E2E
│   └── runner.py                           # Task 24 — run_query top-level entry
└── tests/
    ├── __init__.py
    ├── conftest.py                         # Task 0
    ├── unit/
    │   ├── __init__.py
    │   ├── test_events.py                  # Task 2-4
    │   ├── test_messages.py                # Task 5
    │   ├── test_evidence.py                # Task 6
    │   ├── test_state.py                   # Task 7
    │   ├── test_working_memory.py          # Task 8
    │   ├── test_ulid_gen.py                # Task 9
    │   ├── test_jsonl_writer.py            # Task 10
    │   ├── test_sqlite_indexer.py          # Task 11
    │   ├── test_recorder.py                # Task 12-14
    │   ├── test_json_robust.py             # Task 15
    │   ├── test_stub_provider.py           # Task 17
    │   ├── test_tool_base.py               # Task 18
    │   ├── test_agent_base.py              # Task 19-22
    │   └── test_stub_echo_agent.py         # Task 23
    └── integration/
        ├── __init__.py
        └── test_walking_skeleton.py        # Task 25
```

**Working directory for all tasks:** `/home/xxm/rag/experiments/multi_agent/`

---

## Task 0: Project Bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `multi_agent/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "multi-agent-legal-rag"
version = "0.1.0"
description = "Experimental multi-agent legal RAG (Phase 1: walking skeleton)"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.5",
    "python-ulid>=2.0",
    "aiosqlite>=0.19",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["multi_agent*"]
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.venv/
venv/

# runtime data (not committed)
runs/
run_groups/
memory_store/
qdrant_storage/
*.tmp
```

- [ ] **Step 3: Create `README.md`**

```markdown
# Multi-Agent Legal RAG (experimental)

Phase 1: walking skeleton — trace system + stub agent + asyncio.

See `docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` for full design.

## Run tests
```
pip install -e ".[dev]"
pytest -v
```
```

- [ ] **Step 4: Create empty package init files**

```bash
touch multi_agent/__init__.py tests/__init__.py
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Fresh run directory for each test."""
    d = tmp_path / "runs" / "test-run-0001"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

- [ ] **Step 6: Install in editable mode and verify**

```bash
cd /home/xxm/rag/experiments/multi_agent
pip install -e ".[dev]"
pytest --collect-only
```

Expected: pytest collects 0 tests (no test files yet) without import errors.

- [ ] **Step 7: Commit**

```bash
git add experiments/multi_agent/
git commit -m "phase1(bootstrap): initial project structure + deps"
```

---

## Task 1: Shared Errors

**Files:**
- Create: `multi_agent/errors.py`

- [ ] **Step 1: Write `multi_agent/errors.py`**

```python
"""Centralized exception types. All raised from agents/providers/tools."""


class MultiAgentError(Exception):
    """Base for all package errors."""


class ProviderUnavailable(MultiAgentError):
    """LLM provider unreachable or auth failed."""


class ResponseValidationError(MultiAgentError):
    """LLM response failed schema validation after retries."""

    def __init__(self, message: str, raw: str | None = None):
        super().__init__(message)
        self.raw = raw


class ToolCallParseError(MultiAgentError):
    """Tool args from LLM did not match args_schema."""


class BudgetExceeded(MultiAgentError):
    """Agent exceeded max_steps / max_total_tokens / max_tool_calls."""

    def __init__(self, agent_name: str, budget: str, limit: int):
        super().__init__(f"{agent_name} exceeded {budget}={limit}")
        self.agent_name = agent_name
        self.budget = budget
        self.limit = limit


class AgentTimeout(MultiAgentError):
    """Agent wall-clock exceeded."""


class MemoryReadError(MultiAgentError):
    """memory_store file read/parse failure."""


class MemoryWriteError(MultiAgentError):
    """memory_store write failed."""
```

- [ ] **Step 2: Smoke-import test**

Run: `python -c "from multi_agent.errors import BudgetExceeded; raise BudgetExceeded('x', 'max_steps', 10)"`
Expected: traceback ends with `multi_agent.errors.BudgetExceeded: x exceeded max_steps=10`

- [ ] **Step 3: Commit**

```bash
git add multi_agent/errors.py
git commit -m "phase1(errors): centralized exception types"
```

---

## Task 2: BaseEvent + RunStarted/RunFinished

**Files:**
- Create: `multi_agent/schemas/__init__.py`
- Create: `multi_agent/schemas/events.py`
- Create: `tests/unit/test_events.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_events.py
from datetime import datetime, timezone
from multi_agent.schemas.events import BaseEvent, RunStarted, RunFinished


def test_base_event_required_fields():
    """BaseEvent must carry id, run_id, timestamp, parent_id, event_type."""
    e = RunStarted(
        event_id="01EVENT",
        run_id="01RUN",
        timestamp=datetime.now(timezone.utc),
        parent_id=None,
        query="test query",
        config={"profile": "stub"},
    )
    assert e.event_id == "01EVENT"
    assert e.run_id == "01RUN"
    assert e.parent_id is None
    assert e.event_type == "RunStarted"
    assert e.query == "test query"


def test_run_finished_ok_status():
    e = RunFinished(
        event_id="01EVT", run_id="01RUN",
        timestamp=datetime.now(timezone.utc), parent_id=None,
        status="ok",
        final_answer="42",
        error=None,
    )
    assert e.event_type == "RunFinished"
    assert e.status == "ok"
    assert e.final_answer == "42"


def test_run_finished_rejects_unknown_status():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunFinished(
            event_id="e", run_id="r",
            timestamp=datetime.now(timezone.utc), parent_id=None,
            status="bogus",  # not in Literal
            final_answer=None, error=None,
        )
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_events.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'multi_agent.schemas.events'`.

- [ ] **Step 3: Create `multi_agent/schemas/__init__.py` and `events.py`**

```python
# multi_agent/schemas/__init__.py
```

```python
# multi_agent/schemas/events.py
from __future__ import annotations
from datetime import datetime
from typing import Literal, Any
from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Common fields for every trace event."""
    event_id: str
    run_id: str
    timestamp: datetime
    parent_id: str | None = None
    event_type: str  # subclasses override with Literal

    model_config = {"frozen": False, "extra": "forbid"}


class RunStarted(BaseEvent):
    event_type: Literal["RunStarted"] = "RunStarted"
    query: str
    config: dict[str, Any] = Field(default_factory=dict)


class RunFinished(BaseEvent):
    event_type: Literal["RunFinished"] = "RunFinished"
    status: Literal["ok", "error", "timeout"]
    final_answer: str | None = None
    error: str | None = None
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_events.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/ tests/unit/test_events.py
git commit -m "phase1(schemas): BaseEvent + RunStarted/RunFinished"
```

---

## Task 3: Agent / LLM / Tool / Memory Event Types

**Files:**
- Modify: `multi_agent/schemas/events.py`
- Modify: `tests/unit/test_events.py`

- [ ] **Step 1: Write failing tests for additional event types**

Append to `tests/unit/test_events.py`:

```python
from datetime import datetime, timezone
from multi_agent.schemas.events import (
    AgentInvoked, AgentResponded,
    LLMRequested, LLMResponded,
    ToolCalled, ToolReturned,
    MemoryRead, MemoryWritten,
    SupervisorVerdict,
)


def _ts():
    return datetime.now(timezone.utc)


def test_agent_invoked_and_responded():
    inv = AgentInvoked(
        event_id="e1", run_id="r", timestamp=_ts(), parent_id=None,
        agent_name="lawyer", role="primary_advisor",
        input={"query": "hi"},
    )
    assert inv.event_type == "AgentInvoked"
    assert inv.agent_name == "lawyer"

    resp = AgentResponded(
        event_id="e2", run_id="r", timestamp=_ts(), parent_id="e1",
        agent_name="lawyer", output={"answer": "hello"}, duration_ms=1234,
    )
    assert resp.duration_ms == 1234


def test_llm_request_response():
    req = LLMRequested(
        event_id="l1", run_id="r", timestamp=_ts(), parent_id="e1",
        provider="stub", model="stub-1", messages=[{"role": "user", "content": "hi"}],
        params={"temperature": 0},
    )
    resp = LLMResponded(
        event_id="l2", run_id="r", timestamp=_ts(), parent_id="l1",
        raw_response="ok", usage={"input_tokens": 5, "output_tokens": 1},
        duration_ms=10, finish_reason="end_turn",
    )
    assert req.event_type == "LLMRequested"
    assert resp.finish_reason == "end_turn"


def test_tool_called_and_returned():
    call = ToolCalled(
        event_id="t1", run_id="r", timestamp=_ts(), parent_id="e1",
        tool_name="search", args={"q": "x"}, agent_name="secretary",
    )
    ret = ToolReturned(
        event_id="t2", run_id="r", timestamp=_ts(), parent_id="t1",
        result={"hits": 3}, error=None, duration_ms=8,
    )
    assert call.tool_name == "search"
    assert ret.error is None


def test_memory_events():
    rd = MemoryRead(
        event_id="m1", run_id="r", timestamp=_ts(), parent_id=None,
        target="sticky", query={"intent": "full"}, hits=[{"path": "x"}],
        agent_name="lawyer",
    )
    wr = MemoryWritten(
        event_id="m2", run_id="r", timestamp=_ts(), parent_id=None,
        target="agent_notes", payload={"name": "n"}, path="agent_notes/n.md",
        agent_name="supervisor",
    )
    assert rd.target == "sticky"
    assert wr.path == "agent_notes/n.md"


def test_supervisor_verdict():
    v = SupervisorVerdict(
        event_id="v1", run_id="r", timestamp=_ts(), parent_id=None,
        verdict="pass", issues=[],
    )
    assert v.verdict == "pass"
```

- [ ] **Step 2: Run tests to verify failure**

```bash
pytest tests/unit/test_events.py -v
```

Expected: ImportError on the new event types.

- [ ] **Step 3: Add event types to `multi_agent/schemas/events.py`**

Append to existing file:

```python
# --- Agent events ---

class AgentInvoked(BaseEvent):
    event_type: Literal["AgentInvoked"] = "AgentInvoked"
    agent_name: str
    role: str
    input: dict[str, Any] = Field(default_factory=dict)


class AgentResponded(BaseEvent):
    event_type: Literal["AgentResponded"] = "AgentResponded"
    agent_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int


# --- LLM events ---

class LLMRequested(BaseEvent):
    event_type: Literal["LLMRequested"] = "LLMRequested"
    provider: str
    model: str
    messages: list[dict[str, Any]]
    params: dict[str, Any] = Field(default_factory=dict)


class LLMResponded(BaseEvent):
    event_type: Literal["LLMResponded"] = "LLMResponded"
    raw_response: str
    usage: dict[str, int] = Field(default_factory=dict)
    duration_ms: int
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal"]


# --- Tool events ---

class ToolCalled(BaseEvent):
    event_type: Literal["ToolCalled"] = "ToolCalled"
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    agent_name: str


class ToolReturned(BaseEvent):
    event_type: Literal["ToolReturned"] = "ToolReturned"
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int


# --- Memory events ---

class MemoryRead(BaseEvent):
    event_type: Literal["MemoryRead"] = "MemoryRead"
    target: Literal["sticky", "turn", "agent_notes", "user_history"]
    query: dict[str, Any] = Field(default_factory=dict)
    hits: list[dict[str, Any]] = Field(default_factory=list)
    agent_name: str


class MemoryWritten(BaseEvent):
    event_type: Literal["MemoryWritten"] = "MemoryWritten"
    target: str
    payload: dict[str, Any]
    path: str
    agent_name: str


# --- Supervisor ---

class SupervisorVerdict(BaseEvent):
    event_type: Literal["SupervisorVerdict"] = "SupervisorVerdict"
    verdict: Literal["pass", "revise", "reject"]
    issues: list[str] = Field(default_factory=list)
    suggested_fix: str | None = None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/unit/test_events.py -v
```

Expected: 8 passed (3 original + 5 new).

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/events.py tests/unit/test_events.py
git commit -m "phase1(schemas): agent/llm/tool/memory/supervisor event types"
```

---

## Task 4: AnyEvent Discriminated Union + Serialization

**Files:**
- Modify: `multi_agent/schemas/events.py`
- Modify: `tests/unit/test_events.py`

- [ ] **Step 1: Write failing test for discriminated union**

Append to `tests/unit/test_events.py`:

```python
from multi_agent.schemas.events import AnyEvent, event_from_dict
import json


def test_dump_and_reload_via_union():
    """Any event can be dumped to JSON and reloaded via the union."""
    original = AgentInvoked(
        event_id="e", run_id="r", timestamp=_ts(), parent_id=None,
        agent_name="lawyer", role="primary", input={},
    )
    j = original.model_dump_json()
    raw = json.loads(j)
    reloaded = event_from_dict(raw)
    assert isinstance(reloaded, AgentInvoked)
    assert reloaded.agent_name == "lawyer"


def test_event_type_discriminates():
    raw = {
        "event_type": "RunFinished",
        "event_id": "e", "run_id": "r",
        "timestamp": _ts().isoformat(),
        "parent_id": None, "status": "ok",
        "final_answer": "x", "error": None,
    }
    obj = event_from_dict(raw)
    assert isinstance(obj, RunFinished)
    assert obj.final_answer == "x"


def test_event_unknown_type_raises():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        event_from_dict({"event_type": "WhoKnows", "event_id": "e",
                         "run_id": "r", "timestamp": _ts().isoformat()})
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_events.py::test_dump_and_reload_via_union -v
```

Expected: ImportError for `AnyEvent` / `event_from_dict`.

- [ ] **Step 3: Add union + factory to `multi_agent/schemas/events.py`**

Append at bottom of file:

```python
from typing import Annotated, Union
from pydantic import TypeAdapter, Field as PydField


AnyEvent = Annotated[
    Union[
        RunStarted, RunFinished,
        AgentInvoked, AgentResponded,
        LLMRequested, LLMResponded,
        ToolCalled, ToolReturned,
        MemoryRead, MemoryWritten,
        SupervisorVerdict,
    ],
    PydField(discriminator="event_type"),
]

_event_adapter: TypeAdapter[AnyEvent] = TypeAdapter(AnyEvent)


def event_from_dict(raw: dict) -> AnyEvent:
    """Parse a dict into the correct event subclass via event_type discriminator."""
    return _event_adapter.validate_python(raw)
```

- [ ] **Step 4: Run all event tests to verify pass**

```bash
pytest tests/unit/test_events.py -v
```

Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/events.py tests/unit/test_events.py
git commit -m "phase1(schemas): AnyEvent discriminated union + event_from_dict"
```

---

## Task 5: Message / ToolCall / ToolResult Schemas

**Files:**
- Create: `multi_agent/schemas/messages.py`
- Create: `tests/unit/test_messages.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_messages.py
from multi_agent.schemas.messages import (
    AgentMessage, ToolCallRequest, ToolResult,
)


def test_agent_message_basic():
    m = AgentMessage(role="user", content="hi")
    assert m.role == "user"
    assert m.content == "hi"


def test_agent_message_with_tool_calls():
    m = AgentMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCallRequest(tool_use_id="t1", tool_name="search", args={"q": "x"})
        ],
    )
    assert len(m.tool_calls) == 1
    assert m.tool_calls[0].tool_name == "search"


def test_tool_result_payload_or_error():
    ok = ToolResult(tool_use_id="t1", payload={"hits": 3}, error=None)
    err = ToolResult(tool_use_id="t2", payload=None, error="boom")
    assert ok.payload == {"hits": 3}
    assert err.error == "boom"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_messages.py -v
```

Expected: ImportError on `multi_agent.schemas.messages`.

- [ ] **Step 3: Create `multi_agent/schemas/messages.py`**

```python
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """LLM-issued request to invoke a tool."""
    tool_use_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Result of dispatching a tool. Exactly one of payload/error is set."""
    tool_use_id: str
    payload: dict[str, Any] | None = None
    error: str | None = None


class AgentMessage(BaseModel):
    """A single message in the agent's conversation context."""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_use_id: str | None = None  # set when role == "tool"
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_messages.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/messages.py tests/unit/test_messages.py
git commit -m "phase1(schemas): AgentMessage / ToolCallRequest / ToolResult"
```

---

## Task 6: Evidence Schema

**Files:**
- Create: `multi_agent/schemas/evidence.py`
- Create: `tests/unit/test_evidence.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_evidence.py
from multi_agent.schemas.evidence import Evidence


def test_evidence_fields():
    e = Evidence(
        doc_id="民法典-510",
        law_name="中华人民共和国民法典",
        article_no="510",
        text="当事人就合同补充内容...",
        score=0.85,
        retriever="hybrid",
        metadata={"book": "合同编"},
    )
    assert e.doc_id == "民法典-510"
    assert e.score == 0.85
    assert e.retriever == "hybrid"


def test_evidence_rejects_unknown_retriever():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Evidence(
            doc_id="x", law_name="y", article_no="1", text="t",
            score=0.5, retriever="banana",
        )
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_evidence.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/evidence.py`**

```python
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    doc_id: str
    law_name: str
    article_no: str
    text: str
    score: float
    retriever: Literal["bm25", "dense", "hybrid", "exact", "memory", "case", "history"]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_evidence.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/evidence.py tests/unit/test_evidence.py
git commit -m "phase1(schemas): Evidence schema"
```

---

## Task 7: RunState Schema

**Files:**
- Create: `multi_agent/schemas/state.py`
- Create: `tests/unit/test_state.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_state.py
from multi_agent.schemas.state import RunState


def test_run_state_defaults():
    s = RunState(
        run_id="r1",
        session_id="s1",
        user_query="test query",
    )
    assert s.run_id == "r1"
    assert s.history_messages == []
    assert s.failed_queries == []


def test_run_state_round_trip():
    s = RunState(
        run_id="r1",
        session_id="s1",
        user_query="q",
        history_messages=[{"role": "user", "content": "prev"}],
    )
    raw = s.model_dump()
    restored = RunState.model_validate(raw)
    assert restored.history_messages[0]["content"] == "prev"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_state.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/state.py`**

```python
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class RunState(BaseModel):
    """Top-level state for a single run. Passed by value through agents.

    Phase 1: minimal. Will be expanded in later phases with
    retrieval / memory / planner fields.
    """
    run_id: str
    session_id: str
    user_query: str
    history_messages: list[dict[str, Any]] = Field(default_factory=list)
    failed_queries: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_state.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/state.py tests/unit/test_state.py
git commit -m "phase1(schemas): RunState"
```

---

## Task 8: WorkingMemory Schema

**Files:**
- Create: `multi_agent/schemas/working_memory.py`
- Create: `tests/unit/test_working_memory.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_working_memory.py
from multi_agent.schemas.working_memory import WorkingMemory, Hypothesis
from multi_agent.schemas.evidence import Evidence


def _ev(doc_id="d1"):
    return Evidence(doc_id=doc_id, law_name="x", article_no="1",
                    text="t", score=0.5, retriever="hybrid")


def test_working_memory_starts_empty():
    wm = WorkingMemory()
    assert wm.retrieved_evidence == []
    assert wm.discarded_evidence == []
    assert wm.hypotheses == []


def test_add_evidence_appends():
    wm = WorkingMemory()
    wm.add_evidence(_ev("d1"))
    wm.add_evidence(_ev("d2"))
    assert {e.doc_id for e in wm.retrieved_evidence} == {"d1", "d2"}


def test_discard_records_reason():
    wm = WorkingMemory()
    e = _ev("d1")
    wm.discard(e, reason="not on-point")
    assert len(wm.discarded_evidence) == 1
    assert wm.discarded_evidence[0].reason == "not on-point"
    assert wm.discarded_evidence[0].evidence.doc_id == "d1"


def test_hypothesis_active_to_rejected():
    h = Hypothesis(
        statement="user can refuse rent hike",
        supporting_evidence=["d1"],
        confidence=0.7,
        status="active",
    )
    assert h.status == "active"
    h.status = "rejected"
    assert h.status == "rejected"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_working_memory.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/working_memory.py`**

```python
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field
from multi_agent.schemas.evidence import Evidence


class Hypothesis(BaseModel):
    statement: str
    supporting_evidence: list[str] = Field(default_factory=list)  # evidence doc_ids
    confidence: float
    status: Literal["active", "verified", "rejected"] = "active"


class DiscardedEvidence(BaseModel):
    evidence: Evidence
    reason: str


class WorkingMemory(BaseModel):
    """Run-internal scratchpad shared between agents in one run.

    Written to trace artifacts on RunFinished; NOT persisted to memory_store.
    """
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    retrieved_evidence: list[Evidence] = Field(default_factory=list)
    discarded_evidence: list[DiscardedEvidence] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    intermediate_drafts: list[str] = Field(default_factory=list)

    def add_evidence(self, e: Evidence) -> None:
        self.retrieved_evidence.append(e)

    def discard(self, e: Evidence, reason: str) -> None:
        self.discarded_evidence.append(DiscardedEvidence(evidence=e, reason=reason))
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_working_memory.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/schemas/working_memory.py tests/unit/test_working_memory.py
git commit -m "phase1(schemas): WorkingMemory + Hypothesis + DiscardedEvidence"
```

---

## Task 9: ULID Generation Helpers

**Files:**
- Create: `multi_agent/tracing/__init__.py`
- Create: `multi_agent/tracing/ulid_gen.py`
- Create: `tests/unit/test_ulid_gen.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_ulid_gen.py
import time
from multi_agent.tracing.ulid_gen import fresh_event_id, fresh_run_id


def test_event_id_is_26_chars():
    eid = fresh_event_id()
    assert isinstance(eid, str)
    assert len(eid) == 26  # ULID standard


def test_event_ids_monotonic():
    a = fresh_event_id()
    time.sleep(0.001)
    b = fresh_event_id()
    assert b > a  # ULIDs sort lexicographically by time


def test_run_id_has_prefix():
    rid = fresh_run_id()
    assert rid.startswith("r_")
    assert len(rid) == 28  # "r_" + 26
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_ulid_gen.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tracing/__init__.py` and `ulid_gen.py`**

```python
# multi_agent/tracing/__init__.py
```

```python
# multi_agent/tracing/ulid_gen.py
from __future__ import annotations
from ulid import ULID


def fresh_event_id() -> str:
    """Monotonic 26-char ULID for trace events."""
    return str(ULID())


def fresh_run_id() -> str:
    """Run identifier with 'r_' prefix for visual recognition."""
    return f"r_{ULID()}"
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_ulid_gen.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/__init__.py multi_agent/tracing/ulid_gen.py tests/unit/test_ulid_gen.py
git commit -m "phase1(tracing): ULID generators for event_id/run_id"
```

---

## Task 10: JSONL Writer

**Files:**
- Create: `multi_agent/tracing/jsonl_writer.py`
- Create: `tests/unit/test_jsonl_writer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_jsonl_writer.py
import json
from datetime import datetime, timezone
from multi_agent.tracing.jsonl_writer import JsonlEventWriter
from multi_agent.schemas.events import RunStarted, RunFinished


def _ts():
    return datetime.now(timezone.utc)


def test_appends_one_line_per_event(tmp_path):
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.write(RunStarted(event_id="e1", run_id="r", timestamp=_ts(),
                       parent_id=None, query="q", config={}))
    w.write(RunFinished(event_id="e2", run_id="r", timestamp=_ts(),
                        parent_id=None, status="ok",
                        final_answer="a", error=None))
    w.close()

    lines = p.read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["event_type"] == "RunStarted"
    assert e2["event_type"] == "RunFinished"


def test_flushes_on_each_write(tmp_path):
    """Even before close(), the file should contain written events."""
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.write(RunStarted(event_id="e1", run_id="r", timestamp=_ts(),
                       parent_id=None, query="q", config={}))
    assert p.exists()
    assert "RunStarted" in p.read_text()
    w.close()


def test_close_is_idempotent(tmp_path):
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.close()
    w.close()  # must not raise
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_jsonl_writer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tracing/jsonl_writer.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import TextIO
from multi_agent.schemas.events import BaseEvent


class JsonlEventWriter:
    """Append-only JSONL writer. Flushes after every write so trace
    survives crashes and can be tailed live."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh: TextIO | None = open(path, "a", encoding="utf-8")

    def write(self, event: BaseEvent) -> None:
        if self._fh is None:
            raise RuntimeError("writer already closed")
        line = event.model_dump_json()
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_jsonl_writer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/jsonl_writer.py tests/unit/test_jsonl_writer.py
git commit -m "phase1(tracing): append-only JSONL event writer with eager flush"
```

---

## Task 11: SQLite Event Indexer

**Files:**
- Create: `multi_agent/tracing/sqlite_indexer.py`
- Create: `tests/unit/test_sqlite_indexer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_sqlite_indexer.py
import sqlite3
from datetime import datetime, timezone
from multi_agent.tracing.sqlite_indexer import SqliteEventIndexer
from multi_agent.schemas.events import RunStarted, AgentInvoked


def _ts():
    return datetime.now(timezone.utc)


def test_indexer_creates_events_table(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.close()

    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "events" in tables
    conn.close()


def test_index_writes_event_row(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.index(RunStarted(event_id="e1", run_id="r1", timestamp=_ts(),
                         parent_id=None, query="q", config={}))
    idx.index(AgentInvoked(event_id="e2", run_id="r1", timestamp=_ts(),
                           parent_id="e1", agent_name="lawyer",
                           role="primary", input={}))
    idx.close()

    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT event_id, run_id, event_type, agent_name FROM events ORDER BY event_id")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("e1", "r1", "RunStarted", None)
    assert rows[1] == ("e2", "r1", "AgentInvoked", "lawyer")
    conn.close()


def test_indexer_close_idempotent(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.close()
    idx.close()  # must not raise
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_sqlite_indexer.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tracing/sqlite_indexer.py`**

```python
from __future__ import annotations
import sqlite3
from pathlib import Path
from multi_agent.schemas.events import BaseEvent


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    parent_id  TEXT,
    timestamp  TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run        ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type       ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent      ON events(agent_name);
CREATE INDEX IF NOT EXISTS idx_events_parent     ON events(parent_id);
"""


class SqliteEventIndexer:
    """Indexes events into per-run SQLite for fast structured queries.

    Writes synchronously after each emit; close() flushes/commits.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def index(self, event: BaseEvent) -> None:
        if self._conn is None:
            raise RuntimeError("indexer already closed")
        agent_name = getattr(event, "agent_name", None)
        self._conn.execute(
            "INSERT OR REPLACE INTO events (event_id, run_id, parent_id, timestamp, event_type, agent_name, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.run_id,
                event.parent_id,
                event.timestamp.isoformat(),
                event.event_type,
                agent_name,
                event.model_dump_json(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_sqlite_indexer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/sqlite_indexer.py tests/unit/test_sqlite_indexer.py
git commit -m "phase1(tracing): per-run SQLite event indexer with type/agent/parent indexes"
```

---

## Task 12: Recorder — Basic emit() with Double Write

**Files:**
- Create: `multi_agent/tracing/recorder.py`
- Create: `tests/unit/test_recorder.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_recorder.py
import json
import sqlite3
from multi_agent.tracing.recorder import Recorder
from multi_agent.schemas.events import RunStarted, RunFinished


def test_recorder_writes_to_both_jsonl_and_sqlite(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    rec.emit(RunStarted(event_id="e1", run_id="r1",
                        timestamp=rec.now(), parent_id=None,
                        query="q", config={}))
    rec.emit(RunFinished(event_id="e2", run_id="r1",
                         timestamp=rec.now(), parent_id=None,
                         status="ok", final_answer="a", error=None))
    rec.close()

    lines = (tmp_run_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2

    conn = sqlite3.connect(tmp_run_dir / "events.db")
    cur = conn.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == 2
    conn.close()


def test_recorder_assigns_run_id_to_events(tmp_run_dir):
    """If event.run_id is set we trust it; recorder still validates consistency."""
    import pytest
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with pytest.raises(ValueError):
        rec.emit(RunStarted(event_id="e", run_id="WRONG",
                            timestamp=rec.now(), parent_id=None,
                            query="q", config={}))
    rec.close()
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: ImportError on `Recorder`.

- [ ] **Step 3: Create `multi_agent/tracing/recorder.py`** (basic version)

```python
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from multi_agent.schemas.events import BaseEvent
from multi_agent.tracing.jsonl_writer import JsonlEventWriter
from multi_agent.tracing.sqlite_indexer import SqliteEventIndexer
from multi_agent.tracing.ulid_gen import fresh_event_id


class Recorder:
    """Central trace recorder. Writes every event to both JSONL and SQLite.

    Span context manager will be added in Task 13.
    """

    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = JsonlEventWriter(self.run_dir / "events.jsonl")
        self._sqlite = SqliteEventIndexer(self.run_dir / "events.db")
        self._closed = False

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def emit(self, event: BaseEvent) -> None:
        if self._closed:
            raise RuntimeError("recorder already closed")
        if event.run_id != self.run_id:
            raise ValueError(
                f"event.run_id={event.run_id!r} does not match recorder.run_id={self.run_id!r}"
            )
        self._jsonl.write(event)
        self._sqlite.index(event)

    def close(self) -> None:
        if self._closed:
            return
        self._jsonl.close()
        self._sqlite.close()
        self._closed = True

    def fresh_event_id(self) -> str:
        return fresh_event_id()
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/recorder.py tests/unit/test_recorder.py
git commit -m "phase1(tracing): Recorder with double-write and run_id invariant"
```

---

## Task 13: Recorder — span() Context Manager + parent_id Stack

**Files:**
- Modify: `multi_agent/tracing/recorder.py`
- Modify: `tests/unit/test_recorder.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_recorder.py`:

```python
from multi_agent.schemas.events import AgentInvoked, AgentResponded, LLMRequested


def test_span_emits_start_and_end_events(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="primary") as span:
        span.set_input({"query": "hi"})
        span.attach(AgentResponded(
            event_id=rec.fresh_event_id(), run_id="r1",
            timestamp=rec.now(), parent_id=span.span_id,
            agent_name="lawyer", output={"a": 1}, duration_ms=42,
        ))
    rec.close()

    import json
    lines = [json.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    types = [l["event_type"] for l in lines]
    # Expected: AgentInvoked (auto), AgentResponded (attached) - one span = one start, attach is independent
    assert "AgentInvoked" in types
    assert "AgentResponded" in types


def test_span_nesting_sets_parent_id(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="x") as outer:
        outer_id = outer.span_id
        with rec.span("llm_call", agent_name="lawyer",
                      provider="stub", model="m") as inner:
            assert inner.parent_id == outer_id
    rec.close()


def test_span_records_duration(tmp_run_dir):
    import time
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="x") as span:
        time.sleep(0.01)
    rec.close()

    # Look for AgentResponded with duration_ms > 0
    import json
    lines = [json.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responses = [l for l in lines if l["event_type"] == "AgentResponded"]
    assert len(responses) == 1
    assert responses[0]["duration_ms"] >= 10
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: AttributeError on `rec.span`.

- [ ] **Step 3: Add span() to `multi_agent/tracing/recorder.py`**

Append inside the `Recorder` class (after `fresh_event_id`):

```python
    def span(self, kind: str, **attrs) -> "_SpanCM":
        """Context manager that emits a start event on enter and a matching end event on exit.

        kind ∈ {"agent_invoke", "llm_call", "tool_call"} (extensible).
        attrs go to the start event's required fields (see _SpanCM._build_start).
        """
        return _SpanCM(recorder=self, kind=kind, attrs=attrs)
```

Then add this class at module level below `Recorder`:

```python
from time import monotonic
from multi_agent.schemas.events import (
    AgentInvoked, AgentResponded,
    LLMRequested, LLMResponded,
    ToolCalled, ToolReturned,
)


class _SpanCM:
    """Span context manager. Emits {kind}_start on __enter__,
    {kind}_end on __exit__. Tracks parent_id from outer spans via a per-recorder stack."""

    def __init__(self, recorder: "Recorder", kind: str, attrs: dict):
        self.recorder = recorder
        self.kind = kind
        self.attrs = attrs
        self.span_id: str = ""
        self.parent_id: str | None = None
        self._t0: float = 0.0
        self._input: dict | None = None
        self._output: dict | None = None
        self._error: str | None = None
        # NOTE: parent stack lives on recorder; see Recorder._span_stack init below

    def __enter__(self):
        self.span_id = self.recorder.fresh_event_id()
        self.parent_id = self.recorder.current_parent_id()
        self.recorder.push_span(self.span_id)
        self._t0 = monotonic()
        start = self._build_start()
        self.recorder.emit(start)
        return self

    def __exit__(self, exc_type, exc, tb):
        duration_ms = int((monotonic() - self._t0) * 1000)
        if exc is not None:
            self._error = f"{exc_type.__name__}: {exc}"
        end = self._build_end(duration_ms)
        self.recorder.emit(end)
        self.recorder.pop_span(self.span_id)
        return False  # never swallow

    def set_input(self, payload: dict) -> None:
        self._input = payload

    def set_output(self, payload: dict) -> None:
        self._output = payload

    def attach(self, event: BaseEvent) -> None:
        """Emit an arbitrary event, parented to this span."""
        self.recorder.emit(event)

    def _build_start(self) -> BaseEvent:
        ts = self.recorder.now()
        if self.kind == "agent_invoke":
            return AgentInvoked(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                agent_name=self.attrs.get("agent_name", ""),
                role=self.attrs.get("role", ""),
                input=self._input or {},
            )
        if self.kind == "llm_call":
            return LLMRequested(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                provider=self.attrs.get("provider", ""),
                model=self.attrs.get("model", ""),
                messages=self.attrs.get("messages", []),
                params=self.attrs.get("params", {}),
            )
        if self.kind == "tool_call":
            return ToolCalled(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                tool_name=self.attrs.get("tool_name", ""),
                args=self.attrs.get("args", {}),
                agent_name=self.attrs.get("agent_name", ""),
            )
        raise ValueError(f"unknown span kind: {self.kind}")

    def _build_end(self, duration_ms: int) -> BaseEvent:
        ts = self.recorder.now()
        end_id = self.recorder.fresh_event_id()
        if self.kind == "agent_invoke":
            return AgentResponded(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                agent_name=self.attrs.get("agent_name", ""),
                output=self._output or ({"error": self._error} if self._error else {}),
                duration_ms=duration_ms,
            )
        if self.kind == "llm_call":
            return LLMResponded(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                raw_response="" if self._output is None else str(self._output.get("raw", "")),
                usage=(self._output or {}).get("usage", {}),
                duration_ms=duration_ms,
                finish_reason=(self._output or {}).get("finish_reason", "end_turn"),
            )
        if self.kind == "tool_call":
            return ToolReturned(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                result=self._output, error=self._error, duration_ms=duration_ms,
            )
        raise ValueError(f"unknown span kind: {self.kind}")
```

Also modify `Recorder.__init__` to initialize span stack, and add helpers. Replace the existing `__init__` and add three methods:

```python
    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = JsonlEventWriter(self.run_dir / "events.jsonl")
        self._sqlite = SqliteEventIndexer(self.run_dir / "events.db")
        self._closed = False
        self._span_stack: list[str] = []

    def current_parent_id(self) -> str | None:
        return self._span_stack[-1] if self._span_stack else None

    def push_span(self, span_id: str) -> None:
        self._span_stack.append(span_id)

    def pop_span(self, span_id: str) -> None:
        if not self._span_stack or self._span_stack[-1] != span_id:
            raise RuntimeError(f"span stack corrupted; expected {span_id}")
        self._span_stack.pop()
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/recorder.py tests/unit/test_recorder.py
git commit -m "phase1(tracing): Recorder.span() with parent_id stack and duration"
```

---

## Task 14: Recorder — meta.json + Finalize Invariant

**Files:**
- Modify: `multi_agent/tracing/recorder.py`
- Modify: `tests/unit/test_recorder.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_recorder.py`:

```python
import json as _json


def test_meta_written_on_close(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    rec.set_meta(query="user query goes here", config={"profile": "stub"})
    rec.close()

    meta = _json.loads((tmp_run_dir / "meta.json").read_text())
    assert meta["run_id"] == "r1"
    assert meta["query"] == "user query goes here"
    assert meta["config"]["profile"] == "stub"
    assert "started_at" in meta
    assert "finished_at" in meta


def test_emit_after_close_raises(tmp_run_dir):
    import pytest
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    rec.close()
    with pytest.raises(RuntimeError):
        rec.emit(RunStarted(event_id="e", run_id="r1",
                            timestamp=rec.now(), parent_id=None,
                            query="q", config={}))
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: AttributeError on `rec.set_meta`.

- [ ] **Step 3: Add `set_meta` and meta.json writing to `Recorder`**

In `__init__`, after `self._span_stack = []`, add:

```python
        self._meta: dict = {"run_id": run_id, "started_at": self.now().isoformat()}
```

Add method:

```python
    def set_meta(self, **fields) -> None:
        self._meta.update(fields)
```

Modify `close()`:

```python
    def close(self) -> None:
        if self._closed:
            return
        self._meta["finished_at"] = self.now().isoformat()
        (self.run_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._jsonl.close()
        self._sqlite.close()
        self._closed = True
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_recorder.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tracing/recorder.py tests/unit/test_recorder.py
git commit -m "phase1(tracing): Recorder writes meta.json on close + post-close emit raises"
```

---

## Task 15: parse_json_robust

**Files:**
- Create: `multi_agent/providers/__init__.py`
- Create: `multi_agent/providers/json_robust.py`
- Create: `tests/unit/test_json_robust.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_json_robust.py
import pytest
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.errors import ResponseValidationError


def test_plain_json():
    assert parse_json_robust('{"a": 1}') == {"a": 1}


def test_strips_fenced_json():
    raw = '```json\n{"a": 1}\n```'
    assert parse_json_robust(raw) == {"a": 1}


def test_strips_generic_fence():
    raw = '```\n{"a": 1}\n```'
    assert parse_json_robust(raw) == {"a": 1}


def test_locates_json_in_prose():
    raw = "Here is the answer:\n{\"a\": 1}\nThanks!"
    assert parse_json_robust(raw) == {"a": 1}


def test_invalid_json_raises_with_raw():
    with pytest.raises(ResponseValidationError) as exc:
        parse_json_robust("not json at all")
    assert exc.value.raw == "not json at all"


def test_empty_raises():
    with pytest.raises(ResponseValidationError):
        parse_json_robust("")
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_json_robust.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create files**

```python
# multi_agent/providers/__init__.py
```

```python
# multi_agent/providers/json_robust.py
from __future__ import annotations
import json
from multi_agent.errors import ResponseValidationError


def parse_json_robust(raw: str) -> dict:
    """Tolerant JSON parsing for LLM output.

    Strategy:
      1. strip leading/trailing whitespace
      2. remove ```json ... ``` or ``` ... ``` fences
      3. locate outermost { ... }
      4. json.loads
    Raises ResponseValidationError with .raw on failure.
    """
    if not raw or not raw.strip():
        raise ResponseValidationError("empty response", raw=raw)

    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[: -len("```")]
    cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ResponseValidationError(f"JSON parse failed: {e}", raw=raw)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_json_robust.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/providers/__init__.py multi_agent/providers/json_robust.py tests/unit/test_json_robust.py
git commit -m "phase1(providers): parse_json_robust handles fenced JSON / prose-wrapped JSON"
```

---

## Task 16: LLMProvider ABC + LLMResponse + StreamChunk

**Files:**
- Create: `multi_agent/providers/base.py`

- [ ] **Step 1: Write `multi_agent/providers/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal
from pydantic import BaseModel, Field

from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


class ToolSpec(BaseModel):
    """Lightweight tool definition for LLM tool-use APIs."""
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    parsed: Any | None = None  # Pydantic model instance if response_format used
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    raw: dict[str, Any] = Field(default_factory=dict)
    duration_ms: int = 0
    finish_reason: Literal["end_turn", "tool_use", "max_tokens", "refusal"] = "end_turn"


class StreamChunk(BaseModel):
    kind: Literal["token", "tool_call_start", "tool_call_args", "end_turn", "error"]
    content: str = ""
    tool_use_id: str | None = None
    tool_name: str | None = None


class LLMProvider(ABC):
    """All concrete providers (Anthropic, OpenAI-compatible) implement this."""

    @abstractmethod
    async def complete(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        response_format: type[BaseModel] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> LLMResponse: ...

    @abstractmethod
    async def complete_stream(
        self,
        messages: list[AgentMessage],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        recorder: Recorder,
        agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]: ...
```

- [ ] **Step 2: Smoke-import test**

Run: `python -c "from multi_agent.providers.base import LLMProvider, LLMResponse, StreamChunk; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add multi_agent/providers/base.py
git commit -m "phase1(providers): LLMProvider ABC + LLMResponse + StreamChunk"
```

---

## Task 17: StubProvider — Scripted Responses for Testing

**Files:**
- Create: `multi_agent/providers/stub.py`
- Create: `tests/unit/test_stub_provider.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_stub_provider.py
import pytest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.providers.base import LLMResponse, Usage
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_stub_returns_scripted_text(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[
        ScriptedResponse(text="hello world", finish_reason="end_turn"),
    ])
    resp = await p.complete(
        messages=[AgentMessage(role="user", content="hi")],
        model="stub-1", recorder=rec, agent_name="lawyer",
    )
    rec.close()
    assert isinstance(resp, LLMResponse)
    assert resp.text == "hello world"
    assert resp.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_stub_emits_llm_events(tmp_run_dir):
    import json as _json
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="x")])
    await p.complete(messages=[AgentMessage(role="user", content="hi")],
                     model="stub-1", recorder=rec, agent_name="lawyer")
    rec.close()
    types = [_json.loads(l)["event_type"]
             for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    assert "LLMRequested" in types
    assert "LLMResponded" in types


@pytest.mark.asyncio
async def test_stub_returns_tool_calls(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(
        text="",
        tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "hi"})],
        finish_reason="tool_use",
    )])
    resp = await p.complete(messages=[AgentMessage(role="user", content="x")],
                            model="stub-1", recorder=rec, agent_name="lawyer")
    rec.close()
    assert resp.finish_reason == "tool_use"
    assert resp.tool_calls[0].tool_name == "echo"


@pytest.mark.asyncio
async def test_stub_exhausted_raises(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text="x")])
    await p.complete(messages=[], model="m", recorder=rec, agent_name="a")
    with pytest.raises(RuntimeError, match="exhausted"):
        await p.complete(messages=[], model="m", recorder=rec, agent_name="a")
    rec.close()
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_stub_provider.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/providers/stub.py`**

```python
from __future__ import annotations
from typing import AsyncGenerator
from pydantic import BaseModel, Field

from multi_agent.providers.base import (
    LLMProvider, LLMResponse, StreamChunk, ToolSpec, Usage,
)
from multi_agent.schemas.messages import AgentMessage, ToolCallRequest
from multi_agent.tracing.recorder import Recorder


class ScriptedResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = "end_turn"
    usage_input_tokens: int = 0
    usage_output_tokens: int = 0


class StubProvider(LLMProvider):
    """Returns pre-scripted responses for testing. No network."""

    def __init__(self, responses: list[ScriptedResponse]):
        self._responses = list(responses)
        self._idx = 0

    async def complete(
        self, messages, *, model, tools=None, response_format=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> LLMResponse:
        if self._idx >= len(self._responses):
            raise RuntimeError("StubProvider scripted responses exhausted")
        scripted = self._responses[self._idx]
        self._idx += 1

        with recorder.span(
            "llm_call",
            provider="stub", model=model, agent_name=agent_name,
            messages=[m.model_dump() for m in messages],
            params={"max_tokens": max_tokens, "temperature": temperature},
        ) as span:
            resp = LLMResponse(
                text=scripted.text,
                tool_calls=scripted.tool_calls,
                usage=Usage(input_tokens=scripted.usage_input_tokens,
                            output_tokens=scripted.usage_output_tokens),
                duration_ms=0,
                finish_reason=scripted.finish_reason,  # type: ignore[arg-type]
                raw={"scripted": True},
            )
            span.set_output({"raw": scripted.text,
                             "usage": resp.usage.model_dump(),
                             "finish_reason": scripted.finish_reason})
            return resp

    async def complete_stream(
        self, messages, *, model, tools=None,
        max_tokens=4096, temperature=0.0, recorder: Recorder, agent_name: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        if self._idx >= len(self._responses):
            raise RuntimeError("StubProvider scripted responses exhausted")
        scripted = self._responses[self._idx]
        self._idx += 1
        for ch in scripted.text:
            yield StreamChunk(kind="token", content=ch)
        for tc in scripted.tool_calls:
            yield StreamChunk(kind="tool_call_start",
                              tool_use_id=tc.tool_use_id, tool_name=tc.tool_name)
        yield StreamChunk(kind="end_turn")
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_stub_provider.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/providers/stub.py tests/unit/test_stub_provider.py
git commit -m "phase1(providers): StubProvider with scripted responses + LLM events"
```

---

## Task 18: Tool ABC (async)

**Files:**
- Create: `multi_agent/tools/__init__.py`
- Create: `multi_agent/tools/base.py`
- Create: `tests/unit/test_tool_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_tool_base.py
import pytest
from pydantic import BaseModel
from multi_agent.tools.base import Tool, ToolSpec
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder


class EchoArgs(BaseModel):
    msg: str


class EchoTool(Tool):
    name: str = "echo"
    description: str = "echo back the message"
    args_schema: type[BaseModel] = EchoArgs

    async def call(self, args: EchoArgs, recorder: Recorder) -> ToolResult:
        return ToolResult(tool_use_id="t-internal", payload={"echo": args.msg})


@pytest.mark.asyncio
async def test_tool_call_returns_result(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    t = EchoTool()
    result = await t.call(EchoArgs(msg="hi"), rec)
    rec.close()
    assert result.payload == {"echo": "hi"}


def test_tool_spec_exposes_input_schema():
    t = EchoTool()
    spec = t.to_spec()
    assert spec.name == "echo"
    assert "msg" in spec.input_schema["properties"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_tool_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create files**

```python
# multi_agent/tools/__init__.py
```

```python
# multi_agent/tools/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder


class ToolSpec(BaseModel):
    """JSON-schema-style tool description shown to LLM."""
    name: str
    description: str
    input_schema: dict


class Tool(BaseModel, ABC):
    name: str
    description: str
    args_schema: type[BaseModel]

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    async def call(self, args: BaseModel, recorder: Recorder) -> ToolResult: ...

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.args_schema.model_json_schema(),
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_tool_base.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/tools/__init__.py multi_agent/tools/base.py tests/unit/test_tool_base.py
git commit -m "phase1(tools): Tool ABC (async) + ToolSpec"
```

---

## Task 19: BaseAgent Skeleton (constructor + abstract methods)

**Files:**
- Create: `multi_agent/agents/__init__.py`
- Create: `multi_agent/agents/base.py`
- Create: `tests/unit/test_agent_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_agent_base.py
import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput, AgentOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


class _EchoOutput(BaseModel):
    answer: str


class _DummyAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "you are a test agent"

    def output_schema(self):
        return _EchoOutput


@pytest.mark.asyncio
async def test_agent_construction(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    p = StubProvider(responses=[ScriptedResponse(text='{"answer": "hi"}')])
    agent = _DummyAgent(
        name="dummy", role="test",
        provider=p, recorder=rec,
    )
    rec.close()
    assert agent.name == "dummy"
    assert agent.max_steps == 10  # default
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create files**

```python
# multi_agent/agents/__init__.py
```

```python
# multi_agent/agents/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field

from multi_agent.providers.base import LLMProvider
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class AgentInput(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentOutput(BaseModel):
    payload: BaseModel
    steps_used: int

    model_config = {"arbitrary_types_allowed": True}


class BaseAgent(BaseModel, ABC):
    """Template-method base. Subclasses only override system_prompt() and output_schema()."""
    name: str
    role: str
    provider: LLMProvider
    recorder: Recorder
    max_steps: int = 10
    max_total_tokens: int = 20_000
    max_tool_calls: int = 8
    timeout_seconds: int = 60
    tools: list[Tool] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def output_schema(self) -> type[BaseModel]: ...
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/agents/__init__.py multi_agent/agents/base.py tests/unit/test_agent_base.py
git commit -m "phase1(agents): BaseAgent skeleton with Pydantic config + abstract hooks"
```

---

## Task 20: BaseAgent.run() — Async ReAct Loop + Fan-out Tool Dispatch

**Files:**
- Modify: `multi_agent/agents/base.py`
- Modify: `tests/unit/test_agent_base.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_agent_base.py`:

```python
from pydantic import BaseModel as _BM
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.messages import ToolCallRequest, ToolResult
from multi_agent.tools.base import Tool


class _EchoArgs(_BM):
    msg: str


class _EchoTool(Tool):
    name: str = "echo"
    description: str = "echo a message"
    args_schema: type = _EchoArgs

    async def call(self, args, recorder):
        return ToolResult(tool_use_id="x", payload={"echo": args.msg})


@pytest.mark.asyncio
async def test_agent_runs_to_final_answer(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "done"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider, recorder=rec)
    out = await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert isinstance(out.payload, _EchoOutput)
    assert out.payload.answer == "done"
    assert out.steps_used == 1


@pytest.mark.asyncio
async def test_agent_dispatches_tool_then_answers(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "x"})],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "after tool"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider,
                        recorder=rec, tools=[_EchoTool()])
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert out.payload.answer == "after tool"
    assert out.steps_used == 2


@pytest.mark.asyncio
async def test_fan_out_parallel_tools(tmp_run_dir):
    """Two tool calls in one LLM response are dispatched concurrently."""
    import json as _j
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[
                ToolCallRequest(tool_use_id="t1", tool_name="echo", args={"msg": "a"}),
                ToolCallRequest(tool_use_id="t2", tool_name="echo", args={"msg": "b"}),
            ],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider,
                        recorder=rec, tools=[_EchoTool()])
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    # Two ToolCalled events should share the same parent span
    lines = [_j.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    tool_calls = [l for l in lines if l["event_type"] == "ToolCalled"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["parent_id"] == tool_calls[1]["parent_id"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: AttributeError on `agent.run`.

- [ ] **Step 3: Add `run()` and `_react_loop()` to `BaseAgent`**

Append inside the `BaseAgent` class:

```python
    async def run(self, input: AgentInput) -> AgentOutput:
        """Template method. Subclasses do not override."""
        with self.recorder.span(
            "agent_invoke", agent_name=self.name, role=self.role,
        ) as span:
            span.set_input(input.payload)
            output = await self._react_loop(input)
            span.set_output({"steps_used": output.steps_used})
            return output

    async def _react_loop(self, input: AgentInput) -> AgentOutput:
        from multi_agent.schemas.messages import AgentMessage
        from multi_agent.providers.json_robust import parse_json_robust
        import asyncio

        tools_by_name = {t.name: t for t in self.tools}
        messages: list[AgentMessage] = [
            AgentMessage(role="system", content=self.system_prompt()),
            AgentMessage(role="user", content=str(input.payload.get("query", input.payload))),
        ]

        tool_specs = [t.to_spec() for t in self.tools] if self.tools else None

        for step in range(1, self.max_steps + 1):
            response = await self.provider.complete(
                messages=messages,
                model=getattr(self.provider, "default_model", "stub-1"),
                tools=tool_specs,
                response_format=self.output_schema(),
                recorder=self.recorder,
                agent_name=self.name,
            )

            if response.tool_calls:
                # Fan-out: dispatch all tool calls concurrently
                results = await asyncio.gather(*[
                    self._dispatch_tool(tc, tools_by_name) for tc in response.tool_calls
                ], return_exceptions=True)
                for tc, result in zip(response.tool_calls, results):
                    if isinstance(result, Exception):
                        result = self._wrap_tool_exception(tc, result)
                    messages.append(self._tool_result_message(tc, result))
                continue

            # No tool calls → expect final answer
            schema = self.output_schema()
            parsed_dict = parse_json_robust(response.text)
            parsed = schema.model_validate(parsed_dict)
            return AgentOutput(payload=parsed, steps_used=step)

        from multi_agent.errors import BudgetExceeded
        raise BudgetExceeded(self.name, "max_steps", self.max_steps)

    async def _dispatch_tool(self, tc, tools_by_name):
        from multi_agent.errors import ToolCallParseError
        from multi_agent.schemas.messages import ToolResult
        tool = tools_by_name.get(tc.tool_name)
        if tool is None:
            return ToolResult(tool_use_id=tc.tool_use_id, payload=None,
                              error=f"unknown tool: {tc.tool_name}")
        try:
            args = tool.args_schema.model_validate(tc.args)
        except Exception as e:
            return ToolResult(tool_use_id=tc.tool_use_id, payload=None,
                              error=f"args validation failed: {e}")
        with self.recorder.span(
            "tool_call", tool_name=tc.tool_name, args=tc.args, agent_name=self.name,
        ) as span:
            try:
                result = await tool.call(args, self.recorder)
                span.set_output(result.payload or {"error": result.error})
                # Force the tool_use_id from the LLM (Tool.call may have set its own)
                return result.model_copy(update={"tool_use_id": tc.tool_use_id})
            except Exception as e:
                return ToolResult(tool_use_id=tc.tool_use_id, payload=None, error=str(e))

    def _wrap_tool_exception(self, tc, exc):
        from multi_agent.schemas.messages import ToolResult
        return ToolResult(tool_use_id=tc.tool_use_id, payload=None, error=str(exc))

    def _tool_result_message(self, tc, result):
        from multi_agent.schemas.messages import AgentMessage
        import json as _j
        payload = result.payload if result.error is None else {"error": result.error}
        return AgentMessage(
            role="tool", content=_j.dumps(payload, ensure_ascii=False),
            tool_use_id=tc.tool_use_id,
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: 4 passed (1 from Task 19 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add multi_agent/agents/base.py tests/unit/test_agent_base.py
git commit -m "phase1(agents): async ReAct loop + asyncio.gather fan-out tool dispatch"
```

---

## Task 21: BaseAgent — Budget Enforcement

**Files:**
- Modify: `multi_agent/agents/base.py`
- Modify: `tests/unit/test_agent_base.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_agent_base.py`:

```python
from multi_agent.errors import BudgetExceeded


@pytest.mark.asyncio
async def test_exceeding_max_steps_raises_budget(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    # All responses request more tool calls — agent never finalizes
    provider = StubProvider(responses=[
        ScriptedResponse(
            tool_calls=[ToolCallRequest(tool_use_id=f"t{i}", tool_name="echo", args={"msg": "x"})],
            finish_reason="tool_use",
        )
        for i in range(5)
    ])
    agent = _DummyAgent(name="dummy", role="t",
                        provider=provider, recorder=rec,
                        tools=[_EchoTool()], max_steps=3)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert exc.value.budget == "max_steps"
    assert exc.value.limit == 3


@pytest.mark.asyncio
async def test_exceeding_max_tool_calls_raises_budget(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    # Single LLM response with 5 tool calls; agent allows only 3
    provider = StubProvider(responses=[
        ScriptedResponse(
            tool_calls=[
                ToolCallRequest(tool_use_id=f"t{i}", tool_name="echo", args={"msg": str(i)})
                for i in range(5)
            ],
            finish_reason="tool_use",
        ),
    ])
    agent = _DummyAgent(name="dummy", role="t",
                        provider=provider, recorder=rec,
                        tools=[_EchoTool()], max_tool_calls=3)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert exc.value.budget == "max_tool_calls"
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_agent_base.py::test_exceeding_max_tool_calls_raises_budget -v
```

Expected: FAIL (`max_tool_calls` not yet enforced).

- [ ] **Step 3: Add tool-call budget check in `_react_loop`**

In `_react_loop`, immediately after `if response.tool_calls:`, before `await asyncio.gather`, add:

```python
                if len(response.tool_calls) > self.max_tool_calls:
                    from multi_agent.errors import BudgetExceeded
                    raise BudgetExceeded(self.name, "max_tool_calls", self.max_tool_calls)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/agents/base.py tests/unit/test_agent_base.py
git commit -m "phase1(agents): enforce max_steps and max_tool_calls budgets"
```

---

## Task 22: BaseAgent.run_stream() — Async Generator

**Files:**
- Modify: `multi_agent/agents/base.py`
- Modify: `tests/unit/test_agent_base.py`

- [ ] **Step 1: Write failing test**

Append to `tests/unit/test_agent_base.py`:

```python
from multi_agent.agents.base import StreamEvent


@pytest.mark.asyncio
async def test_run_stream_yields_tokens_and_final(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "done"}', finish_reason="end_turn"),
    ])
    agent = _DummyAgent(name="dummy", role="t", provider=provider, recorder=rec)

    collected: list[StreamEvent] = []
    async for ev in agent.run_stream(AgentInput(payload={"query": "hi"})):
        collected.append(ev)
    rec.close()

    kinds = [e.kind for e in collected]
    assert "agent_start" in kinds
    assert "agent_end" in kinds
    assert "final_answer" in kinds
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_agent_base.py::test_run_stream_yields_tokens_and_final -v
```

Expected: ImportError on `StreamEvent`.

- [ ] **Step 3: Add `StreamEvent` and `run_stream()` to `multi_agent/agents/base.py`**

Add near the top of the file (after imports):

```python
from typing import AsyncGenerator, Literal


class StreamEvent(BaseModel):
    kind: Literal["agent_start", "agent_end", "llm_token", "tool_start", "tool_end", "final_answer", "error"]
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Add method to `BaseAgent` class:

```python
    async def run_stream(self, input: AgentInput) -> AsyncGenerator[StreamEvent, None]:
        """Stream version of run(). Yields high-level events for CLI/Web progress display.

        Phase 1: minimal — only agent_start / agent_end / final_answer / error.
        Token-level streaming requires provider.complete_stream() integration (Phase 2).
        """
        yield StreamEvent(kind="agent_start", content=self.name)
        try:
            output = await self.run(input)
            yield StreamEvent(
                kind="final_answer",
                content=output.payload.model_dump_json(),
                metadata={"steps_used": output.steps_used},
            )
        except Exception as e:
            yield StreamEvent(kind="error", content=str(e),
                              metadata={"type": type(e).__name__})
            raise
        finally:
            yield StreamEvent(kind="agent_end", content=self.name)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_agent_base.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/agents/base.py tests/unit/test_agent_base.py
git commit -m "phase1(agents): run_stream() async generator emitting StreamEvent"
```

---

## Task 23: EchoStubAgent — Concrete Agent for E2E Testing

**Files:**
- Create: `multi_agent/agents/stub_echo.py`
- Create: `tests/unit/test_stub_echo_agent.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_stub_echo_agent.py
import pytest
from multi_agent.agents.stub_echo import EchoStubAgent, EchoStubOutput
from multi_agent.agents.base import AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_echo_stub_runs_end_to_end(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hello back"}'),
    ])
    agent = EchoStubAgent(name="echo", role="stub",
                          provider=provider, recorder=rec)
    out = await agent.run(AgentInput(payload={"query": "hello"}))
    rec.close()
    assert isinstance(out.payload, EchoStubOutput)
    assert out.payload.echoed == "hello back"


def test_echo_stub_system_prompt_mentions_role(tmp_path):
    from multi_agent.providers.stub import StubProvider
    p = StubProvider(responses=[])
    rec = Recorder(run_id="r-prompt-test", run_dir=tmp_path / "runs" / "x")
    a = EchoStubAgent(name="echo", role="stub", provider=p, recorder=rec)
    assert "echo" in a.system_prompt().lower()
    rec.close()
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_stub_echo_agent.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/agents/stub_echo.py`**

```python
from __future__ import annotations
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent


class EchoStubOutput(BaseModel):
    echoed: str


class EchoStubAgent(BaseAgent):
    """Minimal concrete agent for E2E walking-skeleton test.

    Expects provider to return a JSON object with key 'echoed'.
    No tools. No multi-step reasoning. Just shape-checks the full pipeline.
    """

    def system_prompt(self) -> str:
        return (
            "You are an echo agent. Echo the user's message back inside "
            'a JSON object: {"echoed": "<message>"}. Do not add anything else.'
        )

    def output_schema(self):
        return EchoStubOutput
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_stub_echo_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/agents/stub_echo.py tests/unit/test_stub_echo_agent.py
git commit -m "phase1(agents): EchoStubAgent concrete agent for E2E test"
```

---

## Task 24: run_query — Top-Level Orchestration with RunFinished Invariant

**Files:**
- Create: `multi_agent/runner.py`
- Create: `tests/unit/test_runner.py`

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_runner.py
import json
import pytest
from multi_agent.runner import run_query
from multi_agent.agents.stub_echo import EchoStubAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_run_query_writes_meta_and_run_finished(tmp_path):
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hi back"}'),
    ])
    result = await run_query(
        query="hi",
        agent_factory=lambda provider, recorder: EchoStubAgent(
            name="echo", role="stub", provider=provider, recorder=recorder,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub"},
    )
    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "events.db").exists()

    lines = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [l["event_type"] for l in lines]
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert lines[-1]["status"] == "ok"


@pytest.mark.asyncio
async def test_run_query_emits_run_finished_on_exception(tmp_path):
    """Critical invariant: even when agent raises, events.jsonl ends with RunFinished(status='error')."""
    runs_root = tmp_path / "runs"

    class _BoomAgent(EchoStubAgent):
        async def run(self, input):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_query(
            query="hi",
            agent_factory=lambda provider, recorder: _BoomAgent(
                name="boom", role="t", provider=provider, recorder=recorder,
            ),
            provider=StubProvider(responses=[]),
            runs_root=runs_root,
            config={"profile": "stub"},
        )

    # Find the produced run dir
    run_dirs = list(runs_root.glob("r_*"))
    assert len(run_dirs) == 1
    lines = [json.loads(l) for l in (run_dirs[0] / "events.jsonl").read_text().splitlines()]
    assert lines[-1]["event_type"] == "RunFinished"
    assert lines[-1]["status"] == "error"
    assert "boom" in lines[-1]["error"]
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/unit/test_runner.py -v
```

Expected: ImportError on `multi_agent.runner`.

- [ ] **Step 3: Create `multi_agent/runner.py`**

```python
from __future__ import annotations
from pathlib import Path
from typing import Callable, Any
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id
from multi_agent.schemas.events import RunStarted, RunFinished
from multi_agent.providers.base import LLMProvider
from multi_agent.agents.base import BaseAgent, AgentInput


async def run_query(
    *,
    query: str,
    agent_factory: Callable[[LLMProvider, Recorder], BaseAgent],
    provider: LLMProvider,
    runs_root: Path,
    config: dict[str, Any] | None = None,
) -> dict:
    """Top-level entry. Guarantees a RunFinished event regardless of outcome.

    Returns a small dict {run_id, status, final_answer?}.
    """
    run_id = fresh_run_id()
    run_dir = Path(runs_root) / run_id
    recorder = Recorder(run_id=run_id, run_dir=run_dir)
    recorder.set_meta(query=query, config=(config or {}))

    final_answer: str | None = None
    status = "ok"
    error: str | None = None

    try:
        recorder.emit(RunStarted(
            event_id=recorder.fresh_event_id(), run_id=run_id,
            timestamp=recorder.now(), parent_id=None,
            query=query, config=(config or {}),
        ))
        agent = agent_factory(provider, recorder)
        output = await agent.run(AgentInput(payload={"query": query}))
        final_answer = output.payload.model_dump_json()
    except Exception as e:
        status = "error"
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        try:
            recorder.emit(RunFinished(
                event_id=recorder.fresh_event_id(), run_id=run_id,
                timestamp=recorder.now(), parent_id=None,
                status=status, final_answer=final_answer, error=error,
            ))
        finally:
            recorder.close()

    return {"run_id": run_id, "status": status, "final_answer": final_answer}
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/unit/test_runner.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add multi_agent/runner.py tests/unit/test_runner.py
git commit -m "phase1(runner): run_query top-level entry with RunFinished invariant"
```

---

## Task 25: Integration Test — Walking Skeleton End-to-End

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_walking_skeleton.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_walking_skeleton.py
"""Phase 1 acceptance test: full walking skeleton end-to-end.

A single query → stub agent → stub provider → trace files produced.
All invariants from spec §2.6 and §8.3 must hold.
"""
import json
import sqlite3
import pytest

from multi_agent.runner import run_query
from multi_agent.agents.stub_echo import EchoStubAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_walking_skeleton_full_invariants(tmp_path):
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hello world"}',
                         finish_reason="end_turn",
                         usage_input_tokens=10, usage_output_tokens=5),
    ])

    result = await run_query(
        query="hello world",
        agent_factory=lambda p, r: EchoStubAgent(name="echo", role="stub", provider=p, recorder=r),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub", "phase": "1"},
    )

    run_dir = runs_root / result["run_id"]

    # --- Invariant 1: meta.json exists with required fields ---
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["run_id"] == result["run_id"]
    assert meta["query"] == "hello world"
    assert "started_at" in meta and "finished_at" in meta

    # --- Invariant 2: events.jsonl ends with RunFinished ---
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[0]["event_type"] == "RunStarted"
    assert events[-1]["event_type"] == "RunFinished"
    assert events[-1]["status"] == "ok"

    # --- Invariant 3: every LLMRequested has a matching LLMResponded ---
    requested = [e for e in events if e["event_type"] == "LLMRequested"]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(requested) == len(responded) == 1

    # --- Invariant 4: AgentInvoked / AgentResponded paired ---
    invoked = [e for e in events if e["event_type"] == "AgentInvoked"]
    agent_resp = [e for e in events if e["event_type"] == "AgentResponded"]
    assert len(invoked) == len(agent_resp) == 1

    # --- Invariant 5: parent_id chain — LLMRequested.parent_id == AgentInvoked.event_id ---
    assert requested[0]["parent_id"] == invoked[0]["event_id"]

    # --- Invariant 6: SQLite mirror is consistent with JSONL ---
    conn = sqlite3.connect(run_dir / "events.db")
    cur = conn.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == len(events)
    cur = conn.execute("SELECT event_type FROM events ORDER BY timestamp")
    db_types = [r[0] for r in cur.fetchall()]
    json_types = [e["event_type"] for e in events]
    assert sorted(db_types) == sorted(json_types)
    conn.close()

    # --- Invariant 7: final answer is structured ---
    final = json.loads(result["final_answer"])
    assert final["echoed"] == "hello world"


@pytest.mark.asyncio
async def test_walking_skeleton_error_path(tmp_path):
    """When provider responds invalid JSON, run still terminates with RunFinished(error)."""
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text="not valid json at all", finish_reason="end_turn"),
    ])

    with pytest.raises(Exception):
        await run_query(
            query="x",
            agent_factory=lambda p, r: EchoStubAgent(name="echo", role="stub", provider=p, recorder=r),
            provider=provider,
            runs_root=runs_root,
            config={},
        )

    run_dirs = list(runs_root.glob("r_*"))
    assert len(run_dirs) == 1
    events = [json.loads(l) for l in (run_dirs[0] / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "RunFinished"
    assert events[-1]["status"] == "error"
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/integration/test_walking_skeleton.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Run the entire test suite as final acceptance**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest -v
```

Expected: all tests pass (~45-55 tests total, exact count depends on parametrization).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/
git commit -m "phase1(integration): walking skeleton E2E with full invariant checks"
```

- [ ] **Step 5: Tag the Phase 1 completion**

```bash
git tag -a phase1-walking-skeleton -m "Phase 1 complete: trace + schema + stub agent + asyncio"
```

---

## Acceptance Criteria

Phase 1 is complete when:

1. `pytest -v` from `experiments/multi_agent/` runs all tests green
2. Running a query via `run_query()` produces a `runs/<run_id>/` directory containing:
   - `meta.json` with `run_id / query / config / started_at / finished_at`
   - `events.jsonl` starting with `RunStarted` and ending with `RunFinished`
   - `events.db` with the same events queryable by `run_id / event_type / agent_name`
3. The integration test `test_walking_skeleton_full_invariants` enforces every spec §2.6 invariant
4. The error-path test `test_walking_skeleton_error_path` confirms `RunFinished(status="error")` is emitted on uncaught exceptions

## What This Plan Does NOT Cover (Out-of-Scope for Phase 1)

These belong to later phases and have their own plans:

- **Phase 2**: Qdrant indexing + 3 collections, bge-m3 dense encoder, jieba sparse encoder, real Lawyer agent with retrieval, run_stream provider integration
- **Phase 3**: Receptionist, MarkdownMemoryStore, EntityState extraction, WorkingMemory wired into agents, Cross-Turn compression, Multi-issue sub_cases
- **Phase 4**: Secretary as separate agent + agent-as-tool, contract_review / doc_generation / doc_interpret business tools
- **Phase 5**: Supervisor agent, ExperimentRunner + RunGroup, Judges (Claude Opus), Comparator, AblationRunner, LatencyProfiler, Streamlit viewer

## Notes for Implementing Engineer

- **Read spec sections referenced in the goal first** (`docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` §0–§3, §8, §9)
- **All file paths in this plan are relative to `experiments/multi_agent/`** unless explicitly absolute
- **Tests use `pytest-asyncio` auto mode** (set in `pyproject.toml`); `@pytest.mark.asyncio` is required on each async test for explicitness
- **Commit after every task**, never batch — the spec's "frequent commits" principle is non-negotiable
- **If a step fails unexpectedly**, do not skip — investigate. Phase 1 is the foundation; cutting corners here costs 5x in Phase 2+
- **The walking skeleton is the contract**: every later phase will add code, but never break the invariants tested in `test_walking_skeleton.py`
