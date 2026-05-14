# Phase 3b — Memory Integration + WorkingMemory Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Wire `MarkdownMemoryStore` (built in Phase 3) into the actual `run_query` flow so each turn persists to `sessions/<id>/turns/NNN.md` and Receptionist reads sticky context from previous turns. Add `WorkingMemory` threading into agents so retrieval results accumulate across ReAct turns. Multi-turn E2E proves end-to-end persistence.

**Architecture:** `run_query` gains `session_id` + `memory_store` params. After each successful run, `Turn` is appended; if `session_id` provided, `StickyContext` is updated from agent outputs. Receptionist's `_render_input` reads sticky.md (if exists) and injects condensed entity_state + recent turns into the user message. `WorkingMemory` becomes a `BaseAgent` field; `_react_loop` writes retrieved Evidence to it.

**Phase 3 starting point:** Tag `phase3-receptionist-memory`. 166 tests pass + 1 skipped.

---

## Out of scope (later)

- Cross-turn compression (>5 turns → `history_summary`) — Phase 3c
- `ma_user_history` Qdrant collection — Phase 3c (after enough real turns exist)
- Sequential fan-out for sub_cases — Phase 4 with Secretary
- `agent_notes` write by Supervisor — Phase 5

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   ├── runner.py                            # MODIFY: add session_id + memory_store
│   ├── agents/
│   │   ├── base.py                          # MODIFY: WorkingMemory threading
│   │   └── receptionist.py                  # MODIFY: read sticky context
└── tests/
    ├── unit/
    │   ├── test_runner_with_memory.py       # NEW
    │   └── test_working_memory_threading.py # NEW
    └── integration/
        └── test_multi_turn_session_e2e.py   # NEW
```

All tests in `conda run -n qwen35`.

---

## Task 1: Wire `run_query` to `MarkdownMemoryStore`

**Files:**
- Modify: `multi_agent/runner.py`
- Create: `tests/unit/test_runner_with_memory.py`

`run_query` gets two optional params:
- `session_id: str | None` — if set, read existing sticky and append a Turn after success
- `memory_store: MarkdownMemoryStore | None` — if None, skip memory ops

Backward-compatible: existing callers without these params keep working unchanged.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_runner_with_memory.py
import json
import pytest
from datetime import datetime
from pydantic import BaseModel

from multi_agent.runner import run_query
from multi_agent.agents.base import BaseAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import StickyContext


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_run_query_appends_turn_when_session_id_given(tmp_path):
    runs_root = tmp_path / "runs"
    store = MarkdownMemoryStore(root=tmp_path / "memory_store")
    # Seed an empty sticky so the session "exists"
    store.write_sticky(StickyContext(session_id="s_test"))

    provider = StubProvider(responses=[
        ScriptedResponse(text='{"answer": "hi"}'),
    ])
    result = await run_query(
        query="hello?",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider,
        runs_root=runs_root,
        config={},
        session_id="s_test",
        memory_store=store,
    )
    assert result["status"] == "ok"
    # Sticky should now have linked_runs updated
    sticky = store.read_sticky("s_test")
    assert sticky is not None
    assert result["run_id"] in sticky.linked_runs
    # And a turn file should exist
    turns = store.recent_turns("s_test", n=5)
    assert len(turns) == 1
    assert turns[0].question == "hello?"
    assert turns[0].run_id == result["run_id"]


@pytest.mark.asyncio
async def test_run_query_works_without_session_id(tmp_path):
    """Backward-compat: existing callers don't pass session_id; should still work."""
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "x"}')])
    result = await run_query(
        query="hi",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider, runs_root=runs_root, config={},
    )
    assert result["status"] == "ok"


@pytest.mark.asyncio
async def test_run_query_creates_session_if_missing(tmp_path):
    """If session_id given but no sticky exists, create one."""
    runs_root = tmp_path / "runs"
    store = MarkdownMemoryStore(root=tmp_path / "memory_store")
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "y"}')])
    result = await run_query(
        query="new session",
        agent_factory=lambda p, r: _Agent(
            name="dummy", role="t", provider=p, recorder=r, model="stub-1",
        ),
        provider=provider, runs_root=runs_root, config={},
        session_id="s_new",
        memory_store=store,
    )
    assert result["status"] == "ok"
    sticky = store.read_sticky("s_new")
    assert sticky is not None
    assert sticky.session_id == "s_new"
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_runner_with_memory.py -v"
```

Expected: FAIL — `run_query` doesn't accept `session_id`/`memory_store` yet.

- [ ] **Step 3: Modify `multi_agent/runner.py`**

Read the current `runner.py`. Add `session_id` and `memory_store` to the `run_query` signature with defaults `None`. After the agent run succeeds, if both are provided:

1. Read sticky; if missing, create a new `StickyContext(session_id=...)`
2. Append a `Turn` derived from the run
3. Update sticky `linked_runs` and write back

Sample code structure (adapt to existing code):

```python
from multi_agent.schemas.memory import StickyContext, Turn

async def run_query(
    *,
    query: str,
    agent_factory,
    provider,
    runs_root,
    config=None,
    session_id: str | None = None,            # NEW
    memory_store=None,                         # NEW (MarkdownMemoryStore | None)
):
    # ... existing run_id setup, Recorder, agent_factory invocation ...

    # After successful run (existing logic continues):
    # output = await agent.run(AgentInput(...))
    # result = ...

    # NEW: memory integration
    if session_id and memory_store is not None and result["status"] == "ok":
        from datetime import datetime
        sticky = memory_store.read_sticky(session_id) or StickyContext(session_id=session_id)
        if run_id not in sticky.linked_runs:
            sticky.linked_runs.append(run_id)
        # Compute turn number: count existing turns + 1
        existing_turns = memory_store.recent_turns(session_id, n=999)
        next_turn_no = max([t.turn for t in existing_turns], default=0) + 1
        memory_store.append_turn(session_id, Turn(
            turn=next_turn_no,
            run_id=run_id,
            started_at=started_at,            # captured earlier
            finished_at=datetime.now(),
            question=query,
            final_answer=result.get("final_answer", "") or "",
            agents_invoked=[agent.name] if agent is not None else [],
        ))
        memory_store.write_sticky(sticky)

    return result
```

Look at the existing `run_query` to find where `started_at` is recorded (likely the `recorder.set_meta(started_at=...)` call or similar). Capture it as a local variable so it's available at memory-write time.

If the existing code doesn't capture `started_at` separately, add `started_at = recorder.now()` at the top of run_query before the try block.

- [ ] **Step 4: Verify pass + no regression**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_runner_with_memory.py -v && pytest -v 2>&1 | tail -5"
```

Expected: 3 new tests pass; full suite 169 passed + 1 skipped (166 + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/runner.py experiments/multi_agent/tests/unit/test_runner_with_memory.py
git commit -m "phase3b(runner): session_id + memory_store params; persist Turn after run"
```

---

## Task 2: Receptionist Reads Sticky Context

**Files:**
- Modify: `multi_agent/agents/receptionist.py` — override `_render_input` to inject sticky context
- Modify: `multi_agent/agents/base.py` — pass `sticky_context` through `AgentInput.payload`
- Create: `tests/unit/test_receptionist_with_sticky.py`

When Receptionist's input payload includes `sticky_context` (a dict from `StickyContext.model_dump()`), it renders a condensed version into the prompt so the LLM sees: previous topic, key facts, recent turn questions.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_receptionist_with_sticky.py
import pytest
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.providers.stub import StubProvider
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_receptionist_render_with_sticky_context(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="r", role="t", provider=p, recorder=rec)

    input = AgentInput(payload={
        "query": "那依据哪条法律?",
        "sticky_context": {
            "session_id": "s_x",
            "legal_domain": "民事",
            "case_type": "租赁纠纷",
            "last_law_name": "民法典",
            "mentioned_laws": ["民法典"],
            "entity_state": {
                "active_subjects": [{"role": "原告", "identifier": "用户", "attributes": []}],
                "key_facts": [{"fact": "租期1年", "confidence": "high", "source_turn": 1}],
            },
        },
    })
    rendered = agent._render_input(input)
    assert "那依据哪条法律?" in rendered
    # The condensed context should appear
    assert "上一轮主题" in rendered or "previous" in rendered.lower()
    assert "租赁纠纷" in rendered
    assert "民法典" in rendered
    assert "租期1年" in rendered
    rec.close()


def test_receptionist_render_without_sticky(tmp_path):
    """No sticky → behaves like before, just query."""
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="r", role="t", provider=p, recorder=rec)
    input = AgentInput(payload={"query": "房东涨租"})
    rendered = agent._render_input(input)
    assert rendered == "房东涨租"
    rec.close()
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_receptionist_with_sticky.py -v"
```

Expected: FAIL — receptionist doesn't render sticky context yet.

- [ ] **Step 3: Override `_render_input` in `ReceptionistAgent`**

Modify `multi_agent/agents/receptionist.py`:

```python
"""ReceptionistAgent — triage + multi-issue decomposition (spec §3.5)."""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.receptionist import ReceptionistOutput


class ReceptionistAgent(BaseAgent):
    """Tool-less classifier. Reads user query (+ optional sticky_context)."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.receptionist").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[ReceptionistOutput]:
        return ReceptionistOutput

    def _render_input(self, input) -> str:
        payload = input.payload
        query = str(payload.get("query", ""))
        sticky = payload.get("sticky_context")
        if not sticky:
            return query

        # Render condensed previous-turn context
        lines = ["# 上一轮主题(供参考,不要直接复述给用户)"]
        if sticky.get("case_type"):
            lines.append(f"- 案件类型: {sticky['case_type']}")
        if sticky.get("legal_domain"):
            lines.append(f"- 法律领域: {sticky['legal_domain']}")
        if sticky.get("last_law_name"):
            lines.append(f"- 上轮主要法律: {sticky['last_law_name']}")
        mentioned = sticky.get("mentioned_laws") or []
        if mentioned:
            lines.append(f"- 提到过的法律: {', '.join(mentioned)}")
        es = sticky.get("entity_state") or {}
        facts = [f.get("fact", "") if isinstance(f, dict) else str(f) for f in (es.get("key_facts") or [])]
        if facts:
            lines.append(f"- 已知事实: {'; '.join(facts)}")
        lines.append("")
        lines.append(f"# 用户本轮提问\n{query}")
        return "\n".join(lines)
```

- [ ] **Step 4: Verify pass + full suite → 171 passed + 1 skipped (169 + 2 new).**

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/receptionist.py experiments/multi_agent/tests/unit/test_receptionist_with_sticky.py
git commit -m "phase3b(receptionist): _render_input reads sticky_context for follow-up turns"
```

---

## Task 3: WorkingMemory Threading

**Files:**
- Modify: `multi_agent/agents/base.py` — initialize `working_memory` in `_react_loop`, accumulate tool results
- Create: `tests/unit/test_working_memory_threading.py`

`BaseAgent.working_memory` field already exists (Phase 1 schema). Phase 3b actually wires it. After each tool call returns Evidence(s), append to `working_memory.retrieved_evidence`. After the run, the working memory state is accessible from the agent.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_working_memory_threading.py
import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.messages import ToolCallRequest, ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class _Args(BaseModel):
    q: str


class _Tool(Tool):
    name: str = "fake_search"
    description: str = "returns one evidence"
    args_schema: type[BaseModel] = _Args

    async def call(self, args, recorder):
        ev = Evidence(
            doc_id="民法典-510",
            law_name="中华人民共和国民法典",
            law_short="民法典",
            article_no="510",
            text="当事人就合同补充内容...",
            score=0.9,
            retriever="hybrid",
        )
        return ToolResult(tool_use_id="x", payload={"evidences": [ev.model_dump()]})


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_working_memory_accumulates_evidence(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="fake_search", args={"q": "x"})],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   tools=[_Tool()], model="stub-1")
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    # working_memory should be present and contain the retrieved evidence
    assert agent.working_memory is not None
    assert len(agent.working_memory.retrieved_evidence) == 1
    assert agent.working_memory.retrieved_evidence[0].doc_id == "民法典-510"


@pytest.mark.asyncio
async def test_working_memory_empty_when_no_tool_calls(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "ok"}')])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec, model="stub-1")
    await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert agent.working_memory is not None
    assert agent.working_memory.retrieved_evidence == []
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_working_memory_threading.py -v"
```

Expected: FAIL — working_memory isn't being populated.

- [ ] **Step 3: Modify `BaseAgent._react_loop`**

In `multi_agent/agents/base.py`:

a) At the top of `_react_loop`, initialize working_memory if None:

```python
        from multi_agent.schemas.working_memory import WorkingMemory
        if self.working_memory is None:
            self.working_memory = WorkingMemory()
```

b) After each tool dispatch (where `results = await asyncio.gather(...)`), inspect each ToolResult's payload for evidences and accumulate:

```python
                for tc, result in zip(response.tool_calls, results):
                    if isinstance(result, Exception):
                        result = self._wrap_tool_exception(tc, result)
                    # NEW: harvest evidences into working_memory
                    if result.payload and "evidences" in result.payload:
                        from multi_agent.schemas.evidence import Evidence
                        for ev_dict in result.payload["evidences"]:
                            try:
                                self.working_memory.add_evidence(Evidence.model_validate(ev_dict))
                            except Exception:
                                pass  # tolerate malformed payloads
                    elif result.payload and "evidence" in result.payload:
                        # exact_read returns single evidence
                        from multi_agent.schemas.evidence import Evidence
                        try:
                            self.working_memory.add_evidence(Evidence.model_validate(result.payload["evidence"]))
                        except Exception:
                            pass
                    messages.append(self._tool_result_message(tc, result))
```

Note: pydantic might require `BaseAgent.working_memory` field to allow mutation. Since `working_memory: WorkingMemory | None = None` and `WorkingMemory` is a pydantic model, mutation of `working_memory.retrieved_evidence.append(...)` should work as the field accepts None or a WorkingMemory instance. The `add_evidence` method on WorkingMemory just appends to a list field.

- [ ] **Step 4: Verify pass + full suite → 173 passed + 1 skipped (171 + 2 new).**

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/base.py experiments/multi_agent/tests/unit/test_working_memory_threading.py
git commit -m "phase3b(agents): WorkingMemory accumulates retrieved Evidence during ReAct loop"
```

---

## Task 4: Multi-Turn Session E2E

**Files:**
- Create: `tests/integration/test_multi_turn_session_e2e.py`

Two-turn real session: first user asks "房东涨租合法吗",second user asks "那依据哪条法律". Receptionist on second turn should see sticky context from first turn.

- [ ] **Step 1: Write test**

```python
# tests/integration/test_multi_turn_session_e2e.py
"""Phase 3b E2E: 2-turn session with memory persistence + Receptionist follow-up."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.schemas.memory import StickyContext, EntityState, KeyFact
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.base import AgentInput
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.runner import run_query
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_multi_turn_{uuid.uuid4().hex[:8]}"
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
async def test_two_turn_session(statute_index, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    store_root = tmp_path / "memory_store"
    store = MarkdownMemoryStore(root=store_root)
    provider = OpenAICompatibleProvider()
    session_id = "s_phase3b_test"

    # --- Turn 1: 房东涨租合法吗? ---
    # Seed sticky so the session exists
    store.write_sticky(StickyContext(
        session_id=session_id,
        legal_domain="民事",
        case_type="租赁纠纷",
        last_law_name="民法典",
        mentioned_laws=["民法典"],
        entity_state=EntityState(
            key_facts=[KeyFact(fact="租期一年", confidence="high", source_turn=0)],
        ),
    ))

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    result_t1 = await run_query(
        query="我租的房合同一年,房东要涨 30% 房租,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b", specialty="民事",
            max_steps=8, max_tool_calls=10,
        ),
        provider=provider, runs_root=runs_root, config={},
        session_id=session_id, memory_store=store,
    )
    assert result_t1["status"] == "ok"

    # After turn 1: sticky should have run_id linked
    sticky_after_t1 = store.read_sticky(session_id)
    assert result_t1["run_id"] in sticky_after_t1.linked_runs
    turns = store.recent_turns(session_id, n=5)
    assert len(turns) == 1
    assert turns[0].question.startswith("我租的房合同")

    # --- Turn 2: Receptionist reads sticky context ---
    rec_run_id = fresh_run_id()
    rec_recorder = Recorder(run_id=rec_run_id, run_dir=runs_root / rec_run_id)
    receptionist = ReceptionistAgent(
        name="receptionist", role="triage",
        provider=provider, recorder=rec_recorder,
        model="qwen3.5-9b", max_steps=2,
    )
    triage_out = await receptionist.run(AgentInput(payload={
        "query": "那依据哪条法律?",
        "sticky_context": sticky_after_t1.model_dump(),
    }))
    rec_recorder.close()
    assert isinstance(triage_out.payload, ReceptionistOutput)
    # The receptionist should understand this is a follow-up about civil law
    assert triage_out.payload.primary_specialty in ("民事", "房产", "通用")


@pytest.mark.asyncio
async def test_sticky_persists_across_runs(tmp_path):
    """Run-level smoke: write sticky → close store → open new store at same path → sticky still readable."""
    root = tmp_path / "memory_store"
    store1 = MarkdownMemoryStore(root=root)
    store1.write_sticky(StickyContext(
        session_id="s_persist",
        legal_domain="民事",
        case_type="x",
        entity_state=EntityState(
            key_facts=[KeyFact(fact="persistence test", confidence="high")],
        ),
    ))
    # Simulate process restart
    store2 = MarkdownMemoryStore(root=root)
    loaded = store2.read_sticky("s_persist")
    assert loaded is not None
    assert loaded.legal_domain == "民事"
    assert loaded.entity_state.key_facts[0].fact == "persistence test"
```

- [ ] **Step 2: Run + Step 3: Full suite + Step 4: Commit + tag**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_multi_turn_session_e2e.py -v"
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -5"

cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_multi_turn_session_e2e.py
git commit -m "phase3b(integration): multi-turn session E2E with memory persistence"
git tag -a phase3b-memory-integration -m "Phase 3b complete: memory wired into runner + WorkingMemory + multi-turn E2E"
git tag -l "phase*"
```

Expected: 175 passed + 1 skipped (173 + 2 new — multi-turn integration test + sticky persistence test).

## Acceptance Criteria

Phase 3b complete when:

1. Full pytest passes (~175 tests)
2. `run_query` persists Turn after success when `session_id` + `memory_store` given
3. Sticky `linked_runs` updated; turn file appears in `sessions/<id>/turns/NNN-<slug>.md`
4. Receptionist `_render_input` injects sticky context into prompt
5. `WorkingMemory.retrieved_evidence` populates as tools return Evidence
6. Multi-turn E2E proves end-to-end persistence
7. Tag `phase3b-memory-integration` exists

## Out of Scope (carry to Phase 3c / Phase 4)

- Cross-turn compression (>5 turns)
- `ma_user_history` Qdrant collection
- Sequential fan-out per sub_case
- Supervisor writes agent_notes
