# Phase 5b — Eval Framework (MVP) Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Establish the minimum viable eval framework per spec §7 so we can run a fixed QuerySet through the multi-agent pipeline, derive automatic metrics from the trace, and produce a comparable RunGroup artifact. **No LLM judges** in this phase — only rule-based `CitationAccuracyJudge`. Claude-based `GroundednessJudge` / `HelpfulnessJudge` / `Comparator` / `AblationRunner` / `LatencyProfiler` / Streamlit viewer all deferred to Phase 5c+.

**Phase 5a starting point:** Tag `phase5a-supervisor`. 202 tests + 1 skipped.

---

## Out of scope (Phase 5c / later)

- LLM judges (GroundednessJudge, HelpfulnessJudge using Claude Opus)
- Comparator (group-vs-group diff)
- AblationRunner (DisableAgent / SwapModel / DisableTool)
- LatencyProfiler (SpanTiming derivation + flame graphs)
- Trace Viewer (Streamlit)
- Large-scale golden_qa_v1 extraction from `laws_data` (deferred until eval pipeline proven on synthetic set)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── queryset.py          # QuerySet + Query schemas + YAML loader
│   │   ├── metrics.py            # derive_metrics_from_trace(run_dir) → RunMetrics
│   │   ├── judges/
│   │   │   ├── __init__.py
│   │   │   └── citation_accuracy.py   # rule-based, no LLM
│   │   ├── runner.py             # ExperimentRunner + RunGroup
│   │   └── report.py             # render summary.md from RunGroup
│   └── ...
├── evals/
│   └── querysets/
│       └── synthetic_seed_v1.yaml    # 5-8 hand-crafted seed queries
└── tests/
    ├── unit/
    │   ├── test_queryset.py
    │   ├── test_metrics.py
    │   ├── test_citation_accuracy.py
    │   ├── test_experiment_runner.py
    │   └── test_report.py
    └── integration/
        └── test_eval_e2e.py      # real Qwen runs synthetic_seed_v1 + summary.md
```

---

## Task 1: QuerySet schema + YAML loader

**Files:**
- Create: `multi_agent/eval/__init__.py` (empty)
- Create: `multi_agent/eval/queryset.py`
- Create: `tests/unit/test_queryset.py`
- Create: `evals/querysets/synthetic_seed_v1.yaml`

Pydantic models matching spec §7.2 YAML shape. `QuerySet.from_yaml(path)` reads + validates.

- [ ] **Step 1: Seed YAML** — `evals/querysets/synthetic_seed_v1.yaml`

```yaml
meta:
  name: synthetic_seed_v1
  description: "Phase 5b 手写种子集 — 4 类常见民事咨询"
  created: 2026-05-14

queries:
  - id: q001
    text: "房东要在合同期内涨我 30% 房租,合法吗?"
    jurisdiction: CN
    cause: 房产纠纷
    source: hand_written
    tags: [民事, 租赁, 涨租]
    expected:
      should_cite_any: ["民法典-510", "民法典-563", "民法典-703", "民法典-707"]
      expected_answer_mode: evidence_grounded
      confidence: high

  - id: q002
    text: "邻居家漏水把我家天花板泡了,该怎么索赔?"
    jurisdiction: CN
    cause: 邻里纠纷
    source: hand_written
    tags: [民事, 侵权, 损害赔偿]
    expected:
      should_cite_any: ["民法典-1165", "民法典-1184"]
      expected_answer_mode: evidence_grounded
      confidence: medium

  - id: q003
    text: "网购商品到货后发现是假货,商家不退款怎么办?"
    jurisdiction: CN
    cause: 消费纠纷
    source: hand_written
    tags: [民事, 消费者]
    expected:
      should_cite_any: ["民法典-577", "民法典-584"]
      expected_answer_mode: evidence_grounded
      confidence: medium

  - id: q004
    text: "驾车追尾对方,对方车里有伤员,我的责任怎么认定?"
    jurisdiction: CN
    cause: 交通事故
    source: hand_written
    tags: [交通, 责任认定]
    expected:
      should_cite_any: ["道路交通安全法-76"]
      expected_answer_mode: evidence_grounded
      confidence: medium

  - id: q005
    text: "你好"
    jurisdiction: CN
    cause: 闲聊
    source: hand_written
    tags: [safety, smalltalk]
    expected:
      expected_answer_mode: clarification_or_refusal
      confidence: high
```

- [ ] **Step 2: Failing test**

```python
# tests/unit/test_queryset.py
from pathlib import Path
import pytest
from multi_agent.eval.queryset import QuerySet, Query


def test_queryset_loads_seed_yaml():
    path = Path(__file__).parents[2] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(path)
    assert qs.meta.name == "synthetic_seed_v1"
    assert len(qs.queries) >= 5
    assert qs.queries[0].id == "q001"
    assert qs.queries[0].text.startswith("房东")
    assert "民法典-510" in qs.queries[0].expected.should_cite_any


def test_query_has_required_fields():
    q = Query(id="qX", text="t", jurisdiction="CN", cause="c", source="s")
    assert q.tags == []
    assert q.expected.should_cite_any == []
```

- [ ] **Step 3: Verify failure** → ImportError.

- [ ] **Step 4: Implement `multi_agent/eval/queryset.py`**

```python
"""QuerySet schema + YAML loader (Phase 5b)."""
from __future__ import annotations
from datetime import date
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field


class ExpectedAnswer(BaseModel):
    should_cite_any: list[str] = Field(default_factory=list)
    expected_answer_mode: Literal[
        "evidence_grounded", "clarification_or_refusal", "advisory"
    ] | None = None
    confidence: Literal["low", "medium", "high"] | None = None


class Query(BaseModel):
    id: str
    text: str
    jurisdiction: str = "CN"
    cause: str
    source: str
    source_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    expected: ExpectedAnswer = Field(default_factory=ExpectedAnswer)


class QuerySetMeta(BaseModel):
    name: str
    description: str = ""
    created: date | None = None


class QuerySet(BaseModel):
    meta: QuerySetMeta
    queries: list[Query]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "QuerySet":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)
```

- [ ] **Step 5: Verify pass + full suite** → 204 passed + 1 skipped (202 + 2 new).

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/__init__.py \
        experiments/multi_agent/multi_agent/eval/queryset.py \
        experiments/multi_agent/evals/querysets/synthetic_seed_v1.yaml \
        experiments/multi_agent/tests/unit/test_queryset.py
git commit -m "phase5b(eval): QuerySet schema + YAML loader + synthetic_seed_v1"
```

---

## Task 2: Trace-derived metrics

**Files:**
- Create: `multi_agent/eval/metrics.py`
- Create: `tests/unit/test_metrics.py`

Reads a run's `events.jsonl`, aggregates per spec §7.6. **No** SpanTiming — that's the LatencyProfiler in Phase 5c. Just totals.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_metrics.py
import json
from pathlib import Path
import pytest
from multi_agent.eval.metrics import derive_run_metrics, RunMetrics


def test_derive_metrics_from_synthetic_events(tmp_path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    events_path = run_dir / "events.jsonl"
    evs = [
        {"event_id": "1", "kind": "RunStarted", "ts_ms": 1000, "run_id": "r1", "parent_id": None, "data": {}},
        {"event_id": "2", "kind": "AgentInvoked", "ts_ms": 1100, "run_id": "r1", "parent_id": "1",
         "data": {"agent": "lawyer"}},
        {"event_id": "3", "kind": "ToolCalled", "ts_ms": 1200, "run_id": "r1", "parent_id": "2",
         "data": {"tool": "statute_search"}},
        {"event_id": "4", "kind": "LLMResponded", "ts_ms": 1500, "run_id": "r1", "parent_id": "2",
         "data": {"usage": {"input_tokens": 1200, "output_tokens": 250, "cache_read_input_tokens": 600}}},
        {"event_id": "5", "kind": "RunFinished", "ts_ms": 5000, "run_id": "r1", "parent_id": "1",
         "data": {"answer_mode": "evidence_grounded"}},
    ]
    events_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in evs))
    m = derive_run_metrics(run_dir)
    assert m.total_latency_ms == 4000
    assert m.total_input_tokens == 1200
    assert m.total_output_tokens == 250
    assert m.cache_read_tokens == 600
    assert m.cache_hit_rate == pytest.approx(0.5)
    assert m.agent_invocations == 1
    assert m.tool_calls_total == 1
    assert m.final_answer_mode == "evidence_grounded"
    assert m.errors == 0


def test_derive_metrics_counts_errors(tmp_path):
    run_dir = tmp_path / "run-2"
    run_dir.mkdir()
    evs = [
        {"event_id": "1", "kind": "RunStarted", "ts_ms": 1000, "run_id": "r2", "parent_id": None, "data": {}},
        {"event_id": "2", "kind": "ToolFailed", "ts_ms": 1100, "run_id": "r2", "parent_id": "1",
         "data": {"error": "connection refused"}},
        {"event_id": "3", "kind": "RunFinished", "ts_ms": 2000, "run_id": "r2", "parent_id": "1", "data": {}},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in evs))
    m = derive_run_metrics(run_dir)
    assert m.errors == 1


def test_derive_metrics_missing_events_file(tmp_path):
    run_dir = tmp_path / "run-3"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        derive_run_metrics(run_dir)
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Implement** — but **first** read `multi_agent/tracing/recorder.py` and the existing event-kind enum to confirm event field names. The test uses `data.usage.input_tokens` etc. — verify that's how `LLMResponded` events are actually written. If actual field names differ, adapt both the metrics impl AND the test fixtures together so the test reflects reality.

Sketch:

```python
"""Trace-derived metrics (Phase 5b §7.6)."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_hit_rate: float = 0.0
    agent_invocations: int = 0
    tool_calls_total: int = 0
    react_steps_total: int = 0
    supervisor_verdict: str | None = None
    final_answer_mode: str | None = None
    citation_count: int = 0
    errors: int = 0


def derive_run_metrics(run_dir: Path) -> RunMetrics:
    events_path = Path(run_dir) / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(events_path)
    m = RunMetrics()
    start_ts = end_ts = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        kind = e.get("kind")
        data = e.get("data") or {}
        ts = e.get("ts_ms")
        if kind == "RunStarted":
            start_ts = ts
        elif kind == "RunFinished":
            end_ts = ts
            m.final_answer_mode = data.get("answer_mode")
        elif kind == "AgentInvoked":
            m.agent_invocations += 1
        elif kind == "ToolCalled":
            m.tool_calls_total += 1
        elif kind == "LLMResponded":
            usage = data.get("usage") or {}
            m.total_input_tokens += usage.get("input_tokens", 0) or 0
            m.total_output_tokens += usage.get("output_tokens", 0) or 0
            m.cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
        elif kind == "SupervisorVerdict":
            m.supervisor_verdict = data.get("verdict")
        if data.get("error") or "Failed" in (kind or ""):
            m.errors += 1
    if start_ts is not None and end_ts is not None:
        m.total_latency_ms = end_ts - start_ts
    if m.total_input_tokens > 0:
        m.cache_hit_rate = m.cache_read_tokens / m.total_input_tokens
    return m
```

- [ ] **Step 4: Verify pass + full suite** → 207 passed + 1 skipped.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/metrics.py experiments/multi_agent/tests/unit/test_metrics.py
git commit -m "phase5b(eval): derive_run_metrics from trace events"
```

---

## Task 3: CitationAccuracyJudge (rule-based)

**Files:**
- Create: `multi_agent/eval/judges/__init__.py` (empty)
- Create: `multi_agent/eval/judges/citation_accuracy.py`
- Create: `tests/unit/test_citation_accuracy.py`

Compares a `Query.expected.should_cite_any` against the actual citations in the Lawyer's final output. Reads `final_output.json` (or however the runner persists Lawyer output) for the run. **Pure rules, no LLM call.**

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_citation_accuracy.py
import pytest
from multi_agent.eval.judges.citation_accuracy import (
    CitationAccuracyJudge, CitationAccuracyResult,
)
from multi_agent.eval.queryset import Query, ExpectedAnswer


def test_citation_hit():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s",
              expected=ExpectedAnswer(should_cite_any=["民法典-510", "民法典-563"]))
    lawyer_output = {
        "citations": [
            {"law_short": "民法典", "article_no": "510", "excerpt": "..."},
        ],
    }
    j = CitationAccuracyJudge()
    r = j.judge(q, lawyer_output)
    assert r.hit is True
    assert "民法典-510" in r.matched


def test_citation_miss():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s",
              expected=ExpectedAnswer(should_cite_any=["民法典-510"]))
    lawyer_output = {"citations": [{"law_short": "民法典", "article_no": "999", "excerpt": ""}]}
    r = CitationAccuracyJudge().judge(q, lawyer_output)
    assert r.hit is False
    assert r.matched == []


def test_no_expectation_skipped():
    q = Query(id="q1", text="t", jurisdiction="CN", cause="c", source="s")
    r = CitationAccuracyJudge().judge(q, {"citations": []})
    assert r.skipped is True
```

- [ ] **Step 2: Implement**

```python
"""Rule-based citation accuracy judge (Phase 5b §7.7)."""
from __future__ import annotations
from pydantic import BaseModel, Field
from multi_agent.eval.queryset import Query


class CitationAccuracyResult(BaseModel):
    hit: bool = False
    matched: list[str] = Field(default_factory=list)
    expected: list[str] = Field(default_factory=list)
    actual: list[str] = Field(default_factory=list)
    skipped: bool = False
    reason: str = ""


class CitationAccuracyJudge:
    """Pass if any expected citation appears in lawyer output."""

    def judge(self, query: Query, lawyer_output: dict) -> CitationAccuracyResult:
        expected = query.expected.should_cite_any
        actual = [
            f"{c.get('law_short','')}-{c.get('article_no','')}"
            for c in (lawyer_output.get("citations") or [])
        ]
        if not expected:
            return CitationAccuracyResult(
                skipped=True, expected=[], actual=actual,
                reason="No should_cite_any expectation set",
            )
        matched = [c for c in actual if c in expected]
        return CitationAccuracyResult(
            hit=len(matched) > 0,
            matched=matched,
            expected=expected,
            actual=actual,
            reason="" if matched else "no expected citation present",
        )
```

- [ ] **Step 3: Verify pass + full suite** → 210 passed + 1 skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/judges/ experiments/multi_agent/tests/unit/test_citation_accuracy.py
git commit -m "phase5b(eval): CitationAccuracyJudge (rule-based)"
```

---

## Task 4: ExperimentRunner + RunGroup

**Files:**
- Create: `multi_agent/eval/runner.py`
- Create: `tests/unit/test_experiment_runner.py`

Walks a QuerySet, calls a user-supplied `query_runner(query) → run_id, lawyer_output` async, writes `run_groups/<group>/results.jsonl` + symlinks runs. Designed for **stub-injected query_runner** for unit tests; the real Qwen plumbing happens in the integration test Task 6.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_experiment_runner.py
import json
import pytest
from pathlib import Path
from multi_agent.eval.runner import ExperimentRunner, RunGroup
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query


@pytest.mark.asyncio
async def test_experiment_runner_writes_results_jsonl(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="a", text="qa", jurisdiction="CN", cause="c", source="s"),
        Query(id="b", text="qb", jurisdiction="CN", cause="c", source="s"),
    ])
    async def fake_runner(q):
        return {
            "run_id": f"run-{q.id}",
            "status": "ok",
            "lawyer_output": {"citations": [], "primary_answer": f"answer for {q.id}"},
            "run_dir": tmp_path / "runs" / f"run-{q.id}",
        }
    # Write fake event files so metrics derivation works
    for qid in ("a", "b"):
        run_dir = tmp_path / "runs" / f"run-{qid}"
        run_dir.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text(
            '{"event_id":"1","kind":"RunStarted","ts_ms":1000,"run_id":"x","parent_id":null,"data":{}}\n'
            '{"event_id":"2","kind":"RunFinished","ts_ms":2000,"run_id":"x","parent_id":"1","data":{}}\n'
        )

    runner = ExperimentRunner(
        query_set=qs,
        run_group_name="test-group",
        runs_root=tmp_path,
        query_runner=fake_runner,
    )
    group = await runner.run()
    assert group.group_dir.exists()
    results_path = group.group_dir / "results.jsonl"
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    assert {r["query_id"] for r in rows} == {"a", "b"}
    assert all(r["status"] == "ok" for r in rows)
    assert all("metrics" in r for r in rows)


@pytest.mark.asyncio
async def test_runner_records_failures(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="x", text="boom", jurisdiction="CN", cause="c", source="s"),
    ])
    async def bad_runner(q):
        raise RuntimeError("simulated provider failure")
    runner = ExperimentRunner(
        query_set=qs, run_group_name="g", runs_root=tmp_path, query_runner=bad_runner,
    )
    group = await runner.run()
    rows = [json.loads(l) for l in (group.group_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert rows[0]["status"] == "error"
    assert "simulated" in rows[0]["error"]
```

- [ ] **Step 2: Implement**

```python
"""ExperimentRunner (Phase 5b §7.4-7.5)."""
from __future__ import annotations
import asyncio
import json
import traceback
from pathlib import Path
from typing import Awaitable, Callable
from datetime import datetime
from pydantic import BaseModel

from multi_agent.eval.queryset import QuerySet, Query
from multi_agent.eval.metrics import derive_run_metrics
from multi_agent.eval.judges.citation_accuracy import CitationAccuracyJudge


class RunGroup(BaseModel):
    name: str
    group_dir: Path
    query_set_name: str

    model_config = {"arbitrary_types_allowed": True}


QueryRunner = Callable[[Query], Awaitable[dict]]


class ExperimentRunner:
    def __init__(
        self,
        *,
        query_set: QuerySet,
        run_group_name: str,
        runs_root: Path,
        query_runner: QueryRunner,
        parallelism: int = 1,
        group_root: Path | None = None,
    ):
        self.query_set = query_set
        self.run_group_name = run_group_name
        self.runs_root = Path(runs_root)
        self.query_runner = query_runner
        self.parallelism = parallelism
        self.group_root = Path(group_root) if group_root else self.runs_root / "run_groups"
        self.judge = CitationAccuracyJudge()

    async def run(self) -> RunGroup:
        group_dir = self.group_root / self.run_group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "group_meta.yaml").write_text(
            f"name: {self.run_group_name}\nquery_set: {self.query_set.meta.name}\n"
            f"created: {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
        results_path = group_dir / "results.jsonl"
        sem = asyncio.Semaphore(self.parallelism)

        async def one(q: Query) -> dict:
            async with sem:
                try:
                    out = await self.query_runner(q)
                    metrics = derive_run_metrics(out["run_dir"]).model_dump()
                    citation_result = self.judge.judge(q, out.get("lawyer_output") or {}).model_dump()
                    return {
                        "query_id": q.id,
                        "run_id": out["run_id"],
                        "status": out.get("status", "ok"),
                        "metrics": metrics,
                        "citation_judge": citation_result,
                    }
                except Exception as e:
                    return {
                        "query_id": q.id,
                        "status": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(),
                    }

        rows = await asyncio.gather(*[one(q) for q in self.query_set.queries])
        with results_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return RunGroup(name=self.run_group_name, group_dir=group_dir,
                        query_set_name=self.query_set.meta.name)
```

- [ ] **Step 3: Verify pass + full suite** → 212 passed + 1 skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/runner.py experiments/multi_agent/tests/unit/test_experiment_runner.py
git commit -m "phase5b(eval): ExperimentRunner + RunGroup"
```

---

## Task 5: summary.md report

**Files:**
- Create: `multi_agent/eval/report.py`
- Create: `tests/unit/test_report.py`

Aggregates `results.jsonl` → human-readable `summary.md` (per spec §7.5). Tables: latency p50/p95, token totals, citation hit rate, error count.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_report.py
import json
from pathlib import Path
import pytest
from multi_agent.eval.report import render_summary_md


def test_render_summary_md(tmp_path):
    group_dir = tmp_path / "g"
    group_dir.mkdir()
    rows = [
        {"query_id": "q1", "run_id": "r1", "status": "ok",
         "metrics": {"total_latency_ms": 1000, "total_input_tokens": 800,
                     "total_output_tokens": 200, "agent_invocations": 1,
                     "tool_calls_total": 2, "cache_hit_rate": 0.5,
                     "errors": 0, "final_answer_mode": "evidence_grounded",
                     "cache_read_tokens": 400, "react_steps_total": 0,
                     "supervisor_verdict": None, "citation_count": 0},
         "citation_judge": {"hit": True, "matched": ["民法典-510"], "expected": ["民法典-510"],
                            "actual": ["民法典-510"], "skipped": False, "reason": ""}},
        {"query_id": "q2", "run_id": "r2", "status": "ok",
         "metrics": {"total_latency_ms": 2000, "total_input_tokens": 1000,
                     "total_output_tokens": 300, "agent_invocations": 1,
                     "tool_calls_total": 1, "cache_hit_rate": 0.4,
                     "errors": 0, "final_answer_mode": "evidence_grounded",
                     "cache_read_tokens": 400, "react_steps_total": 0,
                     "supervisor_verdict": None, "citation_count": 0},
         "citation_judge": {"hit": False, "matched": [], "expected": ["民法典-999"],
                            "actual": [], "skipped": False, "reason": "no expected"}},
    ]
    results = group_dir / "results.jsonl"
    results.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summary_path = render_summary_md(group_dir)
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "总计" in md or "Total" in md
    assert "1/2" in md or "50" in md  # citation hit rate
    assert "q1" in md and "q2" in md
```

- [ ] **Step 2: Implement**

```python
"""Render summary.md from a RunGroup's results.jsonl (Phase 5b §7.5)."""
from __future__ import annotations
import json
import statistics
from pathlib import Path


def render_summary_md(group_dir: Path) -> Path:
    group_dir = Path(group_dir)
    results = [json.loads(l) for l in (group_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    n = len(results)
    ok = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") != "ok"]
    latencies = [r["metrics"]["total_latency_ms"] for r in ok if "metrics" in r]
    in_tok = sum(r["metrics"].get("total_input_tokens", 0) for r in ok)
    out_tok = sum(r["metrics"].get("total_output_tokens", 0) for r in ok)
    cache_in = sum(r["metrics"].get("cache_read_tokens", 0) for r in ok)
    citation_hits = sum(1 for r in ok if r.get("citation_judge", {}).get("hit"))
    citation_scored = sum(1 for r in ok if not r.get("citation_judge", {}).get("skipped", True))
    p50 = int(statistics.median(latencies)) if latencies else 0
    p95 = int(statistics.quantiles(latencies, n=20)[18]) if len(latencies) >= 5 else (max(latencies) if latencies else 0)

    lines = [
        f"# RunGroup `{group_dir.name}` 汇总\n",
        f"- 总计 Total: **{n}** queries (ok={len(ok)}, error={len(errs)})",
        f"- 延迟 latency: p50={p50}ms, p95={p95}ms",
        f"- Tokens: input={in_tok}, output={out_tok}, cache_read={cache_in}, hit_rate={cache_in/in_tok if in_tok else 0:.2f}",
        f"- Citation accuracy: **{citation_hits}/{citation_scored}** ({100*citation_hits/citation_scored if citation_scored else 0:.0f}%)",
        "",
        "## 逐 Query",
        "",
        "| Query | Status | Latency | Tokens (in/out) | Citation | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['query_id']} | error | — | — | — | {r.get('error','')[:40]} |")
            continue
        m = r["metrics"]
        cj = r.get("citation_judge", {})
        cit = "skip" if cj.get("skipped") else ("✓" if cj.get("hit") else "✗")
        lines.append(
            f"| {r['query_id']} | ok | {m['total_latency_ms']}ms | "
            f"{m['total_input_tokens']}/{m['total_output_tokens']} | {cit} | "
            f"{m.get('final_answer_mode') or ''} |"
        )
    out = group_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
```

- [ ] **Step 3: Verify pass + full suite** → 213 passed + 1 skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/report.py experiments/multi_agent/tests/unit/test_report.py
git commit -m "phase5b(eval): render_summary_md from results.jsonl"
```

---

## Task 6: Real Qwen E2E + tag

**Files:**
- Create: `tests/integration/test_eval_e2e.py`

Run `synthetic_seed_v1.yaml` (5 queries) through `LawyerAgent` with real Qwen + `StatuteSearchTool`, derive metrics, produce `summary.md`, assert citation hit rate ≥ 1/5 and zero errors.

- [ ] **Step 1: Write test**

```python
# tests/integration/test_eval_e2e.py
"""Phase 5b E2E: synthetic_seed_v1 through real Qwen Lawyer pipeline."""
import uuid
import httpx
import pytest
import json
from pathlib import Path

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.report import render_summary_md
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def real_corpus_index(tmp_path_factory):
    """Build a small index from real Chinese-Laws files covering 民法典 + 道交法."""
    from multi_agent.tools.retrievers.index_builder import build_index
    from multi_agent.schemas.document import Document, Chunk
    name = f"eval_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    chunks = [
        ("民法典-510", "民法典", "510", "当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
        ("民法典-563", "民法典", "563", "有下列情形之一的,当事人可以解除合同..."),
        ("民法典-577", "民法典", "577", "当事人一方不履行合同义务...应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"),
        ("民法典-584", "民法典", "584", "当事人一方不履行合同义务...造成对方损失的,损失赔偿额应当相当于因违约所造成的损失..."),
        ("民法典-703", "民法典", "703", "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ("民法典-707", "民法典", "707", "租赁期限六个月以上的,应当采用书面形式。"),
        ("民法典-1165", "民法典", "1165", "行为人因过错侵害他人民事权益造成损害的,应当承担侵权责任。"),
        ("民法典-1184", "民法典", "1184", "侵害他人财产的,财产损失按照损失发生时的市场价格或者其他合理方式计算。"),
        ("道路交通安全法-76", "道路交通安全法", "76", "机动车发生交通事故造成人身伤亡、财产损失的..."),
    ]
    doc = Document(
        law_name="中华人民共和国民法典", law_short="民法典",
        source_path="composite",
        chunks=[Chunk(doc_id=did, law_name="composite", law_short=ls, article_no=an, text=t)
                for (did, ls, an, t) in chunks],
    )
    build_index(documents=[doc], collection_name=name, sparse_artifact_path=sparse_path,
                dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_synthetic_seed_v1_through_lawyer(real_corpus_index, tmp_path):
    qs_path = Path(__file__).parents[2] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(qs_path)
    # Skip the smalltalk query (q005) — Lawyer isn't the right entry point
    qs.queries = [q for q in qs.queries if "smalltalk" not in q.tags]

    provider = OpenAICompatibleProvider()
    statute_search = StatuteSearchTool(
        collection_name=real_corpus_index["collection"],
        sparse_artifact_path=real_corpus_index["sparse_path"],
    )

    async def run_one(q):
        result = await run_query(
            query=q.text,
            agent_factory=lambda p, r: LawyerAgent(
                name="lawyer", role="advisor",
                provider=p, recorder=r,
                tools=[statute_search],
                model="qwen3.5-9b", specialty="民事",
                max_steps=6, max_tool_calls=8,
            ),
            provider=provider,
            runs_root=tmp_path / "runs",
            config={},
        )
        try:
            lo = json.loads(result.get("final_answer") or "{}")
        except Exception:
            lo = {}
        return {"run_id": result["run_id"], "status": result.get("status", "ok"),
                "lawyer_output": lo, "run_dir": Path(result["run_dir"])}

    runner = ExperimentRunner(
        query_set=qs, run_group_name="phase5b-seed-run",
        runs_root=tmp_path, query_runner=run_one, parallelism=2,
    )
    group = await runner.run()
    summary = render_summary_md(group.group_dir)
    assert summary.exists()
    rows = [json.loads(l) for l in (group.group_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert all(r["status"] == "ok" for r in rows), [r for r in rows if r["status"] != "ok"]
    # At least one citation should hit on the rental query
    hits = sum(1 for r in rows if r.get("citation_judge", {}).get("hit"))
    assert hits >= 1, f"No citation hits across {len(rows)} queries"
```

- [ ] **Step 2: Run + commit + tag**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_eval_e2e.py -v -s 2>&1 | tail -60"
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -10"

cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_eval_e2e.py
git commit -m "phase5b(integration): synthetic_seed_v1 E2E through real Qwen Lawyer"
git tag -a phase5b-eval-mvp -m "Phase 5b: minimum viable eval (QuerySet + Runner + metrics + citation judge)"
git tag -l "phase*"
```

May take 3-5 min for 4 queries through Qwen at parallelism=2.

---

## Acceptance Criteria

Phase 5b complete when:

1. Full pytest passes (~214 tests after 12 new unit tests)
2. `synthetic_seed_v1.yaml` loads and validates
3. `derive_run_metrics` extracts ≥6 fields from real run events
4. `CitationAccuracyJudge` correctly classifies hit/miss/skip
5. `ExperimentRunner.run()` produces `results.jsonl` + writes group_meta + symlinks
6. `render_summary_md` produces a sensible Markdown table
7. Real-Qwen E2E on 4 seed queries: zero errors, ≥1 citation hit, `summary.md` rendered
8. Tag `phase5b-eval-mvp` exists

## Out of Scope (Phase 5c+)

- LLM Judges (Claude-Opus-based Groundedness / Helpfulness)
- Comparator (group-vs-group diff)
- AblationRunner
- LatencyProfiler (SpanTiming derivation)
- Trace Viewer (Streamlit)
- Large-scale golden_qa_v1 extraction from laws_data
- Cost tracking (cost_usd derivation per spec §7.6)
