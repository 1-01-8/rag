# Phase 5c — LLM Judges + Comparator Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Implement spec §7.7 LLM-based judges (`GroundednessJudge`, `HelpfulnessJudge` using **Claude Opus** for all runs — spec §7.7 rule) and §7.8 `Comparator` for group-vs-group diffs. After this phase, we can answer the headline question: *Qwen-9B-local vs Claude-Opus, which produces more grounded answers?*

**Phase 5b starting point:** Tag `phase5b-eval-mvp`. 214 tests + 1 skipped.

---

## Cost guardrails

- All unit tests use `StubProvider` — **no API calls**.
- Exactly **one** integration test hits the real Anthropic API, gated behind `ANTHROPIC_API_KEY` env var (skip if absent). Uses 2 queries × 2 judges ≈ 4 LLM calls, each ~$0.005–0.02 → **bounded at ~$0.10 worst case**.
- The integration test is `@pytest.mark.expensive` so default pytest runs skip it unless `-m expensive` is passed.

---

## Out of scope (Phase 5d+)

- AblationRunner (DisableAgent / SwapModel / DisableTool)
- LatencyProfiler (SpanTiming)
- Trace Viewer (Streamlit)
- LLM-judge caching across runs (re-runs cost money each time)
- Faithfulness scoring at evidence-span granularity (current judges score per-answer)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   └── eval/
│       ├── judges/
│       │   ├── base.py                  # LLMJudge ABC + render/parse helpers
│       │   ├── groundedness.py          # Claude-based, JSON output
│       │   └── helpfulness.py           # Claude-based, JSON output
│       ├── comparator.py                # Comparator + ComparisonReport
│       └── runner.py                    # MODIFIED: optionally invoke judges
└── tests/
    ├── unit/
    │   ├── test_judge_base.py
    │   ├── test_groundedness_judge.py
    │   ├── test_helpfulness_judge.py
    │   ├── test_runner_with_judges.py
    │   └── test_comparator.py
    └── integration/
        └── test_claude_judges_e2e.py   # @pytest.mark.expensive, ANTHROPIC_API_KEY-gated
```

---

## Task 1: LLMJudge base + JSON parsing

**Files:**
- Create: `multi_agent/eval/judges/base.py`
- Create: `tests/unit/test_judge_base.py`

Abstract base providing prompt templating, async LLM call via injected `LLMProvider`, JSON-robust parsing of judge output, and standard `JudgeResult` shape (score 0–1 + ungrounded/unhelpful claims + rationale).

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_judge_base.py
import pytest
from pydantic import BaseModel
from multi_agent.eval.judges.base import LLMJudge, JudgeResult
from multi_agent.providers.stub import StubProvider, ScriptedResponse


class _DummyOut(BaseModel):
    score: float
    issues: list[str] = []


class _DummyJudge(LLMJudge[_DummyOut]):
    name = "dummy"
    output_schema = _DummyOut

    def render_prompt(self, *, query, lawyer_output, evidence_pool) -> str:
        return f"judge: {query} -> {lawyer_output}"


@pytest.mark.asyncio
async def test_judge_calls_provider_and_parses():
    p = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.8, "issues": ["x"]}', finish_reason="end_turn"),
    ])
    j = _DummyJudge(provider=p, model="stub")
    result = await j.judge(query="Q", lawyer_output={"a": 1}, evidence_pool=[])
    assert isinstance(result, JudgeResult)
    assert result.judge == "dummy"
    assert result.score == 0.8
    assert result.parsed.issues == ["x"]
    assert result.error is None


@pytest.mark.asyncio
async def test_judge_handles_malformed_json():
    p = StubProvider(responses=[
        ScriptedResponse(text='not json', finish_reason="end_turn"),
    ])
    j = _DummyJudge(provider=p, model="stub")
    result = await j.judge(query="Q", lawyer_output={}, evidence_pool=[])
    assert result.error is not None
    assert result.score == 0.0
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Implement**

```python
"""LLMJudge base class (Phase 5c §7.7)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from multi_agent.providers.base import LLMProvider
from multi_agent.providers.json_robust import parse_json_robust

T = TypeVar("T", bound=BaseModel)


class JudgeResult(BaseModel):
    judge: str
    score: float = 0.0
    parsed: BaseModel | None = None
    raw: str = ""
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class LLMJudge(ABC, Generic[T]):
    name: str = "base"
    output_schema: type[T]
    system_prompt: str = "You are an evaluation judge. Output only JSON."
    temperature: float = 0.0
    max_tokens: int = 1024

    def __init__(self, *, provider: LLMProvider, model: str):
        self.provider = provider
        self.model = model

    @abstractmethod
    def render_prompt(self, *, query: str, lawyer_output: dict, evidence_pool: list[dict]) -> str: ...

    async def judge(self, *, query: str, lawyer_output: dict, evidence_pool: list[dict]) -> JudgeResult:
        user = self.render_prompt(query=query, lawyer_output=lawyer_output, evidence_pool=evidence_pool)
        try:
            resp = await self.provider.complete(
                model=self.model,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user}],
                tools=None,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = (resp.text or "").strip()
            parsed_dict = parse_json_robust(raw)
            parsed = self.output_schema.model_validate(parsed_dict)
            score = float(getattr(parsed, "score", 0.0))
            return JudgeResult(judge=self.name, score=score, parsed=parsed, raw=raw)
        except Exception as e:
            return JudgeResult(judge=self.name, score=0.0, raw="", error=f"{type(e).__name__}: {e}")
```

**Adapt** the call signature if `LLMProvider.complete()` returns something different from `.text` — read `multi_agent/providers/base.py` first.

- [ ] **Step 4: Verify pass + full suite** → 216 passed + 1 skipped.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/judges/base.py experiments/multi_agent/tests/unit/test_judge_base.py
git commit -m "phase5c(eval): LLMJudge base + JSON-robust parsing"
```

---

## Task 2: GroundednessJudge

**Files:**
- Create: `multi_agent/eval/judges/groundedness.py`
- Create: `tests/unit/test_groundedness_judge.py`

Judges: does every factual claim in the Lawyer's answer trace back to evidence in the pool? Output: `{score: 0..1, ungrounded_claims: [str], rationale: str}`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_groundedness_judge.py
import pytest
from multi_agent.eval.judges.groundedness import GroundednessJudge, GroundednessOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_groundedness_judge_grounded_answer():
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.95, "ungrounded_claims": [], "rationale": "All claims have evidence"}',
            finish_reason="end_turn",
        ),
    ])
    j = GroundednessJudge(provider=p, model="stub")
    result = await j.judge(
        query="房东合同期内涨租 30% 合法吗?",
        lawyer_output={"primary_answer": "不合法", "citations": [
            {"law_short": "民法典", "article_no": "703", "excerpt": "..."}
        ]},
        evidence_pool=[{"doc_id": "民法典-703", "law_short": "民法典", "article_no": "703",
                        "text": "租赁合同..."}],
    )
    assert result.error is None
    assert result.score == 0.95
    assert result.parsed.ungrounded_claims == []


@pytest.mark.asyncio
async def test_groundedness_judge_flags_hallucination():
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"score": 0.3, "ungrounded_claims": ["claim about 民法典-999 with no source"], "rationale": "Hallucinated cite"}',
            finish_reason="end_turn",
        ),
    ])
    j = GroundednessJudge(provider=p, model="stub")
    result = await j.judge(query="Q", lawyer_output={"primary_answer": "..."}, evidence_pool=[])
    assert result.score == 0.3
    assert len(result.parsed.ungrounded_claims) == 1
```

- [ ] **Step 2: Implement**

```python
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

    def render_prompt(self, *, query, lawyer_output, evidence_pool) -> str:
        return _PROMPT.format(
            query=query,
            lawyer_output=_json.dumps(lawyer_output, ensure_ascii=False, indent=2),
            evidence_pool=_json.dumps(evidence_pool, ensure_ascii=False, indent=2),
        )
```

- [ ] **Step 3: Verify pass + full suite** → 218 passed + 1 skipped.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/judges/groundedness.py experiments/multi_agent/tests/unit/test_groundedness_judge.py
git commit -m "phase5c(eval): GroundednessJudge (Claude-based)"
```

---

## Task 3: HelpfulnessJudge

**Files:**
- Create: `multi_agent/eval/judges/helpfulness.py`
- Create: `tests/unit/test_helpfulness_judge.py`

Judges: does the answer actually answer the user's question and is it actionable? Output: `{score: 0..1, missing_aspects: [str], rationale: str}`.

Mirror Task 2 structure exactly. Prompt focuses on "用户能根据这个答复做什么"(actionability) + completeness.

- [ ] **Test (2 tests)**
- [ ] **Implement**
- [ ] **Verify** → 220 passed + 1 skipped.
- [ ] **Commit:** `phase5c(eval): HelpfulnessJudge (Claude-based)`

---

## Task 4: Runner integration — optional judges

**Files:**
- Modify: `multi_agent/eval/runner.py`
- Create: `tests/unit/test_runner_with_judges.py`

Extend `ExperimentRunner` with optional `judges: list[LLMJudge]`. Each result row gets a `judges: {judge_name: JudgeResult.model_dump()}` field. Judges run in parallel per query (after the run completes) but are bounded by a separate `judge_semaphore` so we don't blast Anthropic.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_runner_with_judges.py
import pytest
import json
from pathlib import Path
from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query
from multi_agent.eval.judges.groundedness import GroundednessJudge
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_runner_attaches_judge_results(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="a", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(
        '{"event_id":"1","event_type":"RunStarted","timestamp":"2026-05-15T00:00:00","run_id":"x","parent_id":null}\n'
        '{"event_id":"2","event_type":"RunFinished","timestamp":"2026-05-15T00:00:02","run_id":"x","parent_id":"1"}\n'
    )

    async def runner(q):
        return {"run_id": "run-a", "status": "ok",
                "lawyer_output": {"primary_answer": "answer"}, "run_dir": run_dir}

    judge_provider = StubProvider(responses=[
        ScriptedResponse(text='{"score": 0.9, "ungrounded_claims": [], "rationale": "ok"}',
                        finish_reason="end_turn"),
    ])
    judges = [GroundednessJudge(provider=judge_provider, model="stub")]

    exp = ExperimentRunner(
        query_set=qs, run_group_name="g", runs_root=tmp_path,
        query_runner=runner, judges=judges,
    )
    group = await exp.run()
    rows = [json.loads(l) for l in (group.group_dir / "results.jsonl").read_text().splitlines() if l.strip()]
    assert "judges" in rows[0]
    assert rows[0]["judges"]["groundedness"]["score"] == 0.9
```

- [ ] **Step 2: Modify runner.py**

Add `judges: list[LLMJudge] | None = None` to `__init__`, store on self. After deriving metrics in `one(q)`, if `self.judges`, gather their results concurrently:

```python
if self.judges:
    j_results = await asyncio.gather(*[
        j.judge(query=q.text, lawyer_output=out.get("lawyer_output") or {},
                evidence_pool=out.get("evidence_pool") or [])
        for j in self.judges
    ])
    row["judges"] = {j.name: r.model_dump() for j, r in zip(self.judges, j_results)}
```

Also: extend `summary.md` to include judge averages. Update existing `test_report.py` if necessary (or add a separate test for judge-aware summary).

- [ ] **Step 3: Verify pass + full suite** → 221 passed + 1 skipped.

- [ ] **Step 4: Commit:** `phase5c(eval): ExperimentRunner attaches optional LLM judges`

---

## Task 5: Comparator

**Files:**
- Create: `multi_agent/eval/comparator.py`
- Create: `tests/unit/test_comparator.py`

Loads two `RunGroup`s, joins on `query_id`, computes diffs: per-query latency delta, token delta, citation_judge agreement, judge-score delta. Output: `ComparisonReport` (Pydantic) + `comparator.render_md(report) → Path`.

- [ ] **Step 1: Failing test (2-3 tests)**

```python
@pytest.mark.asyncio
async def test_comparator_diffs_two_groups(tmp_path):
    # Build two fake group dirs each with results.jsonl
    ga = tmp_path / "ga"; ga.mkdir()
    gb = tmp_path / "gb"; gb.mkdir()
    rows_a = [
        {"query_id": "q1", "status": "ok",
         "metrics": {"total_latency_ms": 1000, "total_input_tokens": 800,
                     "total_output_tokens": 200, "cache_read_tokens": 0,
                     "cache_hit_rate": 0, "agent_invocations": 1, "tool_calls_total": 2,
                     "react_steps_total": 0, "errors": 0,
                     "final_answer_mode": "evidence_grounded", "citation_count": 1,
                     "supervisor_verdict": None},
         "citation_judge": {"hit": True, "matched": ["民法典-510"], "expected": ["民法典-510"],
                            "actual": ["民法典-510"], "skipped": False, "reason": ""},
         "judges": {"groundedness": {"judge": "groundedness", "score": 0.9, "parsed": None, "raw": "", "error": None}}},
    ]
    rows_b = [{**rows_a[0], "metrics": {**rows_a[0]["metrics"], "total_latency_ms": 2000},
               "judges": {"groundedness": {"judge": "groundedness", "score": 0.7, "parsed": None, "raw": "", "error": None}}}]
    (ga / "results.jsonl").write_text(json.dumps(rows_a[0], ensure_ascii=False) + "\n")
    (gb / "results.jsonl").write_text(json.dumps(rows_b[0], ensure_ascii=False) + "\n")

    from multi_agent.eval.comparator import Comparator
    report = Comparator().compare(group_a_dir=ga, group_b_dir=gb)
    assert report.n_common == 1
    assert report.per_query[0].latency_delta_ms == 1000   # b is slower
    assert report.per_query[0].groundedness_delta == pytest.approx(-0.2)  # b worse
```

- [ ] **Step 2: Implement**

Sketch:

```python
"""Comparator (Phase 5c §7.8)."""
from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, Field


class PerQueryDelta(BaseModel):
    query_id: str
    latency_delta_ms: int = 0
    in_tokens_delta: int = 0
    out_tokens_delta: int = 0
    citation_hit_a: bool = False
    citation_hit_b: bool = False
    groundedness_delta: float | None = None
    helpfulness_delta: float | None = None


class ComparisonReport(BaseModel):
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    n_common: int
    per_query: list[PerQueryDelta] = Field(default_factory=list)


class Comparator:
    def _load(self, group_dir: Path) -> dict[str, dict]:
        rows = [json.loads(l) for l in (Path(group_dir) / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        return {r["query_id"]: r for r in rows}

    def compare(self, *, group_a_dir: Path, group_b_dir: Path) -> ComparisonReport:
        a = self._load(Path(group_a_dir))
        b = self._load(Path(group_b_dir))
        common = sorted(set(a) & set(b))
        per_q = []
        for qid in common:
            ra, rb = a[qid], b[qid]
            ma, mb = ra.get("metrics", {}), rb.get("metrics", {})
            ja = (ra.get("judges") or {}).get("groundedness") or {}
            jb = (rb.get("judges") or {}).get("groundedness") or {}
            ha = (ra.get("judges") or {}).get("helpfulness") or {}
            hb = (rb.get("judges") or {}).get("helpfulness") or {}
            per_q.append(PerQueryDelta(
                query_id=qid,
                latency_delta_ms=mb.get("total_latency_ms", 0) - ma.get("total_latency_ms", 0),
                in_tokens_delta=mb.get("total_input_tokens", 0) - ma.get("total_input_tokens", 0),
                out_tokens_delta=mb.get("total_output_tokens", 0) - ma.get("total_output_tokens", 0),
                citation_hit_a=bool((ra.get("citation_judge") or {}).get("hit")),
                citation_hit_b=bool((rb.get("citation_judge") or {}).get("hit")),
                groundedness_delta=(jb.get("score") - ja.get("score")) if ja and jb and ja.get("score") is not None and jb.get("score") is not None else None,
                helpfulness_delta=(hb.get("score") - ha.get("score")) if ha and hb and ha.get("score") is not None and hb.get("score") is not None else None,
            ))
        return ComparisonReport(
            group_a=Path(group_a_dir).name, group_b=Path(group_b_dir).name,
            n_a=len(a), n_b=len(b), n_common=len(common), per_query=per_q,
        )

    def render_md(self, report: ComparisonReport, out_dir: Path) -> Path:
        # ... per-query table with deltas, summary stats ...
        lines = [f"# Comparison `{report.group_a}` vs `{report.group_b}`",
                 f"- Common queries: {report.n_common}",
                 "",
                 "| Query | Δlat ms | Δin tok | Δout tok | Cite A | Cite B | Δgrounded | Δhelpful |",
                 "|---|---|---|---|---|---|---|---|"]
        for p in report.per_query:
            lines.append(f"| {p.query_id} | {p.latency_delta_ms:+d} | {p.in_tokens_delta:+d} | "
                        f"{p.out_tokens_delta:+d} | {'✓' if p.citation_hit_a else '✗'} | "
                        f"{'✓' if p.citation_hit_b else '✗'} | "
                        f"{f'{p.groundedness_delta:+.2f}' if p.groundedness_delta is not None else '—'} | "
                        f"{f'{p.helpfulness_delta:+.2f}' if p.helpfulness_delta is not None else '—'} |")
        out = Path(out_dir) / f"{report.group_a}_vs_{report.group_b}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
```

- [ ] **Step 3: Verify pass + full suite** → 224 passed + 1 skipped.

- [ ] **Step 4: Commit:** `phase5c(eval): Comparator + render_md`

---

## Task 6: Real Claude judges E2E (gated)

**Files:**
- Create: `tests/integration/test_claude_judges_e2e.py`
- Modify: `pyproject.toml` to register `expensive` marker if not present

Pull `phase5b-seed-run` results (or build a fresh 2-query run with stub Lawyer to keep cost down), invoke GroundednessJudge + HelpfulnessJudge against real Claude Opus, assert non-zero scores + no parse errors.

- [ ] **Step 1: Test**

```python
"""Phase 5c integration — REAL Anthropic API. ~$0.10/run."""
import os
import pytest
from multi_agent.eval.judges.groundedness import GroundednessJudge
from multi_agent.eval.judges.helpfulness import HelpfulnessJudge
from multi_agent.providers.anthropic import AnthropicProvider


pytestmark = [
    pytest.mark.expensive,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                      reason="ANTHROPIC_API_KEY not set"),
]


@pytest.mark.asyncio
async def test_real_claude_judges_score_grounded_answer():
    provider = AnthropicProvider()
    model = "claude-opus-4-7"
    lawyer_output = {
        "primary_answer": "根据《民法典》第703条,租赁合同是出租人将租赁物交付承租人...",
        "citations": [{"law_short": "民法典", "article_no": "703",
                       "excerpt": "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同"}],
    }
    evidence_pool = [{
        "doc_id": "民法典-703", "law_short": "民法典", "article_no": "703",
        "text": "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。",
    }]

    g = GroundednessJudge(provider=provider, model=model)
    h = HelpfulnessJudge(provider=provider, model=model)
    g_res = await g.judge(query="什么是租赁合同?", lawyer_output=lawyer_output, evidence_pool=evidence_pool)
    h_res = await h.judge(query="什么是租赁合同?", lawyer_output=lawyer_output, evidence_pool=evidence_pool)
    assert g_res.error is None, g_res.error
    assert h_res.error is None, h_res.error
    assert g_res.score >= 0.7   # well-grounded
    assert h_res.score >= 0.5   # answer addresses the question
```

- [ ] **Step 2: Run (only if user has key)**

```bash
ANTHROPIC_API_KEY=... conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_claude_judges_e2e.py -v -m expensive"
```

If no key available, skip the run but verify the test file parses + marker works:
```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_claude_judges_e2e.py -v"
# Should show: 1 skipped (ANTHROPIC_API_KEY not set)
```

- [ ] **Step 3: Tag**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_claude_judges_e2e.py experiments/multi_agent/pyproject.toml
git commit -m "phase5c(integration): Claude Opus judges E2E (gated)"
git tag -a phase5c-llm-judges -m "Phase 5c: LLM judges (Claude Opus) + Comparator"
git tag -l "phase*"
```

---

## Acceptance Criteria

Phase 5c complete when:

1. Full pytest passes (~224 tests; integration skipped if no key)
2. `LLMJudge` base handles malformed JSON without raising
3. `GroundednessJudge` + `HelpfulnessJudge` parse Claude output correctly
4. `ExperimentRunner` attaches `judges:` to each result row when judges provided
5. `Comparator.compare()` computes per-query deltas including judge scores
6. `Comparator.render_md()` produces a sensible diff table
7. (If `ANTHROPIC_API_KEY` set) real Claude scores grounded reference answer ≥0.7
8. Tag `phase5c-llm-judges` exists

## Out of Scope (Phase 5d+)

- AblationRunner
- LatencyProfiler (SpanTiming)
- Trace Viewer (Streamlit)
- Comparator failure-mode clustering
- Judge result caching / dedup
