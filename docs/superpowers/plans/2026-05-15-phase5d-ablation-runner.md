# Phase 5d — AblationRunner Implementation Plan

> Sub-skill: superpowers:subagent-driven-development

**Goal:** Spec §7.9 — `AblationRunner` to systematically vary one factor at a time (disable an agent, swap a model, disable a tool, disable memory) and report the per-ablation impact on metrics. This is the lever we'll use later to answer: *how much does the Supervisor actually contribute? Does removing case retrieval hurt or help?*

**Phase 5c starting point:** Tag `phase5c-llm-judges`. ~210 unit tests + 1 skipped + integrations.

---

## Out of scope (Phase 5e+)

- LatencyProfiler (SpanTiming — Phase 5e)
- Streamlit Trace Viewer (Phase 5f)
- Headline Qwen-vs-Claude experiment (requires ANTHROPIC_API_KEY)
- Ablation result-summary plotting (CLI only)

---

## File Structure

```
experiments/multi_agent/
├── multi_agent/
│   └── eval/
│       ├── ablations.py             # Ablation ABC + 4 concrete subclasses + apply()
│       └── ablation_runner.py       # AblationRunner + AblationReport
└── tests/
    └── unit/
        ├── test_ablations.py
        └── test_ablation_runner.py
```

---

## Task 1: Ablation primitives

**Files:**
- Create: `multi_agent/eval/ablations.py`
- Create: `tests/unit/test_ablations.py`

Each `Ablation` is a Pydantic dataclass + an `apply(config: AblationConfig)` that mutates a `RunConfig`-shaped dict so the query_runner can honor it. The runner doesn't introspect into agent internals — instead, ablations express their effect declaratively (e.g., `DisableTool(tool="case_search")` sets `disabled_tools={"case_search"}` in config) and the caller's `query_runner` reads that config.

This keeps the runner clean and lets the caller's `query_runner` factory decide how to honor each ablation.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_ablations.py
import pytest
from multi_agent.eval.ablations import (
    Ablation, DisableAgent, SwapModel, DisableTool, DisableMemory, apply_ablation,
)


def test_disable_tool_writes_into_config():
    cfg = {}
    ab = DisableTool(tool="case_search")
    apply_ablation(cfg, ab)
    assert "case_search" in cfg.get("disabled_tools", set())


def test_swap_model_writes_provider_and_model():
    cfg = {}
    ab = SwapModel(agent="lawyer", provider="anthropic", model="claude-opus-4-7")
    apply_ablation(cfg, ab)
    overrides = cfg.get("model_overrides", {})
    assert overrides["lawyer"]["provider"] == "anthropic"
    assert overrides["lawyer"]["model"] == "claude-opus-4-7"


def test_disable_agent():
    cfg = {}
    apply_ablation(cfg, DisableAgent(agent="supervisor"))
    assert "supervisor" in cfg.get("disabled_agents", set())


def test_disable_memory():
    cfg = {}
    apply_ablation(cfg, DisableMemory())
    assert cfg.get("disable_memory") is True


def test_ablation_name_for_reporting():
    assert DisableTool(tool="case_search").name == "disable_tool:case_search"
    assert SwapModel(agent="lawyer", provider="anthropic", model="claude-opus-4-7").name == "swap_model:lawyer→claude-opus-4-7"
    assert DisableAgent(agent="supervisor").name == "disable_agent:supervisor"
    assert DisableMemory().name == "disable_memory"
```

- [ ] **Step 2: Implement**

```python
"""Ablation primitives (Phase 5d §7.9)."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class Ablation(BaseModel):
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()


class DisableAgent(Ablation):
    agent: str

    @property
    def name(self) -> str:
        return f"disable_agent:{self.agent}"


class SwapModel(Ablation):
    agent: str
    provider: str
    model: str

    @property
    def name(self) -> str:
        return f"swap_model:{self.agent}→{self.model}"


class DisableTool(Ablation):
    tool: str

    @property
    def name(self) -> str:
        return f"disable_tool:{self.tool}"


class DisableMemory(Ablation):
    @property
    def name(self) -> str:
        return "disable_memory"


def apply_ablation(config: dict[str, Any], ablation: Ablation) -> None:
    """Mutate `config` in place to express `ablation`."""
    if isinstance(ablation, DisableAgent):
        config.setdefault("disabled_agents", set()).add(ablation.agent)
    elif isinstance(ablation, SwapModel):
        config.setdefault("model_overrides", {})[ablation.agent] = {
            "provider": ablation.provider,
            "model": ablation.model,
        }
    elif isinstance(ablation, DisableTool):
        config.setdefault("disabled_tools", set()).add(ablation.tool)
    elif isinstance(ablation, DisableMemory):
        config["disable_memory"] = True
    else:
        raise ValueError(f"Unknown ablation: {ablation}")
```

- [ ] **Step 3: Verify pass + full unit suite** → 212-213 passed.

- [ ] **Step 4: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/ablations.py experiments/multi_agent/tests/unit/test_ablations.py
git commit -m "phase5d(eval): Ablation primitives (DisableAgent/SwapModel/DisableTool/DisableMemory)"
```

---

## Task 2: AblationRunner + AblationReport

**Files:**
- Create: `multi_agent/eval/ablation_runner.py`
- Create: `tests/unit/test_ablation_runner.py`

`AblationRunner` runs (baseline + N ablations) × QuerySet by repeatedly invoking `ExperimentRunner` with a different config per ablation. Each ablation produces its own `RunGroup` under `run_groups/<base>__<ablation_name>/`. Final `AblationReport` summarizes deltas vs baseline.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_ablation_runner.py
import asyncio
import json
import pytest
from pathlib import Path
from multi_agent.eval.queryset import QuerySet, QuerySetMeta, Query
from multi_agent.eval.ablations import DisableTool, DisableMemory
from multi_agent.eval.ablation_runner import AblationRunner


def _fake_events(run_dir: Path, latency_ms: int = 1000):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "events.jsonl").write_text(
        f'{{"event_id":"1","event_type":"RunStarted","timestamp":"2026-05-15T00:00:00","run_id":"x","parent_id":null}}\n'
        f'{{"event_id":"2","event_type":"RunFinished","timestamp":"2026-05-15T00:00:0{latency_ms//1000}","run_id":"x","parent_id":"1"}}\n'
    )


@pytest.mark.asyncio
async def test_ablation_runner_runs_baseline_plus_ablations(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="q1", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])

    seen_configs: list[dict] = []

    async def query_runner_factory(config: dict):
        async def runner(q):
            seen_configs.append(dict(config))
            run_dir = tmp_path / "runs" / f"{config.get('label','base')}-{q.id}"
            _fake_events(run_dir)
            return {"run_id": run_dir.name, "status": "ok",
                    "lawyer_output": {"citations": []}, "run_dir": run_dir}
        return runner

    ar = AblationRunner(
        query_set=qs,
        runs_root=tmp_path,
        query_runner_factory=query_runner_factory,
        run_group_base="ab-test",
    )
    report = await ar.run(ablations=[DisableTool(tool="case_search"), DisableMemory()])
    assert report.n_ablations == 2
    assert report.baseline.group_dir.exists()
    assert len(report.ablations) == 2
    # baseline + 2 ablations = 3 config invocations × 1 query = 3 seen
    assert len(seen_configs) == 3
    # ablation configs should differ from baseline
    assert any("disabled_tools" in c for c in seen_configs)
    assert any(c.get("disable_memory") is True for c in seen_configs)


@pytest.mark.asyncio
async def test_ablation_report_writes_summary_md(tmp_path):
    qs = QuerySet(meta=QuerySetMeta(name="t"), queries=[
        Query(id="q1", text="qa", jurisdiction="CN", cause="c", source="s"),
    ])

    async def factory(config):
        async def runner(q):
            run_dir = tmp_path / "runs" / f"r-{q.id}-{id(config)}"
            _fake_events(run_dir)
            return {"run_id": run_dir.name, "status": "ok",
                    "lawyer_output": {}, "run_dir": run_dir}
        return runner

    ar = AblationRunner(query_set=qs, runs_root=tmp_path,
                        query_runner_factory=factory, run_group_base="ab")
    report = await ar.run(ablations=[DisableMemory()])
    summary_path = report.group_dir / "ablation_summary.md"
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "baseline" in md.lower()
    assert "disable_memory" in md
```

- [ ] **Step 2: Implement**

```python
"""AblationRunner (Phase 5d §7.9)."""
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Awaitable, Callable
from pydantic import BaseModel, Field

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.runner import ExperimentRunner, RunGroup
from multi_agent.eval.ablations import Ablation, apply_ablation


QueryRunnerFactory = Callable[[dict], Awaitable]


class AblationReport(BaseModel):
    group_base: str
    group_dir: Path
    baseline: RunGroup
    ablations: list[tuple[str, RunGroup]] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def n_ablations(self) -> int:
        return len(self.ablations)


class AblationRunner:
    def __init__(
        self,
        *,
        query_set: QuerySet,
        runs_root: Path,
        query_runner_factory: QueryRunnerFactory,
        run_group_base: str,
        parallelism: int = 1,
    ):
        self.query_set = query_set
        self.runs_root = Path(runs_root)
        self.factory = query_runner_factory
        self.base = run_group_base
        self.parallelism = parallelism

    async def _run_one(self, label: str, config: dict) -> RunGroup:
        runner = await self.factory(config)
        exp = ExperimentRunner(
            query_set=self.query_set,
            run_group_name=f"{self.base}__{label}",
            runs_root=self.runs_root,
            query_runner=runner,
            parallelism=self.parallelism,
        )
        return await exp.run()

    async def run(self, *, ablations: list[Ablation]) -> AblationReport:
        base_cfg: dict = {"label": "baseline"}
        baseline_group = await self._run_one("baseline", base_cfg)

        per_ablation: list[tuple[str, RunGroup]] = []
        for ab in ablations:
            cfg: dict = {"label": ab.name}
            apply_ablation(cfg, ab)
            grp = await self._run_one(ab.name.replace(":", "_").replace("→", "to"), cfg)
            per_ablation.append((ab.name, grp))

        group_dir = self.runs_root / "run_groups" / self.base
        group_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary(group_dir, baseline_group, per_ablation)
        return AblationReport(
            group_base=self.base, group_dir=group_dir,
            baseline=baseline_group, ablations=per_ablation,
        )

    def _write_summary(self, group_dir: Path, baseline: RunGroup,
                       ablations: list[tuple[str, RunGroup]]) -> None:
        import json, statistics
        def stats(g: RunGroup) -> dict:
            rows = [json.loads(l) for l in (g.group_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            ok = [r for r in rows if r.get("status") == "ok"]
            lats = [r["metrics"]["total_latency_ms"] for r in ok if "metrics" in r]
            return {
                "n": len(rows), "ok": len(ok),
                "p50_lat": int(statistics.median(lats)) if lats else 0,
                "in_tok": sum(r["metrics"].get("total_input_tokens", 0) for r in ok),
                "out_tok": sum(r["metrics"].get("total_output_tokens", 0) for r in ok),
                "cite_hits": sum(1 for r in ok if (r.get("citation_judge") or {}).get("hit")),
            }
        b = stats(baseline)
        lines = [
            f"# Ablation report `{self.base}`\n",
            f"## baseline ({baseline.name})",
            f"- queries: {b['n']} (ok={b['ok']})",
            f"- p50 latency: {b['p50_lat']}ms",
            f"- tokens: in={b['in_tok']} out={b['out_tok']}",
            f"- citation hits: {b['cite_hits']}",
            "",
            "| Ablation | Δp50 lat ms | Δin tok | Δout tok | Δcite hits |",
            "|---|---|---|---|---|",
        ]
        for name, grp in ablations:
            s = stats(grp)
            lines.append(
                f"| {name} | {s['p50_lat']-b['p50_lat']:+d} | "
                f"{s['in_tok']-b['in_tok']:+d} | {s['out_tok']-b['out_tok']:+d} | "
                f"{s['cite_hits']-b['cite_hits']:+d} |"
            )
        (group_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 3: Verify pass + unit suite** → 214-215 passed.

- [ ] **Step 4: Commit + tag**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/eval/ablation_runner.py experiments/multi_agent/tests/unit/test_ablation_runner.py
git commit -m "phase5d(eval): AblationRunner + AblationReport"
git tag -a phase5d-ablation -m "Phase 5d: AblationRunner (baseline + N ablations × QuerySet)"
git tag -l "phase*"
```

---

## Acceptance Criteria

Phase 5d complete when:

1. Full pytest passes (~214-215 unit tests)
2. 4 ablation types each write expected keys into config
3. `AblationRunner.run()` runs baseline + N ablations, produces N+1 RunGroups
4. `ablation_summary.md` shows per-ablation deltas vs baseline
5. Tag `phase5d-ablation` exists

## Out of Scope (Phase 5e+)

- LatencyProfiler (SpanTiming derivation)
- Trace Viewer (Streamlit)
- Real Qwen E2E ablation (would need a query_runner factory that consults `disabled_tools` to actually skip statute_search — caller's responsibility, beyond this phase's plumbing)
- Statistical significance (need N≥30 per cell)
