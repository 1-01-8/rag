"""AblationRunner (Phase 5d §7.9)."""
from __future__ import annotations

import asyncio
import json
import statistics
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
    """Run baseline + N ablations × QuerySet via ExperimentRunner.

    For each ablation, a fresh config dict is built by calling
    ``apply_ablation(cfg, ab)`` and then passed to ``query_runner_factory``
    so the caller can honour it (e.g. skip a tool, swap a model).

    Results land in ``runs_root/run_groups/<base>__<label>/`` and a combined
    ``ablation_summary.md`` is written to ``runs_root/run_groups/<base>/``.
    """

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    @staticmethod
    def _safe_label(name: str) -> str:
        """Convert an ablation name into a filesystem-safe directory label."""
        return name.replace(":", "_").replace("→", "to")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, *, ablations: list[Ablation]) -> AblationReport:
        """Execute baseline + all ablations sequentially, return AblationReport."""
        # --- Baseline ---
        base_cfg: dict = {"label": "baseline"}
        baseline_group = await self._run_one("baseline", base_cfg)

        # --- Per-ablation ---
        per_ablation: list[tuple[str, RunGroup]] = []
        for ab in ablations:
            cfg: dict = {"label": ab.name}
            apply_ablation(cfg, ab)
            grp = await self._run_one(self._safe_label(ab.name), cfg)
            per_ablation.append((ab.name, grp))

        # --- Summary dir (one level up from individual group dirs) ---
        group_dir = self.runs_root / "run_groups" / self.base
        group_dir.mkdir(parents=True, exist_ok=True)
        self._write_summary(group_dir, baseline_group, per_ablation)

        return AblationReport(
            group_base=self.base,
            group_dir=group_dir,
            baseline=baseline_group,
            ablations=per_ablation,
        )

    # ------------------------------------------------------------------
    # Summary writer
    # ------------------------------------------------------------------

    def _group_stats(self, g: RunGroup) -> dict:
        """Compute aggregate stats from a RunGroup's results.jsonl."""
        results_path = g.group_dir / "results.jsonl"
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ok = [r for r in rows if r.get("status") == "ok"]
        lats = [
            r["metrics"]["total_latency_ms"]
            for r in ok
            if "metrics" in r and "total_latency_ms" in r["metrics"]
        ]
        return {
            "n": len(rows),
            "ok": len(ok),
            "p50_lat": int(statistics.median(lats)) if lats else 0,
            "in_tok": sum(
                r["metrics"].get("total_input_tokens", 0) for r in ok if "metrics" in r
            ),
            "out_tok": sum(
                r["metrics"].get("total_output_tokens", 0) for r in ok if "metrics" in r
            ),
            "cite_hits": sum(
                1
                for r in ok
                if (r.get("citation_judge") or {}).get("hit")
            ),
        }

    def _write_summary(
        self,
        group_dir: Path,
        baseline: RunGroup,
        ablations: list[tuple[str, RunGroup]],
    ) -> None:
        b = self._group_stats(baseline)
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
            s = self._group_stats(grp)
            lines.append(
                f"| {name} | {s['p50_lat'] - b['p50_lat']:+d} | "
                f"{s['in_tok'] - b['in_tok']:+d} | {s['out_tok'] - b['out_tok']:+d} | "
                f"{s['cite_hits'] - b['cite_hits']:+d} |"
            )
        (group_dir / "ablation_summary.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )
