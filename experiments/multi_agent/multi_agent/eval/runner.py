"""ExperimentRunner + RunGroup (Phase 5b §7.4-7.5).

Walks a QuerySet, calls a user-supplied async query_runner(query) → dict,
writes run_groups/<group>/results.jsonl with per-row metrics + citation_judge.
"""
from __future__ import annotations

import asyncio
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

from pydantic import BaseModel

from multi_agent.eval.queryset import Query, QuerySet
from multi_agent.eval.metrics import derive_run_metrics
from multi_agent.eval.judges.citation_accuracy import CitationAccuracyJudge


class RunGroup(BaseModel):
    """Metadata about a completed experiment run group."""

    name: str
    group_dir: Path
    query_set_name: str

    model_config = {"arbitrary_types_allowed": True}


#: Type alias for the async callable supplied by callers.
QueryRunner = Callable[[Query], Awaitable[dict]]


class ExperimentRunner:
    """Walk a QuerySet, execute each query via query_runner, collect results.

    Args:
        query_set: The QuerySet to evaluate.
        run_group_name: A label for this run group (used as directory name).
        runs_root: Root directory where per-run directories live and where the
            ``run_groups/`` sub-directory will be created.
        query_runner: Async callable ``(query: Query) → dict`` with keys:
            ``run_id``, ``status``, ``lawyer_output`` (dict), ``run_dir`` (Path).
        parallelism: Max concurrent queries (semaphore). Default 1 (serial).
        group_root: Override the directory for run_groups. Defaults to
            ``runs_root / "run_groups"``.
    """

    def __init__(
        self,
        *,
        query_set: QuerySet,
        run_group_name: str,
        runs_root: Path,
        query_runner: QueryRunner,
        parallelism: int = 1,
        group_root: Path | None = None,
    ) -> None:
        self.query_set = query_set
        self.run_group_name = run_group_name
        self.runs_root = Path(runs_root)
        self.query_runner = query_runner
        self.parallelism = parallelism
        self.group_root = (
            Path(group_root) if group_root else self.runs_root / "run_groups"
        )
        self.judge = CitationAccuracyJudge()

    async def run(self) -> RunGroup:
        """Execute all queries and write results.jsonl.  Returns a RunGroup."""
        group_dir = self.group_root / self.run_group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        # Write group metadata
        (group_dir / "group_meta.yaml").write_text(
            f"name: {self.run_group_name}\n"
            f"query_set: {self.query_set.meta.name}\n"
            f"created: {datetime.now().isoformat()}\n",
            encoding="utf-8",
        )

        results_path = group_dir / "results.jsonl"
        sem = asyncio.Semaphore(self.parallelism)

        async def _one(q: Query) -> dict:
            async with sem:
                try:
                    out = await self.query_runner(q)
                    metrics = derive_run_metrics(out["run_dir"]).model_dump()
                    citation_result = self.judge.judge(
                        q, out.get("lawyer_output") or {}
                    ).model_dump()
                    return {
                        "query_id": q.id,
                        "run_id": out["run_id"],
                        "status": out.get("status", "ok"),
                        "metrics": metrics,
                        "citation_judge": citation_result,
                    }
                except Exception as exc:  # noqa: BLE001
                    return {
                        "query_id": q.id,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }

        rows = await asyncio.gather(*[_one(q) for q in self.query_set.queries])

        with results_path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        return RunGroup(
            name=self.run_group_name,
            group_dir=group_dir,
            query_set_name=self.query_set.meta.name,
        )
