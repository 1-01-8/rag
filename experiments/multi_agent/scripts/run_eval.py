#!/usr/bin/env python
"""Run a QuerySet through Qwen Lawyer + StatuteSearch + write a RunGroup.

Usage:
    python scripts/run_eval.py --queryset evals/querysets/synthetic_seed_v1.yaml \\
        --statutes-collection ma_statutes \\
        --statutes-sparse data/indexes/statutes_sparse.json \\
        --runs-root runs \\
        --group-name baseline_$(date +%Y%m%d_%H%M)
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from pathlib import Path

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.report import render_summary_md
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.runner import run_query


async def main_async(args) -> int:
    qs = QuerySet.from_yaml(args.queryset)
    if args.max_queries:
        qs.queries = qs.queries[: args.max_queries]

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    provider = OpenAICompatibleProvider(
        base_url=args.qwen_base_url,  # None falls back to env / localhost:8000
    )
    statute_search = StatuteSearchTool(
        collection_name=args.statutes_collection,
        sparse_artifact_path=Path(args.statutes_sparse),
    )

    async def query_runner(q):
        result = await run_query(
            query=q.text,
            agent_factory=lambda p, r: LawyerAgent(
                name="lawyer",
                role="advisor",
                provider=p,
                recorder=r,
                tools=[statute_search],
                model=args.model,
                specialty=args.specialty,
                max_steps=args.max_steps,
                max_tool_calls=args.max_tool_calls,
                max_pre_tool_rejections=2,
            ),
            provider=provider,
            runs_root=runs_root,
            config={},
        )
        try:
            lo = json.loads(result.get("final_answer") or "{}")
        except Exception:
            lo = {}
        return {
            "run_id": result["run_id"],
            "status": result.get("status", "ok"),
            "lawyer_output": lo,
            "evidence_pool": result.get("evidence_pool") or [],
            "run_dir": runs_root / result["run_id"],
        }

    runner = ExperimentRunner(
        query_set=qs,
        run_group_name=args.group_name,
        runs_root=runs_root,
        query_runner=query_runner,
        parallelism=args.parallelism,
    )
    group = await runner.run()
    summary_path = render_summary_md(group.group_dir)
    print(f"RunGroup written to: {group.group_dir}")
    print(f"Summary: {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a QuerySet through Qwen Lawyer pipeline."
    )
    parser.add_argument(
        "--queryset", type=Path, required=True, help="Path to QuerySet YAML"
    )
    parser.add_argument(
        "--statutes-collection", required=True, help="Qdrant collection name"
    )
    parser.add_argument(
        "--statutes-sparse", required=True, help="Path to sparse JSON artifact"
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--group-name", required=True)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit to first N queries (for smoke testing)",
    )
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--specialty", default="民事")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument(
        "--qwen-base-url",
        default=None,
        help="vLLM base URL (default: env OPENAI_COMPAT_BASE_URL or http://localhost:8000/v1)",
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
