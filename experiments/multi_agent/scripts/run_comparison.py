#!/usr/bin/env python
"""Run two ExperimentRunner invocations + Comparator on the same QuerySet.

Typical usage:
    ANTHROPIC_API_KEY=... python scripts/run_comparison.py \\
        --queryset evals/querysets/synthetic_seed_v1.yaml \\
        --statutes-collection ma_statutes \\
        --statutes-sparse data/indexes/statutes_sparse.json \\
        --runs-root runs \\
        --group-a-name qwen_baseline --profile-a all-local \\
        --group-b-name claude_baseline --profile-b all-claude \\
        --judges
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.report import render_summary_md
from multi_agent.eval.comparator import Comparator
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.runner import run_query
from multi_agent.providers.profile import build_provider_for


async def _run_one_group(
    *,
    profile_name: str,
    group_name: str,
    qs: QuerySet,
    statute_search: StatuteSearchTool,
    runs_root: Path,
    parallelism: int,
    judges: list | None,
) -> "RunGroup":  # noqa: F821
    # build_provider_for returns (provider, model) for the given agent role
    provider, model = build_provider_for("lawyer", profile_name=profile_name)

    async def query_runner(q):
        result = await run_query(
            query=q.text,
            agent_factory=lambda p, r: LawyerAgent(
                name="lawyer",
                role="advisor",
                provider=p,
                recorder=r,
                tools=[statute_search],
                model=model,
                specialty="民事",
                max_steps=6,
                max_tool_calls=8,
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

    return await ExperimentRunner(
        query_set=qs,
        run_group_name=group_name,
        runs_root=runs_root,
        query_runner=query_runner,
        parallelism=parallelism,
        judges=judges,
    ).run()


async def main_async(args) -> int:
    qs = QuerySet.from_yaml(args.queryset)
    if args.max_queries:
        qs.queries = qs.queries[: args.max_queries]

    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "comparison_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    statute_search = StatuteSearchTool(
        collection_name=args.statutes_collection,
        sparse_artifact_path=Path(args.statutes_sparse),
    )

    judges = None
    if args.judges:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("--judges requested but ANTHROPIC_API_KEY not set; skipping judges")
        else:
            from multi_agent.eval.judges.groundedness import GroundednessJudge
            from multi_agent.eval.judges.helpfulness import HelpfulnessJudge
            from multi_agent.providers.anthropic import AnthropicProvider
            ap = AnthropicProvider()
            judges = [
                GroundednessJudge(provider=ap, model="claude-opus-4-7"),
                HelpfulnessJudge(provider=ap, model="claude-opus-4-7"),
            ]

    group_a = await _run_one_group(
        profile_name=args.profile_a,
        group_name=args.group_a_name,
        qs=qs,
        statute_search=statute_search,
        runs_root=runs_root,
        parallelism=args.parallelism,
        judges=judges,
    )
    render_summary_md(group_a.group_dir)

    group_b = await _run_one_group(
        profile_name=args.profile_b,
        group_name=args.group_b_name,
        qs=qs,
        statute_search=statute_search,
        runs_root=runs_root,
        parallelism=args.parallelism,
        judges=judges,
    )
    render_summary_md(group_b.group_dir)

    comparator = Comparator()
    report = comparator.compare(
        group_a_dir=group_a.group_dir,
        group_b_dir=group_b.group_dir,
    )
    out_path = comparator.render_md(report=report, out_dir=out_dir)

    print(f"Group A: {group_a.group_dir}")
    print(f"Group B: {group_b.group_dir}")
    print(f"Comparison: {out_path}")
    print(f"Winners: A={report.winners_a} B={report.winners_b} ties={report.ties}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run a 2-profile comparison through ExperimentRunner + Comparator"
    )
    p.add_argument("--queryset", type=Path, required=True,
                   help="Path to QuerySet YAML")
    p.add_argument("--runs-root", type=Path, default=Path("runs"),
                   help="Root directory for individual run output")
    p.add_argument("--group-a-name", required=True,
                   help="Label for the first run group (profile A)")
    p.add_argument("--group-b-name", required=True,
                   help="Label for the second run group (profile B)")
    p.add_argument("--profile-a", required=True,
                   help="ProviderProfile name for group A (e.g. all-local)")
    p.add_argument("--profile-b", required=True,
                   help="ProviderProfile name for group B (e.g. all-claude)")
    p.add_argument("--statutes-collection", required=True,
                   help="Qdrant collection name for statute search")
    p.add_argument("--statutes-sparse", required=True,
                   help="Path to sparse JSON artifact for statute search")
    p.add_argument("--max-queries", type=int, default=None,
                   help="Limit to first N queries (smoke testing)")
    p.add_argument("--parallelism", type=int, default=1,
                   help="Max concurrent queries per group")
    p.add_argument("--judges", action="store_true",
                   help="Enable Claude Opus LLM judges (requires ANTHROPIC_API_KEY)")
    p.add_argument("--out-dir", default=None,
                   help="Directory for comparison report output (default: runs/comparison_reports)")
    return asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
