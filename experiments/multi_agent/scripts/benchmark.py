#!/usr/bin/env python
"""Phase 5w benchmark: synthetic_seed_v1 through Lawyer+Supervisor stack.

Runs all substantive queries from synthetic_seed_v1 against real Qwen with
a composite statute index, then prints:
  - per-query latency, tokens, cost_usd, citation hit, supervisor verdict
  - aggregate p50 / p95 / total tokens / hit rate
  - latency flame for the slowest query

Standalone (not a pytest test) so the user can run it on demand:
    python scripts/benchmark.py
"""
from __future__ import annotations
import asyncio
import json
import statistics
import sys
import uuid
from pathlib import Path

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.metrics import derive_run_metrics
from multi_agent.eval.judges.citation_accuracy import CitationAccuracyJudge
from multi_agent.eval.latency import LatencyProfiler
from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.orchestration.supervised import run_with_supervisor


# Composite seed index — same articles that synthetic_seed_v1 expects.
_CHUNKS = [
    ("民法典-510", "民法典", "510",
     "当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
    ("民法典-563", "民法典", "563",
     "有下列情形之一的,当事人可以解除合同:法律规定的其他情形。"),
    ("民法典-577", "民法典", "577",
     "当事人一方不履行合同义务的,应当承担违约责任。"),
    ("民法典-584", "民法典", "584",
     "造成对方损失的,损失赔偿额应当相当于因违约所造成的损失。"),
    ("民法典-703", "民法典", "703",
     "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
    ("民法典-707", "民法典", "707",
     "租赁期限六个月以上的,应当采用书面形式。"),
    ("民法典-1165", "民法典", "1165",
     "行为人因过错侵害他人民事权益造成损害的,应当承担侵权责任。"),
    ("民法典-1184", "民法典", "1184",
     "侵害他人财产的,财产损失按照损失发生时的市场价格或者其他合理方式计算。"),
    ("道路交通安全法-76", "道路交通安全法", "76",
     "机动车之间发生交通事故的,由有过错的一方承担赔偿责任。"),
]


async def main() -> int:
    import argparse, os
    p = argparse.ArgumentParser(description="Benchmark synthetic_seed_v1 × Lawyer+Supervisor")
    p.add_argument("--provider",
                   choices=["local", "deepseek", "siliconflow"], default="local")
    p.add_argument("--model", default=None,
                   help="默认: local=qwen3.5-9b / deepseek=deepseek-chat")
    args = p.parse_args()

    runs_root = Path("runs") / f"bench_{uuid.uuid4().hex[:6]}"
    runs_root.mkdir(parents=True, exist_ok=True)

    # 1. Build composite index
    coll = f"bench_{uuid.uuid4().hex[:8]}"
    sparse_path = runs_root / "sparse.json"
    docs = [Document(
        law_name="composite", law_short="composite", source_path="composite",
        chunks=[Chunk(doc_id=did, law_name="composite", law_short=ls,
                     article_no=an, text=t) for (did, ls, an, t) in _CHUNKS],
    )]
    print(f"Building index {coll}...")
    build_index(documents=docs, collection_name=coll,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())

    # 2. Load queryset (skip smalltalk q005)
    qs_path = Path(__file__).parents[1] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(qs_path)
    qs.queries = [q for q in qs.queries if "smalltalk" not in q.tags]
    print(f"Running {len(qs.queries)} queries through Lawyer + Supervisor...\n")

    if args.provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("❌ --provider deepseek 但 DEEPSEEK_API_KEY 未设置")
            return 1
        provider = OpenAICompatibleProvider(
            base_url="https://api.deepseek.com/v1", api_key=api_key,
        )
        model_name = args.model or "deepseek-chat"
        print(f"Provider: DeepSeek  model={model_name}")
    elif args.provider == "siliconflow":
        api_key = os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            print("❌ --provider siliconflow 但 SILICONFLOW_API_KEY 未设置")
            return 1
        provider = OpenAICompatibleProvider(
            base_url="https://api.siliconflow.cn/v1", api_key=api_key,
        )
        model_name = args.model or "deepseek-ai/DeepSeek-V4-Flash"
        print(f"Provider: SiliconFlow  model={model_name}")
    else:
        provider = OpenAICompatibleProvider()
        model_name = args.model or "qwen3.5-9b"
        print(f"Provider: local vLLM  model={model_name}")
    statute_search = StatuteSearchTool(
        collection_name=coll, sparse_artifact_path=sparse_path,
    )
    judge = CitationAccuracyJudge()

    rows: list[dict] = []
    try:
        for i, q in enumerate(qs.queries, 1):
            print(f"[{i}/{len(qs.queries)}] {q.id} — {q.text[:50]}...")
            result = await run_with_supervisor(
                query=q.text,
                lawyer_factory=lambda p, r: LawyerAgent(
                    name="lawyer", role="advisor", provider=p, recorder=r,
                    tools=[statute_search],
                    model=model_name, specialty="民事",
                    max_steps=6, max_tool_calls=8, max_pre_tool_rejections=2,
                ),
                supervisor_factory=lambda p, r: SupervisorAgent(
                    name="supervisor", role="qa", provider=p, recorder=r,
                    model=model_name, max_steps=3, max_pre_tool_rejections=5,
                ),
                lawyer_provider=provider, supervisor_provider=provider,
                runs_root=runs_root,
            )
            lawyer_run_dir = runs_root / result["lawyer_run_id"]
            m = derive_run_metrics(lawyer_run_dir)
            try:
                lo = json.loads(result["lawyer_result"].get("final_answer") or "{}")
            except Exception:
                lo = {}
            cj = judge.judge(q, lo)
            verdict = result["supervisor_verdict"]["verdict"]
            rows.append({
                "query_id": q.id, "text": q.text,
                "lawyer_run": result["lawyer_run_id"],
                "metrics": m.model_dump(),
                "citation_hit": cj.hit, "citation_matched": cj.matched,
                "supervisor_verdict": verdict,
            })
            print(f"    lat={m.total_latency_ms}ms in={m.total_input_tokens} "
                  f"out={m.total_output_tokens} cost=${m.cost_usd:.4f} "
                  f"cite={'✓' if cj.hit else '✗'} sup={verdict}")
    finally:
        drop_collection(coll)

    # 3. Aggregate
    lats = [r["metrics"]["total_latency_ms"] for r in rows]
    in_tok = sum(r["metrics"]["total_input_tokens"] for r in rows)
    out_tok = sum(r["metrics"]["total_output_tokens"] for r in rows)
    total_cost = sum(r["metrics"]["cost_usd"] for r in rows)
    hits = sum(1 for r in rows if r["citation_hit"])

    p50 = int(statistics.median(lats)) if lats else 0
    p95 = max(lats) if len(lats) < 20 else int(statistics.quantiles(lats, n=20)[18])

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Queries:        {len(rows)}")
    print(f"  Latency p50:    {p50}ms")
    print(f"  Latency p95:    {p95}ms (= max for N<20)")
    print(f"  Latency total:  {sum(lats)}ms")
    print(f"  Input tokens:   {in_tok}")
    print(f"  Output tokens:  {out_tok}")
    print(f"  Cost USD:       ${total_cost:.4f}  (local Qwen = $0)")
    print(f"  Citation hits:  {hits}/{len(rows)} ({100*hits/max(len(rows),1):.0f}%)")
    sup_counts: dict[str, int] = {}
    for r in rows:
        sup_counts[r["supervisor_verdict"]] = sup_counts.get(r["supervisor_verdict"], 0) + 1
    print(f"  Supervisor:     {sup_counts}")
    print("=" * 70)

    # 4. Flame for slowest
    if rows:
        slowest = max(rows, key=lambda r: r["metrics"]["total_latency_ms"])
        print(f"\nLatency flame — slowest query ({slowest['query_id']}):")
        profile = LatencyProfiler().profile(runs_root / slowest["lawyer_run"])
        print(LatencyProfiler.render_flame(profile))

    # 5. Persist
    bench_path = runs_root / "benchmark.json"
    bench_path.write_text(json.dumps({
        "queries": rows,
        "aggregate": {
            "n": len(rows), "p50_ms": p50, "p95_ms": p95,
            "in_tokens": in_tok, "out_tokens": out_tok,
            "cost_usd": total_cost, "citation_hits": hits,
            "supervisor_counts": sup_counts,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results: {bench_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
