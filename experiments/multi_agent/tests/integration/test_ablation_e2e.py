"""Phase 5g real-Qwen E2E: baseline vs DisableTool ablation."""
from __future__ import annotations
import json
import uuid
from pathlib import Path

import httpx
import pytest

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.ablations import DisableTool
from multi_agent.eval.ablation_runner import AblationRunner
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.schemas.document import Document, Chunk
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
    """Composite index, copied from test_eval_e2e.py fixture pattern."""
    name = f"abl_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    chunks_data = [
        ("民法典-510", "民法典", "510",
         "当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
        ("民法典-563", "民法典", "563",
         "有下列情形之一的,当事人可以解除合同:法律规定的其他情形。"),
        ("民法典-703", "民法典", "703",
         "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ("民法典-707", "民法典", "707",
         "租赁期限六个月以上的,应当采用书面形式。"),
        ("民法典-1165", "民法典", "1165",
         "行为人因过错侵害他人民事权益造成损害的,应当承担侵权责任。"),
        ("民法典-1184", "民法典", "1184",
         "侵害他人财产的,财产损失按照损失发生时的市场价格或者其他合理方式计算。"),
        ("民法典-577", "民法典", "577",
         "当事人一方不履行合同义务的,应当承担违约责任。"),
        ("民法典-584", "民法典", "584",
         "当事人一方不履行合同义务造成对方损失的,损失赔偿额应当相当于因违约所造成的损失。"),
        ("道路交通安全法-76", "道路交通安全法", "76",
         "机动车之间发生交通事故的,由有过错的一方承担赔偿责任。"),
    ]
    chunks = [
        Chunk(doc_id=did, law_name="composite", law_short=ls, article_no=an, text=t)
        for (did, ls, an, t) in chunks_data
    ]
    doc = Document(law_name="composite", law_short="composite", source_path="composite", chunks=chunks)
    build_index(documents=[doc], collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_ablation_disable_statute_search_e2e(real_corpus_index, tmp_path):
    """Real Qwen: baseline (with statute_search) vs ablation (no tools).
    Expect baseline citation hits >= ablation citation hits (or at least no errors)."""
    qs_path = Path(__file__).parents[2] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(qs_path)
    # Restrict to 2 substantive queries to keep cost/time bounded — 2 queries × 2 runs = 4 Qwen calls
    qs.queries = [q for q in qs.queries if q.id in ("q001", "q002")]
    assert len(qs.queries) == 2

    provider = OpenAICompatibleProvider()
    statute_search = StatuteSearchTool(
        collection_name=real_corpus_index["collection"],
        sparse_artifact_path=real_corpus_index["sparse_path"],
    )

    async def factory(config: dict):
        disabled = config.get("disabled_tools", set())
        tools = [] if "statute_search" in disabled else [statute_search]

        async def runner(q):
            result = await run_query(
                query=q.text,
                agent_factory=lambda p, r: LawyerAgent(
                    name="lawyer", role="advisor",
                    provider=p, recorder=r,
                    tools=tools,
                    model="qwen3.5-9b", specialty="民事",
                    max_steps=5, max_tool_calls=5,
                    max_pre_tool_rejections=2 if tools else 0,
                ),
                provider=provider,
                runs_root=tmp_path / "runs",
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
                "run_dir": tmp_path / "runs" / result["run_id"],
            }

        return runner

    ar = AblationRunner(
        query_set=qs, runs_root=tmp_path,
        query_runner_factory=factory, run_group_base="ablate-statute-search",
        parallelism=1,
    )
    report = await ar.run(ablations=[DisableTool(tool="statute_search")])

    assert report.n_ablations == 1
    summary_path = report.group_dir / "ablation_summary.md"
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "disable_tool:statute_search" in md
    assert "baseline" in md.lower()

    # All runs should complete without raised exceptions
    base_rows = [
        json.loads(l)
        for l in (report.baseline.group_dir / "results.jsonl").read_text().splitlines()
        if l.strip()
    ]
    abl_rows = [
        json.loads(l)
        for l in (report.ablations[0][1].group_dir / "results.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert all(r["status"] == "ok" for r in base_rows), [r for r in base_rows if r["status"] != "ok"]
    assert all(r["status"] == "ok" for r in abl_rows), [r for r in abl_rows if r["status"] != "ok"]

    base_hits = sum(1 for r in base_rows if r.get("citation_judge", {}).get("hit"))
    abl_hits = sum(1 for r in abl_rows if r.get("citation_judge", {}).get("hit"))
    # Baseline should hit at least once; ablation usually zero (no retrieval, no real citations).
    # Hedge: assert baseline >= ablation. (Equal is fine — the experiment found a null result.)
    assert base_hits >= abl_hits, (
        f"baseline_hits={base_hits} < ablation_hits={abl_hits} — unexpected"
    )
    # Print for visibility — even if assertion holds, the gap is the interesting signal
    print(f"\n=== Ablation result: baseline={base_hits}/{len(base_rows)} hits, "
          f"DisableTool(statute_search)={abl_hits}/{len(abl_rows)} hits ===\n")
    print(md)
