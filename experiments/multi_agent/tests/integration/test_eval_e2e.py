"""Phase 5b E2E: synthetic_seed_v1 through real Qwen Lawyer pipeline.

Runs 4 substantive queries (smalltalk q005 skipped) through LawyerAgent +
StatuteSearchTool backed by real Qwen vLLM, then asserts:
  - zero errors
  - at least 1 citation hit across 4 queries
  - summary.md rendered
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest

from multi_agent.eval.queryset import QuerySet
from multi_agent.eval.runner import ExperimentRunner
from multi_agent.eval.report import render_summary_md
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
    """Build a small composite index covering 民法典 + 道路交通安全法 articles
    referenced by synthetic_seed_v1 expected citations."""
    name = f"eval_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"

    # Each tuple: (doc_id, law_short, article_no, text)
    chunks_data = [
        ("民法典-510", "民法典", "510",
         "当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
        ("民法典-563", "民法典", "563",
         "有下列情形之一的,当事人可以解除合同:"
         "（一）因不可抗力致使不能实现合同目的；"
         "（二）在履行期限届满前,当事人一方明确表示或者以自己的行为表明不履行主要债务；"
         "（三）当事人一方迟延履行主要债务,经催告后在合理期限内仍未履行；"
         "（四）当事人一方迟延履行债务或者有其他违约行为致使不能实现合同目的；"
         "（五）法律规定的其他情形。"),
        ("民法典-577", "民法典", "577",
         "当事人一方不履行合同义务或者履行合同义务不符合约定的,"
         "应当承担继续履行、采取补救措施或者赔偿损失等违约责任。"),
        ("民法典-584", "民法典", "584",
         "当事人一方不履行合同义务或者履行合同义务不符合约定,"
         "造成对方损失的,损失赔偿额应当相当于因违约所造成的损失,"
         "包括合同履行后可以获得的利益。"),
        ("民法典-703", "民法典", "703",
         "租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ("民法典-707", "民法典", "707",
         "租赁期限六个月以上的,应当采用书面形式。"
         "当事人未采用书面形式,无法确定租赁期限的,视为不定期租赁。"),
        ("民法典-1165", "民法典", "1165",
         "行为人因过错侵害他人民事权益造成损害的,应当承担侵权责任。"
         "依照法律规定推定行为人有过错,其不能证明自己没有过错的,应当承担侵权责任。"),
        ("民法典-1184", "民法典", "1184",
         "侵害他人财产的,财产损失按照损失发生时的市场价格或者其他合理方式计算。"),
        ("道路交通安全法-76", "道路交通安全法", "76",
         "机动车发生交通事故造成人身伤亡、财产损失的,"
         "由保险公司在机动车第三者责任强制保险责任限额范围内予以赔偿;"
         "不足的部分,机动车之间发生交通事故的,由有过错的一方承担赔偿责任;"
         "双方都有过错的,按照各自过错的比例分担责任。"),
    ]

    chunks = [
        Chunk(
            doc_id=doc_id,
            law_name=f"中华人民共和国{law_short}" if law_short != "道路交通安全法" else "中华人民共和国道路交通安全法",
            law_short=law_short,
            article_no=article_no,
            text=text,
        )
        for (doc_id, law_short, article_no, text) in chunks_data
    ]

    doc = Document(
        law_name="composite_seed",
        law_short="composite",
        source_path="composite",
        chunks=chunks,
    )

    build_index(
        documents=[doc],
        collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_synthetic_seed_v1_through_lawyer(real_corpus_index, tmp_path):
    """Run 4 substantive queries from synthetic_seed_v1 through LawyerAgent.

    Assertions:
    - All 4 runs complete with status == ok
    - At least 1 citation hit (CitationAccuracyJudge finds a match)
    - summary.md is rendered
    """
    qs_path = Path(__file__).parents[2] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(qs_path)
    # Skip smalltalk query q005 — Lawyer is not the right entry point for it
    qs.queries = [q for q in qs.queries if "smalltalk" not in q.tags]
    assert len(qs.queries) == 4, f"Expected 4 substantive queries, got {len(qs.queries)}"

    provider = OpenAICompatibleProvider()
    statute_search = StatuteSearchTool(
        collection_name=real_corpus_index["collection"],
        sparse_artifact_path=real_corpus_index["sparse_path"],
    )
    runs_root = tmp_path / "runs"

    async def run_one(q):
        result = await run_query(
            query=q.text,
            agent_factory=lambda p, r: LawyerAgent(
                name="lawyer",
                role="advisor",
                provider=p,
                recorder=r,
                tools=[statute_search],
                model="qwen3.5-9b",
                specialty="民事",
                max_steps=8,
                max_tool_calls=10,
            ),
            provider=provider,
            runs_root=runs_root,
            config={"phase": "5b", "query_id": q.id},
        )
        # run_dir is runs_root / run_id (run_query does not return run_dir key)
        run_dir = runs_root / result["run_id"]
        # Parse lawyer output for citation judge
        try:
            lo = json.loads(result.get("final_answer") or "{}")
        except Exception:
            lo = {}
        return {
            "run_id": result["run_id"],
            "status": result.get("status", "ok"),
            "lawyer_output": lo,
            "run_dir": run_dir,
        }

    runner = ExperimentRunner(
        query_set=qs,
        run_group_name="phase5b-seed-run",
        runs_root=tmp_path,
        query_runner=run_one,
        parallelism=2,
    )
    group = await runner.run()

    # Render summary
    summary = render_summary_md(group.group_dir)
    assert summary.exists(), "summary.md was not rendered"

    # Load results
    results_path = group.group_dir / "results.jsonl"
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 4, f"Expected 4 result rows, got {len(rows)}"

    # All runs must complete successfully
    failed = [r for r in rows if r.get("status") != "ok"]
    assert not failed, (
        f"{len(failed)} run(s) failed:\n"
        + "\n".join(
            f"  query_id={r.get('query_id')} error={r.get('error','')}" for r in failed
        )
    )

    # At least one query must hit a citation from the expected set
    hits = sum(1 for r in rows if r.get("citation_judge", {}).get("hit"))
    assert hits >= 1, (
        f"No citation hits across {len(rows)} queries. "
        f"Citation judge results: "
        + str([{"id": r["query_id"], "cj": r.get("citation_judge")} for r in rows])
    )
