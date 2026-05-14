import uuid
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.all_sources_search import (
    AllSourcesSearchTool, AllSourcesArgs, _rrf_merge,
)
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def both_indexes(tmp_path_factory):
    statutes_name = f"test_s_{uuid.uuid4().hex[:8]}"
    cases_name = f"test_c_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    s_sparse = tmp / "s_sparse.json"
    c_sparse = tmp / "c_sparse.json"

    encoder = DenseEncoder()

    # Build statutes
    stat_docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照交易习惯确定。"),
        ],
    )]
    build_index(documents=stat_docs, collection_name=statutes_name,
                sparse_artifact_path=s_sparse, dense_encoder=encoder)

    # Build cases
    cases_jsonl = tmp / "cases.jsonl"
    with cases_jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租 30%,合不合法?",
            answer="一般不合法,可拒绝。",
            extracted_cite_ids=["民法典-510"],
        ).model_dump_json() + "\n")
    from scripts.build_cases_index import build_cases_index
    build_cases_index(jsonl_path=cases_jsonl, collection_name=cases_name,
                     sparse_artifact_path=c_sparse, dense_encoder=encoder)

    yield {
        "statutes": statutes_name, "statutes_sparse": s_sparse,
        "cases": cases_name, "cases_sparse": c_sparse,
    }
    drop_collection(statutes_name)
    drop_collection(cases_name)


@pytest.mark.asyncio
async def test_all_sources_returns_mixed_evidence(both_indexes, tmp_run_dir):
    tool = AllSourcesSearchTool(
        statutes_collection=both_indexes["statutes"],
        statutes_sparse=both_indexes["statutes_sparse"],
        cases_collection=both_indexes["cases"],
        cases_sparse=both_indexes["cases_sparse"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        AllSourcesArgs(query="房东涨租 民法典 第510条", k=5),
        rec,
    )
    rec.close()
    assert result.error is None
    evidences = result.payload["evidences"]
    retrievers = {Evidence.model_validate(e).retriever for e in evidences}
    assert "hybrid" in retrievers or "case" in retrievers
    assert len(evidences) >= 1


def test_rrf_merge_namespaces_by_retriever():
    """Two evidences with the same doc_id but different retrievers must NOT merge."""
    # Statute side
    stat_ev = Evidence(
        doc_id="train_001",          # collision-shaped id
        law_name="民法典", law_short="民法典", article_no="510",
        text="statute text", score=0.5, retriever="hybrid",
    )
    # Case side with same doc_id
    case_ev = Evidence(
        doc_id="train_001",          # SAME doc_id
        law_name="(case)", law_short="", article_no="train_001",
        text="case text", score=0.5, retriever="case",
    )
    fused = _rrf_merge([[stat_ev], [case_ev]], top_k=5)
    # Both should appear independently (2 fused evidences, not 1)
    assert len(fused) == 2
    retrievers = {e.retriever for e in fused}
    assert "hybrid" in retrievers
    assert "case" in retrievers
