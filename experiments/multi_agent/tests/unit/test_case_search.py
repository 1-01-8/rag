import json
import uuid
import pytest
from pathlib import Path

from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.case_search import CaseSearchTool, CaseSearchArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def case_index(tmp_path_factory):
    name = f"test_case_search_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    jsonl = tmp / "cases.jsonl"
    sparse_path = tmp / "sparse.json"
    with jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租 30%,合不合法?",
            answer="一般不合法,可拒绝并起诉。",
            extracted_cite_ids=["民法典-510"],
            extraction_confidence=0.9,
        ).model_dump_json() + "\n")
        f.write(CaseQA(
            case_id="train_002", cause="交通事故",
            question="撞了人住院要赔多少钱?",
            answer="参照伤残等级和实际损失,先走保险。",
            extracted_cite_ids=["道路交通安全法-76"],
            extraction_confidence=0.85,
        ).model_dump_json() + "\n")
    from scripts.build_cases_index import build_cases_index
    build_cases_index(
        jsonl_path=jsonl, collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_case_search_returns_evidence(case_index, tmp_run_dir):
    tool = CaseSearchTool(
        collection_name=case_index["collection"],
        sparse_artifact_path=case_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(CaseSearchArgs(query="房东涨租", k=2), rec)
    rec.close()
    assert result.error is None
    evidences = result.payload["evidences"]
    assert len(evidences) >= 1
    top = Evidence.model_validate(evidences[0])
    assert top.retriever == "case"
    assert "涨房租" in top.text or "涨租" in top.text


@pytest.mark.asyncio
async def test_case_search_filter_by_cause(case_index, tmp_run_dir):
    tool = CaseSearchTool(
        collection_name=case_index["collection"],
        sparse_artifact_path=case_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        CaseSearchArgs(query="撞人", k=5, cause="交通事故"),
        rec,
    )
    rec.close()
    for h in result.payload["evidences"]:
        assert Evidence.model_validate(h).metadata.get("cause") == "交通事故"
