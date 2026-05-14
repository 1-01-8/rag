import json
import uuid
import pytest
from multi_agent.schemas.case import CaseQA
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client, drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


@pytest.fixture
def cases_jsonl(tmp_path):
    """Write a tiny extracted-cases JSONL."""
    p = tmp_path / "cases.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租怎么办?",
            answer="协商不成可起诉。", extracted_cite_ids=["民法典-510"],
            extraction_confidence=0.9,
        ).model_dump_json() + "\n")
        f.write(CaseQA(
            case_id="train_002", cause="交通事故",
            question="撞了人住院要赔多少?",
            answer="参照伤残等级和实际损失。", extracted_cite_ids=["道路交通安全法-76"],
            extraction_confidence=0.85,
        ).model_dump_json() + "\n")
    return p


@pytest.fixture
def temp_collection():
    name = f"test_cases_{uuid.uuid4().hex[:8]}"
    yield name
    drop_collection(name)


def test_build_cases_index_creates_points(cases_jsonl, temp_collection, tmp_path):
    from scripts.build_cases_index import build_cases_index
    artifacts = build_cases_index(
        jsonl_path=cases_jsonl,
        collection_name=temp_collection,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    client = get_qdrant_client()
    count = client.count(temp_collection).count
    assert count == 2


def test_build_cases_index_payload_has_extracted_cites(cases_jsonl, temp_collection, tmp_path):
    from scripts.build_cases_index import build_cases_index
    build_cases_index(
        jsonl_path=cases_jsonl,
        collection_name=temp_collection,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    client = get_qdrant_client()
    points, _ = client.scroll(collection_name=temp_collection, limit=10, with_payload=True)
    for pt in points:
        assert "case_id" in pt.payload
        assert "cause" in pt.payload
        assert "extracted_cite_ids" in pt.payload
