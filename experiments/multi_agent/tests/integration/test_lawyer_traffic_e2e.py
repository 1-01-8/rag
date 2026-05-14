# tests/integration/test_lawyer_traffic_e2e.py
"""LawyerAgent 交通 specialty E2E. Corpus has 道路交通安全法 — good coverage."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(), reason="Qwen vLLM not running",
)


@pytest.fixture(scope="module")
def traffic_index(tmp_path_factory):
    name = f"test_traffic_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国道路交通安全法", law_short="道路交通安全法",
            source_path="t",
            chunks=[
                Chunk(doc_id="道路交通安全法-76", law_name="中华人民共和国道路交通安全法",
                      law_short="道路交通安全法", article_no="76",
                      text="机动车发生交通事故造成人身伤亡、财产损失的,由保险公司在机动车第三者责任强制保险责任限额范围内予以赔偿。"),
            ],
        ),
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-1208", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="1208",
                      text="机动车发生交通事故造成损害的,依照道路交通安全法律和本法的有关规定承担赔偿责任。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_traffic_lawyer_produces_valid_output(traffic_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=traffic_index["collection"],
        sparse_artifact_path=traffic_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="我开车不小心撞了人,对方住院了,我要承担什么责任?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty="交通",
            max_steps=6,
            max_tool_calls=8,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "traffic"},
    )
    assert result["status"] == "ok"
    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    assert out.mode == "consultation"
    # Citations (if any) should be from indexed articles only
    indexed = {"道路交通安全法-76", "民法典-1208"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated citation: {doc_id}"
