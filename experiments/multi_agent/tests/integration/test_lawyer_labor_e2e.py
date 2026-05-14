# tests/integration/test_lawyer_labor_e2e.py
"""LawyerAgent 劳动 specialty smoke test against real Qwen.
Tolerant: 劳动合同法 is NOT in our corpus (ADR-15); test only checks that the
agent runs to a valid LawyerOutput without crashing and doesn't fabricate."""
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
def labor_index(tmp_path_factory):
    name = f"test_labor_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国社会保险法", law_short="社会保险法", source_path="t",
            chunks=[
                Chunk(doc_id="社会保险法-1", law_name="中华人民共和国社会保险法",
                      law_short="社会保险法", article_no="1",
                      text="为了规范社会保险关系,维护公民参加社会保险和享受社会保险待遇的合法权益,制定本法。"),
                Chunk(doc_id="社会保险法-58", law_name="中华人民共和国社会保险法",
                      law_short="社会保险法", article_no="58",
                      text="用人单位应当自用工之日起三十日内为其职工向社会保险经办机构申请办理社会保险登记。"),
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
async def test_labor_lawyer_produces_valid_output(labor_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=labor_index["collection"],
        sparse_artifact_path=labor_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="公司没给我交社保,我能怎么办?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty="劳动",
            max_steps=6,
            max_tool_calls=8,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "labor"},
    )
    assert result["status"] == "ok"
    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    # Loose checks — labor corpus is incomplete per ADR-15
    assert out.mode == "consultation"
    assert len(out.primary_answer) > 0

    # If lawyer cited articles, they must be from our test index
    indexed = {"社会保险法-1", "社会保险法-58"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated citation: {doc_id}"
