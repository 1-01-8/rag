"""Phase 2c flagship test: real LawyerAgent (民事 specialty) handles a real
rental-dispute query using real Qwen + real Qdrant statute_search.
Skipped if vLLM not reachable."""
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
from multi_agent.tools.retrievers.exact_read import ExactReadTool
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
    not _qwen_reachable(),
    reason="Qwen vLLM not running at http://localhost:8000",
)


@pytest.fixture(scope="module")
def civil_index(tmp_path_factory):
    name = f"test_civil_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照合同相关条款或者交易习惯确定。"),
                Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="703",
                      text="租赁合同是出租人将租赁物交付承租人使用、收益，承租人支付租金的合同。"),
                Chunk(doc_id="民法典-720", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="720",
                      text="在租赁期限内因占有、使用租赁物获得的收益，归承租人所有，但是当事人另有约定的除外。"),
                Chunk(doc_id="民法典-188", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="188",
                      text="向人民法院请求保护民事权利的诉讼时效期间为三年。"),
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
async def test_civil_lawyer_handles_rental_dispute(civil_index, tmp_path):
    runs_root = tmp_path / "runs"
    statute_search = StatuteSearchTool(
        collection_name=civil_index["collection"],
        sparse_artifact_path=civil_index["sparse_path"],
    )
    read_article = ExactReadTool(collection_name=civil_index["collection"])
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="我租的房子合同还没到期,房东突然要涨 30% 房租,合法吗?我应该怎么办?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search, read_article],
            model="qwen3.5-9b",
            specialty="民事",
            max_steps=8,
            max_tool_calls=10,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2c", "test": "civil_rental_dispute"},
    )

    assert result["status"] == "ok"

    final_data = json.loads(result["final_answer"])
    out = LawyerOutput.model_validate(final_data)
    assert out.mode == "consultation"
    assert out.five_section is not None
    assert len(out.five_section.dispute_analysis) > 20
    assert len(out.five_section.applicable_laws) > 20
    assert len(out.five_section.remedy_suggestions) > 20

    # If lawyer cited any articles, they must be from our index (no fabrication)
    indexed_articles = {"民法典-510", "民法典-703", "民法典-720", "民法典-188"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed_articles, (
            f"Lawyer cited {doc_id} which is NOT in our test index. "
            f"This indicates Qwen fabricated a citation."
        )

    # Verify the lawyer actually called retrieval at least once
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    assert len(tool_calls) >= 1, "Lawyer should have called at least one retrieval tool"
