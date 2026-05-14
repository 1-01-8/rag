"""Phase 5a E2E: Lawyer + Supervisor pipeline against real Qwen."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.supervisor import SupervisorAgent
from multi_agent.orchestration.supervised import run_with_supervisor


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_sup_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
            Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="703",
                  text="租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ],
    )]
    build_index(documents=docs, collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_supervised_lawyer_passes(statute_index, tmp_path):
    runs_root = tmp_path / "runs"
    provider = OpenAICompatibleProvider()

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    result = await run_with_supervisor(
        query="房东合同期内涨租 30% 合法吗?",
        lawyer_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b", specialty="民事",
            max_steps=8, max_tool_calls=10,
        ),
        supervisor_factory=lambda p, r: SupervisorAgent(
            name="supervisor", role="qa",
            provider=p, recorder=r,
            model="qwen3.5-9b",
            max_steps=3, max_pre_tool_rejections=5,
        ),
        lawyer_provider=provider,
        supervisor_provider=provider,
        runs_root=runs_root,
    )

    assert result["lawyer_result"]["status"] == "ok"
    verdict = result["supervisor_verdict"]["verdict"]
    # Supervisor should give pass or revise — reject would indicate Lawyer fabricated
    assert verdict in ("pass", "revise"), (
        f"Unexpected verdict: {verdict}. Issues: {result['supervisor_verdict'].get('issues')}"
    )
