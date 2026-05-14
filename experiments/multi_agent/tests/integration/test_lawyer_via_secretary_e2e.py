"""Phase 4 E2E: Lawyer delegates research to Secretary; Secretary uses retrievers."""
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
from multi_agent.agents.secretary import SecretaryAgent, SecretaryAsTool
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_sec_{uuid.uuid4().hex[:8]}"
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
async def test_lawyer_delegates_to_secretary(statute_index, tmp_path):
    """Lawyer gets ONLY [ask_secretary]; Secretary internally has statute_search."""
    runs_root = tmp_path / "runs"
    provider = OpenAICompatibleProvider()

    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    def lawyer_factory(p, r):
        secretary = SecretaryAgent(
            name="secretary", role="research",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            max_steps=5, max_tool_calls=8,
        )
        return LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[SecretaryAsTool(secretary_agent=secretary)],
            model="qwen3.5-9b",
            specialty="民事",
            max_steps=8, max_tool_calls=10,
        )

    result = await run_query(
        query="房东合同期内涨租 30% 合法吗?",
        agent_factory=lawyer_factory,
        provider=provider, runs_root=runs_root, config={},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"
    events = [json.loads(l) for l in (runs_root / result["run_id"] / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    tool_names = {e["tool_name"] for e in tool_calls}
    assert "ask_secretary" in tool_names, f"Lawyer didn't call ask_secretary; tools called: {tool_names}"
