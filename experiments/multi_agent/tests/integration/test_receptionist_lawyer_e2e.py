"""Phase 3 flagship: Receptionist classifies + decomposes, then Lawyer handles
the case. Real Qwen, real Qdrant."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.base import AgentInput
from multi_agent.runner import run_query
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_r2l_{uuid.uuid4().hex[:8]}"
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
async def test_receptionist_then_lawyer(statute_index, tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    provider = OpenAICompatibleProvider()

    # Step 1: Receptionist classifies
    rec_run_id = fresh_run_id()
    rec_recorder = Recorder(run_id=rec_run_id, run_dir=runs_root / rec_run_id)
    receptionist = ReceptionistAgent(
        name="receptionist", role="triage",
        provider=provider, recorder=rec_recorder,
        model="qwen3.5-9b", max_steps=2,
    )
    triage_input = AgentInput(payload={"query": "我租的房合同一年,房东要涨 30% 房租,合法吗?"})
    triage_out = await receptionist.run(triage_input)
    rec_recorder.close()
    assert isinstance(triage_out.payload, ReceptionistOutput)
    # Specialty should be one of the civil-related categories
    assert triage_out.payload.primary_specialty in (
        "民事", "房产", "通用", "民法", "合同",
    )

    # Step 2: Lawyer with retrieval, specialty from Receptionist
    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )
    # Map Receptionist's primary_specialty to a Lawyer specialty (accept fallback)
    receptionist_specialty = triage_out.payload.primary_specialty
    if receptionist_specialty in ("民事", "房产", "婚姻", "劳动", "交通"):
        lawyer_specialty = receptionist_specialty
    else:
        lawyer_specialty = "民事"   # fallback to a known lawyer specialty

    result = await run_query(
        query="我租的房合同一年,房东要涨 30% 房租,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty=lawyer_specialty,
            max_steps=8, max_tool_calls=10,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "3", "received_from_receptionist": triage_out.payload.model_dump()},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"

    # No fabricated citations
    indexed = {"民法典-510", "民法典-703"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated: {doc_id}"
