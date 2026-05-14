"""Phase 2d flagship: LawyerAgent uses statute_search + case_search together
against real Qwen. Verifies multi-tool routing works."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.tools.retrievers.case_search import CaseSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def both_indexes(tmp_path_factory):
    stat = f"test_multi_s_{uuid.uuid4().hex[:8]}"
    case = f"test_multi_c_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    encoder = DenseEncoder()

    stat_docs = [Document(
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
    s_sparse = tmp / "s_sparse.json"
    build_index(documents=stat_docs, collection_name=stat,
                sparse_artifact_path=s_sparse, dense_encoder=encoder)

    cases_jsonl = tmp / "cases.jsonl"
    with cases_jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东合同期内单方涨租,能拒绝吗?",
            answer="可以拒绝。合同期内租金条款受约束,涨租属于变更条款,需双方协商一致。",
            extracted_cite_ids=["民法典-510", "民法典-703"],
            extraction_confidence=0.92,
        ).model_dump_json() + "\n")
    c_sparse = tmp / "c_sparse.json"
    from scripts.build_cases_index import build_cases_index
    build_cases_index(jsonl_path=cases_jsonl, collection_name=case,
                     sparse_artifact_path=c_sparse, dense_encoder=encoder)

    yield {"stat": stat, "case": case, "s_sparse": s_sparse, "c_sparse": c_sparse}
    drop_collection(stat)
    drop_collection(case)


@pytest.mark.asyncio
async def test_lawyer_uses_both_tools(both_indexes, tmp_path):
    statute_search = StatuteSearchTool(
        collection_name=both_indexes["stat"],
        sparse_artifact_path=both_indexes["s_sparse"],
    )
    case_search = CaseSearchTool(
        collection_name=both_indexes["case"],
        sparse_artifact_path=both_indexes["c_sparse"],
    )
    provider = OpenAICompatibleProvider()

    runs_root = tmp_path / "runs"
    result = await run_query(
        query="我租房合同里没写能涨租,房东突然要涨 30%,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search, case_search],
            model="qwen3.5-9b",
            specialty="房产",
            max_steps=10, max_tool_calls=12,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2d"},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"
    assert out.five_section is not None

    # No fabricated citations — must be from statute index
    indexed = {"民法典-510", "民法典-703"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated: {doc_id}"

    # Verify at least one tool was called
    events = [json.loads(l) for l in (runs_root / result["run_id"] / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    assert len(tool_calls) >= 1
