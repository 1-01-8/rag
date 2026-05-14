"""Phase 2b acceptance test: real Qwen vLLM + real Qdrant statute_search +
multi_agent base agent ReAct loop. No mocks, no stubs (except the agent's
'system_prompt' which is minimal).

Skipped if vLLM not reachable.
"""
import json
import uuid
import httpx
import pytest
from pydantic import BaseModel

from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.base import BaseAgent
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _qwen_reachable(),
    reason="Qwen vLLM not running at http://localhost:8000 — start it per Task 1",
)


class _Out(BaseModel):
    summary: str


class _LegalAgent(BaseAgent):
    def system_prompt(self) -> str:
        return (
            "你是法律助手。当用户问法条相关问题时,先调用 statute_search 工具检索,"
            '然后用 JSON 总结结果: {"summary": "<简要说明>"}。'
            "禁止编造法条号。"
        )
    def output_schema(self):
        return _Out


@pytest.fixture(scope="module")
def populated_index(tmp_path_factory):
    name = f"test_qwen_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照交易习惯确定。"),
                Chunk(doc_id="民法典-563", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="563",
                      text="一方违约时，对方可以解除合同。"),
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
async def test_real_qwen_finds_civil_code_510(populated_index, tmp_path):
    runs_root = tmp_path / "runs"
    search_tool = StatuteSearchTool(
        collection_name=populated_index["collection"],
        sparse_artifact_path=populated_index["sparse_path"],
    )
    provider = OpenAICompatibleProvider()

    result = await run_query(
        query="合同补充内容怎么确定?",
        agent_factory=lambda p, r: _LegalAgent(
            name="lawyer", role="lookup",
            provider=p, recorder=r,
            tools=[search_tool],
            model="qwen3.5-9b",
            max_steps=5,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "all-local-qwen+statute_search"},
    )

    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]

    # Real provider should have emitted ≥1 LLMRequested + matching LLMResponded
    n_req = types.count("LLMRequested")
    n_resp = types.count("LLMResponded")
    assert n_req >= 1 and n_resp == n_req

    # If the agent called the tool, verify the retrieval surfaced the contract article
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    if tool_calls:
        for ret in events:
            if ret["event_type"] == "ToolReturned" and ret.get("result"):
                if "evidences" in ret["result"]:
                    doc_ids = [e.get("doc_id") for e in ret["result"]["evidences"]]
                    assert "民法典-510" in doc_ids, f"expected 510 in retrieved evidences, got {doc_ids}"
                    break

    # Final answer should be valid JSON conforming to _Out schema
    final = json.loads(result["final_answer"])
    assert "summary" in final
    assert isinstance(final["summary"], str) and len(final["summary"]) > 0
