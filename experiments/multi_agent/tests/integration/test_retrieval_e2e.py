"""Phase 2a acceptance test: stub agent uses statute_search tool against
a real (small) Qdrant index, full trace is consistent.
"""
import json
import uuid
from pathlib import Path
import pytest
from pydantic import BaseModel

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.messages import ToolCallRequest
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.runner import run_query


class _Output(BaseModel):
    summary: str


class _SearchAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "Use statute_search then summarize."

    def output_schema(self):
        return _Output


@pytest.fixture(scope="module")
def populated_index(tmp_path_factory):
    name = f"test_e2e_{uuid.uuid4().hex[:8]}"
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
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_retrieval_e2e_trace_invariants(populated_index, tmp_path):
    runs_root = tmp_path / "runs"
    search_tool = StatuteSearchTool(
        collection_name=populated_index["collection"],
        sparse_artifact_path=populated_index["sparse_path"],
    )

    # Scripted: first response calls statute_search, second produces final answer
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(
                tool_use_id="t1", tool_name="statute_search",
                args={"query": "合同补充", "k": 2},
            )],
            finish_reason="tool_use",
        ),
        ScriptedResponse(
            text='{"summary": "found contract supplementation rules"}',
            finish_reason="end_turn",
        ),
    ])

    result = await run_query(
        query="搜索合同补充相关法条",
        agent_factory=lambda p, r: _SearchAgent(
            name="searcher", role="lookup",
            provider=p, recorder=r,
            tools=[search_tool],
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub+retrieval"},
    )

    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]

    # Required chain: RunStarted → AgentInvoked → LLMRequested → LLMResponded
    # → ToolCalled → ToolReturned → LLMRequested → LLMResponded
    # → AgentResponded → RunFinished
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert "ToolCalled" in types
    assert "ToolReturned" in types

    # Tool call must be statute_search and produce a non-empty result
    tool_called = next(e for e in events if e["event_type"] == "ToolCalled")
    assert tool_called["tool_name"] == "statute_search"

    tool_returned = next(e for e in events if e["event_type"] == "ToolReturned")
    assert tool_returned["error"] is None
    assert tool_returned["result"]["count"] >= 1
    # At least one of the returned evidences is the contract article
    doc_ids = [e["doc_id"] for e in tool_returned["result"]["evidences"]]
    assert "民法典-510" in doc_ids

    # ContextVar invariant: ToolCalled.parent_id must be the agent_invoke span_id
    agent_invoked = next(e for e in events if e["event_type"] == "AgentInvoked")
    assert tool_called["parent_id"] == agent_invoked["event_id"]
