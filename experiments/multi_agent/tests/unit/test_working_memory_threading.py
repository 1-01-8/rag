import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.schemas.messages import ToolCallRequest, ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder


class _Args(BaseModel):
    q: str


class _Tool(Tool):
    name: str = "fake_search"
    description: str = "returns one evidence"
    args_schema: type[BaseModel] = _Args

    async def call(self, args, recorder):
        ev = Evidence(
            doc_id="民法典-510",
            law_name="中华人民共和国民法典",
            law_short="民法典",
            article_no="510",
            text="当事人就合同补充内容...",
            score=0.9,
            retriever="hybrid",
        )
        return ToolResult(tool_use_id="x", payload={"evidences": [ev.model_dump()]})


class _Out(BaseModel):
    answer: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_working_memory_accumulates_evidence(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(tool_use_id="t1", tool_name="fake_search", args={"q": "x"})],
            finish_reason="tool_use",
        ),
        ScriptedResponse(text='{"answer": "ok"}', finish_reason="end_turn"),
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   tools=[_Tool()], model="stub-1")
    out = await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert agent.working_memory is not None
    assert len(agent.working_memory.retrieved_evidence) == 1
    assert agent.working_memory.retrieved_evidence[0].doc_id == "民法典-510"


@pytest.mark.asyncio
async def test_working_memory_empty_when_no_tool_calls(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[ScriptedResponse(text='{"answer": "ok"}')])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec, model="stub-1")
    await agent.run(AgentInput(payload={"query": "x"}))
    rec.close()
    assert agent.working_memory is not None
    assert agent.working_memory.retrieved_evidence == []
