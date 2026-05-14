import pytest
from multi_agent.tools.business.doc_generation import DocGenerationTool
from multi_agent.schemas.doc_generation import DocGenRequest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_doc_generation_returns_structured_doc(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "doc_type": "民事起诉状",
  "content": "原告: 张三\\n被告: 李四\\n诉讼请求: ...",
  "placeholders_filled": {"plaintiff": "张三", "defendant": "李四"},
  "meta": {"jurisdiction": "北京市朝阳区人民法院"}
}
```''', finish_reason="end_turn"),
    ])
    tool = DocGenerationTool(provider=provider, model="stub-1")
    result = await tool.call(
        DocGenRequest(
            doc_type="民事起诉状",
            case_facts="原告与被告...",
            parties={"plaintiff": "张三", "defendant": "李四"},
        ),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert payload["doc_type"] == "民事起诉状"
    assert "原告" in payload["content"]
