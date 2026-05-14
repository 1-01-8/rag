import pytest
from multi_agent.tools.business.doc_interpret import DocInterpretTool
from multi_agent.schemas.doc_interpret import DocInterpretRequest
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_doc_interpret_returns_plain_language(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "key_clauses": [{"clause": "第三条", "summary": "保密义务"}],
  "rights_obligations": "甲方负保密义务,乙方有权要求审计",
  "risks": ["违约金过高", "管辖条款不利"],
  "plain_language_summary": "这是一份保密协议,主要保护商业秘密..."
}
```''', finish_reason="end_turn"),
    ])
    tool = DocInterpretTool(provider=provider, model="stub-1")
    result = await tool.call(
        DocInterpretRequest(doc_text="第三条 保密义务\n甲方应..."),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert "保密" in payload["plain_language_summary"]
    assert len(payload["risks"]) == 2
