import pytest
from multi_agent.tools.business.contract_review import (
    ContractReviewTool, ContractReviewArgs,
)
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_contract_review_returns_structured_result(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='''```json
{
  "risk_items": [{"level": "high", "clause": "第5条", "reason": "霸王条款", "suggestion": "改为协商一致"}],
  "missing_clauses": ["违约金条款"],
  "summary": "合同存在高风险条款",
  "score": 60
}
```''', finish_reason="end_turn"),
    ])
    tool = ContractReviewTool(provider=provider, model="stub-1")
    result = await tool.call(
        ContractReviewArgs(contract_text="甲方应无条件接受乙方任何条款..."),
        rec,
    )
    rec.close()
    assert result.error is None
    payload = result.payload
    assert payload["score"] == 60
    assert len(payload["risk_items"]) == 1
    assert payload["risk_items"][0]["level"] == "high"


@pytest.mark.asyncio
async def test_contract_review_handles_malformed_json(tmp_run_dir):
    """If LLM returns invalid JSON, tool returns error gracefully."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text="this is not JSON", finish_reason="end_turn"),
    ])
    tool = ContractReviewTool(provider=provider, model="stub-1")
    result = await tool.call(ContractReviewArgs(contract_text="..."), rec)
    rec.close()
    assert result.error is not None
    assert "parse" in result.error.lower() or "json" in result.error.lower()
