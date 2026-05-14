import pytest
from pydantic import BaseModel
from multi_agent.tools.base import Tool, ToolSpec
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder


class EchoArgs(BaseModel):
    msg: str


class EchoTool(Tool):
    name: str = "echo"
    description: str = "echo back the message"
    args_schema: type[BaseModel] = EchoArgs

    async def call(self, args: EchoArgs, recorder: Recorder) -> ToolResult:
        return ToolResult(tool_use_id="t-internal", payload={"echo": args.msg})


@pytest.mark.asyncio
async def test_tool_call_returns_result(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    t = EchoTool()
    result = await t.call(EchoArgs(msg="hi"), rec)
    rec.close()
    assert result.payload == {"echo": "hi"}


def test_tool_spec_exposes_input_schema():
    t = EchoTool()
    spec = t.to_spec()
    assert spec.name == "echo"
    assert "msg" in spec.input_schema["properties"]
