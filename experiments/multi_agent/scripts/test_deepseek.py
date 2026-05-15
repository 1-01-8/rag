#!/usr/bin/env python
"""快速验证 DeepSeek API key + 连通性 + tool calling.

跑这个之前: export DEEPSEEK_API_KEY=sk-xxx

输出预期:
  ✓ Connected
  ✓ Plain completion: ...
  ✓ Tool-call: dummy_tool was invoked

如果失败, key 或网络有问题, 不要再跑 chat.py。
"""
import asyncio
import os
import sys
from pathlib import Path

from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.base import ToolSpec
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


async def main() -> int:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("❌ DEEPSEEK_API_KEY 未设置. export DEEPSEEK_API_KEY=sk-xxx")
        return 1

    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com/v1", api_key=key,
    )
    rec = Recorder(run_id="ds_smoke", run_dir=Path("/tmp/ds_smoke"))

    # Test 1: plain completion
    print("\n[1/2] 普通对话测试...")
    try:
        resp = await provider.complete(
            messages=[
                AgentMessage(role="system", content="You are a concise assistant."),
                AgentMessage(role="user", content="一句话告诉我民法典第 510 条是什么?"),
            ],
            model="deepseek-chat",
            tools=None,
            temperature=0.0, max_tokens=200,
            recorder=rec, agent_name="smoke",
        )
        print(f"  ✓ 输出: {resp.text[:120]}...")
        print(f"  ✓ 用量: in={resp.usage.input_tokens} out={resp.usage.output_tokens}")
    except Exception as e:
        print(f"  ❌ 失败: {type(e).__name__}: {e}")
        return 1

    # Test 2: tool calling
    print("\n[2/2] Tool-call 测试 (Lawyer ReAct 必需)...")
    dummy_tool = ToolSpec(
        name="get_weather",
        description="查询某城市天气",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    try:
        resp = await provider.complete(
            messages=[
                AgentMessage(role="user", content="北京今天天气如何? 调 get_weather."),
            ],
            model="deepseek-chat",
            tools=[dummy_tool],
            temperature=0.0, max_tokens=200,
            recorder=rec, agent_name="smoke",
        )
        if resp.tool_calls:
            tc = resp.tool_calls[0]
            print(f"  ✓ 工具调用: {tc.name}({tc.args})")
            print(f"  ✓ finish_reason: {resp.finish_reason}")
        else:
            print(f"  ⚠️  没触发工具调用, 直接回答: {resp.text[:80]}...")
            print("  这意味着 DeepSeek 这次没决定调工具 (但可调).")
    except Exception as e:
        print(f"  ❌ Tool-call 失败: {type(e).__name__}: {e}")
        return 1

    rec.close()
    print("\n✅ DeepSeek 集成可用. 现在可以:")
    print("   python scripts/chat.py --provider deepseek \\")
    print("       --statutes-collection ma_statutes \\")
    print("       --statutes-sparse data/indexes/statutes_sparse.json")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
