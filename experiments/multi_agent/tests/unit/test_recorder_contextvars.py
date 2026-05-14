"""Verify span stack is async-task-local. Without this, concurrent
spans on the same recorder corrupt the parent_id chain.
"""
import asyncio
import json
import pytest
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_concurrent_spans_have_independent_parents(tmp_run_dir):
    """Two coroutines opening tool_call spans concurrently must each have
    the SAME outer parent (the agent_invoke span), not each other."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)

    async def inner_tool(name: str, delay: float):
        with rec.span("tool_call", tool_name=name, args={}, agent_name="a") as s:
            await asyncio.sleep(delay)
            return s.span_id, s.parent_id

    with rec.span("agent_invoke", agent_name="a", role="t") as outer:
        outer_id = outer.span_id
        # Two concurrent tools that overlap in time
        results = await asyncio.gather(
            inner_tool("alpha", 0.05),
            inner_tool("beta", 0.01),  # finishes first
        )
    rec.close()

    span_a, parent_a = results[0]
    span_b, parent_b = results[1]
    # Both inner spans must have the outer agent_invoke as parent
    # Without ContextVar, the second-to-exit would have its sibling as parent
    assert parent_a == outer_id, f"alpha's parent {parent_a} != outer {outer_id}"
    assert parent_b == outer_id, f"beta's parent {parent_b} != outer {outer_id}"
    assert span_a != span_b


@pytest.mark.asyncio
async def test_nested_spans_still_chain_correctly(tmp_run_dir):
    """Single-coroutine nested spans must still form a chain (no regression)."""
    rec = Recorder(run_id="r2", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="a", role="t") as outer:
        outer_id = outer.span_id
        with rec.span("tool_call", tool_name="x", args={}, agent_name="a") as inner:
            assert inner.parent_id == outer_id
            with rec.span("llm_call", provider="stub", model="m", agent_name="a") as deeper:
                assert deeper.parent_id == inner.span_id
    rec.close()
