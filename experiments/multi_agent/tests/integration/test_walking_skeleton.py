"""Phase 1 acceptance test: full walking skeleton end-to-end.

A single query → stub agent → stub provider → trace files produced.
All invariants from spec §2.6 and §8.3 must hold.
"""
import json
import sqlite3
import pytest

from multi_agent.runner import run_query
from multi_agent.agents.stub_echo import EchoStubAgent
from multi_agent.providers.stub import StubProvider, ScriptedResponse


@pytest.mark.asyncio
async def test_walking_skeleton_full_invariants(tmp_path):
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"echoed": "hello world"}',
                         finish_reason="end_turn",
                         usage_input_tokens=10, usage_output_tokens=5),
    ])

    result = await run_query(
        query="hello world",
        agent_factory=lambda p, r: EchoStubAgent(name="echo", role="stub", provider=p, recorder=r),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub", "phase": "1"},
    )

    run_dir = runs_root / result["run_id"]

    # --- Invariant 1: meta.json exists with required fields ---
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["run_id"] == result["run_id"]
    assert meta["query"] == "hello world"
    assert "started_at" in meta and "finished_at" in meta

    # --- Invariant 2: events.jsonl ends with RunFinished ---
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    assert events[0]["event_type"] == "RunStarted"
    assert events[-1]["event_type"] == "RunFinished"
    assert events[-1]["status"] == "ok"

    # --- Invariant 3: every LLMRequested has a matching LLMResponded ---
    requested = [e for e in events if e["event_type"] == "LLMRequested"]
    responded = [e for e in events if e["event_type"] == "LLMResponded"]
    assert len(requested) == len(responded) == 1

    # --- Invariant 4: AgentInvoked / AgentResponded paired ---
    invoked = [e for e in events if e["event_type"] == "AgentInvoked"]
    agent_resp = [e for e in events if e["event_type"] == "AgentResponded"]
    assert len(invoked) == len(agent_resp) == 1

    # --- Invariant 5: parent_id chain — LLMRequested.parent_id == AgentInvoked.event_id ---
    assert requested[0]["parent_id"] == invoked[0]["event_id"]

    # --- Invariant 6: SQLite mirror is consistent with JSONL ---
    conn = sqlite3.connect(run_dir / "events.db")
    cur = conn.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == len(events)
    cur = conn.execute("SELECT event_type FROM events ORDER BY timestamp")
    db_types = [r[0] for r in cur.fetchall()]
    json_types = [e["event_type"] for e in events]
    assert sorted(db_types) == sorted(json_types)
    conn.close()

    # --- Invariant 7: final answer is structured ---
    final = json.loads(result["final_answer"])
    assert final["echoed"] == "hello world"


@pytest.mark.asyncio
async def test_walking_skeleton_error_path(tmp_path):
    """When provider responds invalid JSON, run still terminates with RunFinished(error)."""
    runs_root = tmp_path / "runs"
    provider = StubProvider(responses=[
        ScriptedResponse(text="not valid json at all", finish_reason="end_turn"),
    ])

    with pytest.raises(Exception):
        await run_query(
            query="x",
            agent_factory=lambda p, r: EchoStubAgent(name="echo", role="stub", provider=p, recorder=r),
            provider=provider,
            runs_root=runs_root,
            config={},
        )

    run_dirs = list(runs_root.glob("r_*"))
    assert len(run_dirs) == 1
    events = [json.loads(l) for l in (run_dirs[0] / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event_type"] == "RunFinished"
    assert events[-1]["status"] == "error"
