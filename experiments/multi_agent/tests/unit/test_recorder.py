import json
import sqlite3
from multi_agent.tracing.recorder import Recorder
from multi_agent.schemas.events import RunStarted, RunFinished, AgentInvoked, AgentResponded, LLMRequested


def test_recorder_writes_to_both_jsonl_and_sqlite(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    rec.emit(RunStarted(event_id="e1", run_id="r1",
                        timestamp=rec.now(), parent_id=None,
                        query="q", config={}))
    rec.emit(RunFinished(event_id="e2", run_id="r1",
                         timestamp=rec.now(), parent_id=None,
                         status="ok", final_answer="a", error=None))
    rec.close()

    lines = (tmp_run_dir / "events.jsonl").read_text().splitlines()
    assert len(lines) == 2

    conn = sqlite3.connect(tmp_run_dir / "events.db")
    cur = conn.execute("SELECT COUNT(*) FROM events")
    assert cur.fetchone()[0] == 2
    conn.close()


def test_recorder_assigns_run_id_to_events(tmp_run_dir):
    """If event.run_id is set we trust it; recorder still validates consistency."""
    import pytest
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with pytest.raises(ValueError):
        rec.emit(RunStarted(event_id="e", run_id="WRONG",
                            timestamp=rec.now(), parent_id=None,
                            query="q", config={}))
    rec.close()


def test_span_emits_start_and_end_events(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="primary") as span:
        span.set_input({"query": "hi"})
        span.attach(AgentResponded(
            event_id=rec.fresh_event_id(), run_id="r1",
            timestamp=rec.now(), parent_id=span.span_id,
            agent_name="lawyer", output={"a": 1}, duration_ms=42,
        ))
    rec.close()

    import json
    lines = [json.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    types = [l["event_type"] for l in lines]
    # Expected: AgentInvoked (auto), AgentResponded (attached) - one span = one start, attach is independent
    assert "AgentInvoked" in types
    assert "AgentResponded" in types


def test_span_nesting_sets_parent_id(tmp_run_dir):
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="x") as outer:
        outer_id = outer.span_id
        with rec.span("llm_call", agent_name="lawyer",
                      provider="stub", model="m") as inner:
            assert inner.parent_id == outer_id
    rec.close()


def test_span_records_duration(tmp_run_dir):
    import time
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="lawyer", role="x") as span:
        time.sleep(0.01)
    rec.close()

    # Look for AgentResponded with duration_ms > 0
    import json
    lines = [json.loads(l) for l in (tmp_run_dir / "events.jsonl").read_text().splitlines()]
    responses = [l for l in lines if l["event_type"] == "AgentResponded"]
    assert len(responses) == 1
    assert responses[0]["duration_ms"] >= 10
