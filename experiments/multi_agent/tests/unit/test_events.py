from datetime import datetime, timezone
from multi_agent.schemas.events import BaseEvent, RunStarted, RunFinished


def test_base_event_required_fields():
    """BaseEvent must carry id, run_id, timestamp, parent_id, event_type."""
    e = RunStarted(
        event_id="01EVENT",
        run_id="01RUN",
        timestamp=datetime.now(timezone.utc),
        parent_id=None,
        query="test query",
        config={"profile": "stub"},
    )
    assert e.event_id == "01EVENT"
    assert e.run_id == "01RUN"
    assert e.parent_id is None
    assert e.event_type == "RunStarted"
    assert e.query == "test query"


def test_run_finished_ok_status():
    e = RunFinished(
        event_id="01EVT", run_id="01RUN",
        timestamp=datetime.now(timezone.utc), parent_id=None,
        status="ok",
        final_answer="42",
        error=None,
    )
    assert e.event_type == "RunFinished"
    assert e.status == "ok"
    assert e.final_answer == "42"


def test_run_finished_rejects_unknown_status():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunFinished(
            event_id="e", run_id="r",
            timestamp=datetime.now(timezone.utc), parent_id=None,
            status="bogus",  # not in Literal
            final_answer=None, error=None,
        )


from multi_agent.schemas.events import (
    AgentInvoked, AgentResponded,
    LLMRequested, LLMResponded,
    ToolCalled, ToolReturned,
    MemoryRead, MemoryWritten,
    SupervisorVerdict,
)


def _ts():
    return datetime.now(timezone.utc)


def test_agent_invoked_and_responded():
    inv = AgentInvoked(
        event_id="e1", run_id="r", timestamp=_ts(), parent_id=None,
        agent_name="lawyer", role="primary_advisor",
        input={"query": "hi"},
    )
    assert inv.event_type == "AgentInvoked"
    assert inv.agent_name == "lawyer"

    resp = AgentResponded(
        event_id="e2", run_id="r", timestamp=_ts(), parent_id="e1",
        agent_name="lawyer", output={"answer": "hello"}, duration_ms=1234,
    )
    assert resp.duration_ms == 1234


def test_llm_request_response():
    req = LLMRequested(
        event_id="l1", run_id="r", timestamp=_ts(), parent_id="e1",
        provider="stub", model="stub-1", messages=[{"role": "user", "content": "hi"}],
        params={"temperature": 0},
    )
    resp = LLMResponded(
        event_id="l2", run_id="r", timestamp=_ts(), parent_id="l1",
        raw_response="ok", usage={"input_tokens": 5, "output_tokens": 1},
        duration_ms=10, finish_reason="end_turn",
    )
    assert req.event_type == "LLMRequested"
    assert resp.finish_reason == "end_turn"


def test_tool_called_and_returned():
    call = ToolCalled(
        event_id="t1", run_id="r", timestamp=_ts(), parent_id="e1",
        tool_name="search", args={"q": "x"}, agent_name="secretary",
    )
    ret = ToolReturned(
        event_id="t2", run_id="r", timestamp=_ts(), parent_id="t1",
        result={"hits": 3}, error=None, duration_ms=8,
    )
    assert call.tool_name == "search"
    assert ret.error is None


def test_memory_events():
    rd = MemoryRead(
        event_id="m1", run_id="r", timestamp=_ts(), parent_id=None,
        target="sticky", query={"intent": "full"}, hits=[{"path": "x"}],
        agent_name="lawyer",
    )
    wr = MemoryWritten(
        event_id="m2", run_id="r", timestamp=_ts(), parent_id=None,
        target="agent_notes", payload={"name": "n"}, path="agent_notes/n.md",
        agent_name="supervisor",
    )
    assert rd.target == "sticky"
    assert wr.path == "agent_notes/n.md"


def test_supervisor_verdict():
    v = SupervisorVerdict(
        event_id="v1", run_id="r", timestamp=_ts(), parent_id=None,
        verdict="pass", issues=[],
    )
    assert v.verdict == "pass"


from multi_agent.schemas.events import AnyEvent, event_from_dict
import json


def test_dump_and_reload_via_union():
    """Any event can be dumped to JSON and reloaded via the union."""
    original = AgentInvoked(
        event_id="e", run_id="r", timestamp=_ts(), parent_id=None,
        agent_name="lawyer", role="primary", input={},
    )
    j = original.model_dump_json()
    raw = json.loads(j)
    reloaded = event_from_dict(raw)
    assert isinstance(reloaded, AgentInvoked)
    assert reloaded.agent_name == "lawyer"


def test_event_type_discriminates():
    raw = {
        "event_type": "RunFinished",
        "event_id": "e", "run_id": "r",
        "timestamp": _ts().isoformat(),
        "parent_id": None, "status": "ok",
        "final_answer": "x", "error": None,
    }
    obj = event_from_dict(raw)
    assert isinstance(obj, RunFinished)
    assert obj.final_answer == "x"


def test_event_unknown_type_raises():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        event_from_dict({"event_type": "WhoKnows", "event_id": "e",
                         "run_id": "r", "timestamp": _ts().isoformat()})
