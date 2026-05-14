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
