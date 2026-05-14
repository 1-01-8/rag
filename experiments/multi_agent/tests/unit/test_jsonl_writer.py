import json
from datetime import datetime, timezone
from multi_agent.tracing.jsonl_writer import JsonlEventWriter
from multi_agent.schemas.events import RunStarted, RunFinished


def _ts():
    return datetime.now(timezone.utc)


def test_appends_one_line_per_event(tmp_path):
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.write(RunStarted(event_id="e1", run_id="r", timestamp=_ts(),
                       parent_id=None, query="q", config={}))
    w.write(RunFinished(event_id="e2", run_id="r", timestamp=_ts(),
                        parent_id=None, status="ok",
                        final_answer="a", error=None))
    w.close()

    lines = p.read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["event_type"] == "RunStarted"
    assert e2["event_type"] == "RunFinished"


def test_flushes_on_each_write(tmp_path):
    """Even before close(), the file should contain written events."""
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.write(RunStarted(event_id="e1", run_id="r", timestamp=_ts(),
                       parent_id=None, query="q", config={}))
    assert p.exists()
    assert "RunStarted" in p.read_text()
    w.close()


def test_close_is_idempotent(tmp_path):
    p = tmp_path / "events.jsonl"
    w = JsonlEventWriter(p)
    w.close()
    w.close()  # must not raise
