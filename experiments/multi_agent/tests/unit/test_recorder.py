import json
import sqlite3
from multi_agent.tracing.recorder import Recorder
from multi_agent.schemas.events import RunStarted, RunFinished


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
