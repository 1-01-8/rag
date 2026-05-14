import sqlite3
from datetime import datetime, timezone
from multi_agent.tracing.sqlite_indexer import SqliteEventIndexer
from multi_agent.schemas.events import RunStarted, AgentInvoked


def _ts():
    return datetime.now(timezone.utc)


def test_indexer_creates_events_table(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.close()

    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    assert "events" in tables
    conn.close()


def test_index_writes_event_row(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.index(RunStarted(event_id="e1", run_id="r1", timestamp=_ts(),
                         parent_id=None, query="q", config={}))
    idx.index(AgentInvoked(event_id="e2", run_id="r1", timestamp=_ts(),
                           parent_id="e1", agent_name="lawyer",
                           role="primary", input={}))
    idx.close()

    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT event_id, run_id, event_type, agent_name FROM events ORDER BY event_id")
    rows = cur.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("e1", "r1", "RunStarted", None)
    assert rows[1] == ("e2", "r1", "AgentInvoked", "lawyer")
    conn.close()


def test_indexer_close_idempotent(tmp_path):
    db = tmp_path / "events.db"
    idx = SqliteEventIndexer(db)
    idx.close()
    idx.close()  # must not raise
