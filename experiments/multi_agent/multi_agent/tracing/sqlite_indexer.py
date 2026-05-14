from __future__ import annotations
import sqlite3
from pathlib import Path
from multi_agent.schemas.events import BaseEvent


_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id   TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL,
    parent_id  TEXT,
    timestamp  TEXT NOT NULL,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    payload    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run        ON events(run_id);
CREATE INDEX IF NOT EXISTS idx_events_type       ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_agent      ON events(agent_name);
CREATE INDEX IF NOT EXISTS idx_events_parent     ON events(parent_id);
"""


class SqliteEventIndexer:
    """Indexes events into per-run SQLite for fast structured queries.

    Writes synchronously after each emit; close() flushes/commits.
    """

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def index(self, event: BaseEvent) -> None:
        if self._conn is None:
            raise RuntimeError("indexer already closed")
        agent_name = getattr(event, "agent_name", None)
        self._conn.execute(
            "INSERT OR REPLACE INTO events (event_id, run_id, parent_id, timestamp, event_type, agent_name, payload) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.run_id,
                event.parent_id,
                event.timestamp.isoformat(),
                event.event_type,
                agent_name,
                event.model_dump_json(),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
