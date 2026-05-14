from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

from multi_agent.schemas.events import BaseEvent
from multi_agent.tracing.jsonl_writer import JsonlEventWriter
from multi_agent.tracing.sqlite_indexer import SqliteEventIndexer
from multi_agent.tracing.ulid_gen import fresh_event_id


class Recorder:
    """Central trace recorder. Writes every event to both JSONL and SQLite.

    Span context manager will be added in Task 13.
    """

    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = JsonlEventWriter(self.run_dir / "events.jsonl")
        self._sqlite = SqliteEventIndexer(self.run_dir / "events.db")
        self._closed = False

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def emit(self, event: BaseEvent) -> None:
        if self._closed:
            raise RuntimeError("recorder already closed")
        if event.run_id != self.run_id:
            raise ValueError(
                f"event.run_id={event.run_id!r} does not match recorder.run_id={self.run_id!r}"
            )
        self._jsonl.write(event)
        self._sqlite.index(event)

    def close(self) -> None:
        if self._closed:
            return
        self._jsonl.close()
        self._sqlite.close()
        self._closed = True

    def fresh_event_id(self) -> str:
        return fresh_event_id()
