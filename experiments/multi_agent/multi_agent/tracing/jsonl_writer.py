from __future__ import annotations
from pathlib import Path
from typing import TextIO
from multi_agent.schemas.events import BaseEvent


class JsonlEventWriter:
    """Append-only JSONL writer. Flushes after every write so trace
    survives crashes and can be tailed live."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh: TextIO | None = open(path, "a", encoding="utf-8")

    def write(self, event: BaseEvent) -> None:
        if self._fh is None:
            raise RuntimeError("writer already closed")
        line = event.model_dump_json()
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
