from __future__ import annotations
import json
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from multi_agent.schemas.events import (
    BaseEvent,
    AgentInvoked, AgentResponded,
    LLMRequested, LLMResponded,
    ToolCalled, ToolReturned,
)
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
        # Async-task-local stack: each coroutine gets its own view.
        # Use ContextVar at recorder level so isolated runs don't share state.
        self._span_stack_var: ContextVar[tuple[str, ...]] = ContextVar(
            f"span_stack_{run_id}", default=()
        )
        self._meta: dict = {"run_id": run_id, "started_at": self.now().isoformat()}

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
        self._meta["finished_at"] = self.now().isoformat()
        (self.run_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._jsonl.close()
        self._sqlite.close()
        self._closed = True

    def fresh_event_id(self) -> str:
        return fresh_event_id()

    def current_parent_id(self) -> str | None:
        stack = self._span_stack_var.get()
        return stack[-1] if stack else None

    def push_span(self, span_id: str) -> None:
        stack = self._span_stack_var.get()
        self._span_stack_var.set(stack + (span_id,))

    def pop_span(self, span_id: str) -> None:
        stack = self._span_stack_var.get()
        if not stack or stack[-1] != span_id:
            raise RuntimeError(f"span stack corrupted; expected {span_id}")
        self._span_stack_var.set(stack[:-1])

    def set_meta(self, **fields) -> None:
        self._meta.update(fields)

    def span(self, kind: str, **attrs) -> "_SpanCM":
        """Context manager that emits a start event on enter and a matching end event on exit.

        kind ∈ {"agent_invoke", "llm_call", "tool_call"} (extensible).
        attrs go to the start event's required fields (see _SpanCM._build_start).
        """
        return _SpanCM(recorder=self, kind=kind, attrs=attrs)


class _SpanCM:
    """Span context manager. Emits {kind}_start on __enter__,
    {kind}_end on __exit__. Tracks parent_id from outer spans via a per-recorder stack."""

    def __init__(self, recorder: "Recorder", kind: str, attrs: dict):
        self.recorder = recorder
        self.kind = kind
        self.attrs = attrs
        self.span_id: str = ""
        self.parent_id: str | None = None
        self._t0: float = 0.0
        self._input: dict | None = None
        self._output: dict | None = None
        self._error: str | None = None

    def __enter__(self):
        self.span_id = self.recorder.fresh_event_id()
        self.parent_id = self.recorder.current_parent_id()
        self.recorder.push_span(self.span_id)
        self._t0 = monotonic()
        start = self._build_start()
        self.recorder.emit(start)
        return self

    def __exit__(self, exc_type, exc, _tb):
        del _tb  # protocol-required traceback param, intentionally unused
        duration_ms = int((monotonic() - self._t0) * 1000)
        if exc is not None:
            self._error = f"{exc_type.__name__}: {exc}"
        end = self._build_end(duration_ms)
        self.recorder.emit(end)
        self.recorder.pop_span(self.span_id)
        return False  # never swallow

    def set_input(self, payload: dict) -> None:
        self._input = payload

    def set_output(self, payload: dict) -> None:
        self._output = payload

    def attach(self, event: BaseEvent) -> None:
        """Emit an arbitrary event, parented to this span."""
        self.recorder.emit(event)

    def _build_start(self) -> BaseEvent:
        ts = self.recorder.now()
        if self.kind == "agent_invoke":
            return AgentInvoked(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                agent_name=self.attrs.get("agent_name", ""),
                role=self.attrs.get("role", ""),
                input=self._input or {},
            )
        if self.kind == "llm_call":
            return LLMRequested(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                provider=self.attrs.get("provider", ""),
                model=self.attrs.get("model", ""),
                messages=self.attrs.get("messages", []),
                params=self.attrs.get("params", {}),
            )
        if self.kind == "tool_call":
            return ToolCalled(
                event_id=self.span_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.parent_id,
                tool_name=self.attrs.get("tool_name", ""),
                args=self.attrs.get("args", {}),
                agent_name=self.attrs.get("agent_name", ""),
            )
        raise ValueError(f"unknown span kind: {self.kind}")

    def _build_end(self, duration_ms: int) -> BaseEvent:
        ts = self.recorder.now()
        end_id = self.recorder.fresh_event_id()
        if self.kind == "agent_invoke":
            return AgentResponded(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                agent_name=self.attrs.get("agent_name", ""),
                output=self._output or ({"error": self._error} if self._error else {}),
                duration_ms=duration_ms,
            )
        if self.kind == "llm_call":
            return LLMResponded(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                raw_response="" if self._output is None else str(self._output.get("raw", "")),
                usage=(self._output or {}).get("usage", {}),
                duration_ms=duration_ms,
                finish_reason=(self._output or {}).get("finish_reason", "end_turn"),
            )
        if self.kind == "tool_call":
            return ToolReturned(
                event_id=end_id, run_id=self.recorder.run_id,
                timestamp=ts, parent_id=self.span_id,
                result=self._output, error=self._error, duration_ms=duration_ms,
            )
        raise ValueError(f"unknown span kind: {self.kind}")
