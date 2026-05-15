"""Trace-derived metrics (Phase 5b §7.6).

Reads a run's events.jsonl and aggregates counters from the actual event
schema written by Recorder / JsonlEventWriter.

Actual event field names (differ from the plan sketch):
  - event_type  (plan sketch used "kind")
  - timestamp   ISO-8601 string (plan sketch used "ts_ms" int)
  - usage sub-fields: input_tokens, output_tokens, cache_read_tokens
    All fields are top-level (no nested "data" wrapper)
  - Errors are detected via ToolReturned.error or RunFinished.status == "error"
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field


class RunMetrics(BaseModel):
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_hit_rate: float = 0.0
    agent_invocations: int = 0
    tool_calls_total: int = 0
    react_steps_total: int = 0
    supervisor_verdict: str | None = None
    final_answer_mode: str | None = None
    citation_count: int = 0
    errors: int = 0
    cost_usd: float = 0.0


def _parse_ts(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp string to datetime (handles both +00:00 and Z suffixes)."""
    ts_str = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(ts_str)


def derive_run_metrics(run_dir: Path) -> RunMetrics:
    """Derive RunMetrics by scanning events.jsonl in run_dir.

    Args:
        run_dir: Path to the run directory containing events.jsonl.

    Returns:
        RunMetrics populated from the trace events.

    Raises:
        FileNotFoundError: If events.jsonl does not exist in run_dir.
    """
    events_path = Path(run_dir) / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    from multi_agent.eval.pricing import compute_cost_usd

    m = RunMetrics()
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    model_by_request_id: dict[str, str] = {}
    per_model_usage: dict[str, dict[str, int]] = {}

    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        event_type = e.get("event_type") or e.get("kind", "")
        ts_raw = e.get("timestamp")

        # Parse timestamp — support both ISO string (real events) and int ms (test stubs)
        ts: datetime | None = None
        if isinstance(ts_raw, str):
            try:
                ts = _parse_ts(ts_raw)
            except ValueError:
                pass
        elif isinstance(ts_raw, (int, float)):
            # Fallback: treat as ms since some legacy test may use int ts_ms
            from datetime import timezone, timedelta
            ts = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=ts_raw)

        if event_type == "RunStarted":
            start_ts = ts
        elif event_type == "RunFinished":
            end_ts = ts
            # Phase 5q: parse final_answer JSON to populate answer_mode + citation_count
            fa = e.get("final_answer")
            if fa:
                try:
                    payload = json.loads(fa) if isinstance(fa, str) else fa
                    if isinstance(payload, dict):
                        if not m.final_answer_mode:
                            m.final_answer_mode = payload.get("mode")
                        cits = payload.get("citations")
                        if isinstance(cits, list):
                            m.citation_count = len(cits)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Do not count RunFinished.status as an independent error —
            # it typically reflects a downstream tool/agent failure already counted.

        elif event_type == "AgentInvoked":
            m.agent_invocations += 1

        elif event_type == "AgentResponded":
            # Phase 5q: each AgentResponded marks one completed ReAct cycle for
            # the agent. Use it as the proxy for "react steps". (LLMRequested
            # would over-count because tool-call retries fire multiple LLM rounds
            # per logical step.)
            m.react_steps_total += 1

        elif event_type == "ToolCalled":
            m.tool_calls_total += 1

        elif event_type == "ToolReturned":
            # A tool error is recorded in the error field
            if e.get("error"):
                m.errors += 1

        elif event_type == "LLMRequested":
            rid = e.get("event_id")
            model = e.get("model", "")
            if rid and model:
                model_by_request_id[rid] = model

        elif event_type == "LLMResponded":
            usage = e.get("usage") or {}
            m.total_input_tokens += usage.get("input_tokens", 0) or 0
            m.total_output_tokens += usage.get("output_tokens", 0) or 0
            m.cache_read_tokens += usage.get("cache_read_tokens", 0) or 0
            parent = e.get("parent_id")
            model = model_by_request_id.get(parent, "") if parent else ""
            if model:
                bucket = per_model_usage.setdefault(
                    model, {"input": 0, "output": 0, "cache_read": 0}
                )
                bucket["input"] += usage.get("input_tokens", 0) or 0
                bucket["output"] += usage.get("output_tokens", 0) or 0
                bucket["cache_read"] += usage.get("cache_read_tokens", 0) or 0

        elif event_type == "SupervisorVerdict":
            m.supervisor_verdict = e.get("verdict")

    # Latency from RunStarted to RunFinished timestamps
    if start_ts is not None and end_ts is not None:
        delta_ms = int((end_ts - start_ts).total_seconds() * 1000)
        m.total_latency_ms = delta_ms

    # Cache hit rate = cache_read_tokens / total_input_tokens
    if m.total_input_tokens > 0:
        m.cache_hit_rate = m.cache_read_tokens / m.total_input_tokens

    # Derive total cost from per-model token usage
    total_cost = 0.0
    for model, u in per_model_usage.items():
        total_cost += compute_cost_usd(
            model=model,
            input_tokens=u["input"],
            output_tokens=u["output"],
            cache_read_tokens=u["cache_read"],
        )
    m.cost_usd = round(total_cost, 6)

    return m
