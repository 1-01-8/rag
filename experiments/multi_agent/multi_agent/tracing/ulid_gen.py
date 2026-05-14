from __future__ import annotations
from ulid import ULID


def fresh_event_id() -> str:
    """Monotonic 26-char ULID for trace events."""
    return str(ULID())


def fresh_run_id() -> str:
    """Run identifier with 'r_' prefix for visual recognition."""
    return f"r_{ULID()}"
