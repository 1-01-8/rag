from __future__ import annotations
from datetime import datetime
from typing import Literal, Any
from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Common fields for every trace event."""
    event_id: str
    run_id: str
    timestamp: datetime
    parent_id: str | None = None
    event_type: str  # subclasses override with Literal

    model_config = {"frozen": False, "extra": "forbid"}


class RunStarted(BaseEvent):
    event_type: Literal["RunStarted"] = "RunStarted"
    query: str
    config: dict[str, Any] = Field(default_factory=dict)


class RunFinished(BaseEvent):
    event_type: Literal["RunFinished"] = "RunFinished"
    status: Literal["ok", "error", "timeout"]
    final_answer: str | None = None
    error: str | None = None
