"""Cross-turn compaction (Spec §5.4.1).

maybe_compact() compresses old turns into StickyContext.history_summary when a
session exceeds COMPACTION_THRESHOLD turns.  Old turn files are NOT deleted
(traceability).
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.providers.base import LLMProvider
from multi_agent.schemas.messages import AgentMessage
from multi_agent.schemas.memory import Turn
from multi_agent.tracing.recorder import Recorder

COMPACTION_THRESHOLD = 5
KEEP_RECENT_TURNS = 3

COMPACT_SYSTEM_PROMPT = """你是法律咨询会话压缩器。
给你一段历史 turn 流水,请压缩成 200 字以内的中文 prose summary,
保留:
1. 用户问题主题与演进
2. Lawyer 提及的关键法条(law_short-article)
3. 已达成的结论或方向
不要列表式,不要标题。直接输出 summary。"""


def _format_turns(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        lines.append(f"## Turn {t.turn}")
        lines.append(f"Q: {t.question}")
        lines.append(f"A: {t.final_answer[:500]}")
    return "\n".join(lines)


async def maybe_compact(
    session_id: str,
    store: MarkdownMemoryStore,
    *,
    provider: LLMProvider,
    model: str,
) -> bool:
    """Compact old turns into history_summary if session exceeds threshold.

    Returns True if a compaction ran, False otherwise.
    Is idempotent: if history_summary already contains text, does nothing.
    """
    sticky = store.read_sticky(session_id)
    if sticky is None:
        return False
    if sticky.history_summary and sticky.history_summary.strip():
        return False    # already compacted; idempotent

    # recent_turns returns newest-first; reverse to get oldest-first
    all_turns = list(reversed(store.recent_turns(session_id, n=999)))
    if len(all_turns) < COMPACTION_THRESHOLD:
        return False

    # compact the oldest turns, keep the most recent KEEP_RECENT_TURNS intact
    to_compact = all_turns[:-KEEP_RECENT_TURNS]
    body = _format_turns(to_compact)

    messages = [
        AgentMessage(role="system", content=COMPACT_SYSTEM_PROMPT),
        AgentMessage(role="user", content=body),
    ]

    # Lightweight throwaway recorder (same pattern as judges/base.py)
    tmp = Path(tempfile.mkdtemp(prefix="compact_"))
    recorder = Recorder(run_id=f"compact-{session_id}", run_dir=tmp)
    try:
        resp = await provider.complete(
            messages,
            model=model,
            tools=None,
            temperature=0.0,
            max_tokens=512,
            recorder=recorder,
            agent_name="compactor",
        )
    finally:
        recorder.close()

    summary = (resp.text or "").strip()
    if not summary:
        return False

    sticky.history_summary = summary
    sticky.updated_at = datetime.now(timezone.utc)
    store.write_sticky(sticky)
    return True
