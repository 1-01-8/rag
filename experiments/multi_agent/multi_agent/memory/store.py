"""File-based memory store (spec §5).

sticky.md: per-session running state, frontmatter = StickyContext fields.
turns/NNN-slug.md: per-turn record, frontmatter = Turn fields.
agent_notes/<slug>.md: cross-session learning, frontmatter = AgentNote fields.

Indexes (_index.json + MEMORY.md) are eagerly regenerated on every write.
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from multi_agent.schemas.memory import (
    StickyContext, Turn, AgentNote,
    EntityState, CitedArticle,
    StickyIntent, StickyEntitiesView, StickyCitationsView, StickySummaryView,
)


_INTENT_VALUES: frozenset[str] = frozenset(
    {"full", "entities_only", "recent_citations", "summary_only"}
)


def _slugify(text: str, max_len: int = 30) -> str:
    """Filesystem-safe slug from Chinese/ASCII text."""
    cleaned = re.sub(r"[\s/\\<>:\"|?*]+", "-", text.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        cleaned = "turn"
    return cleaned[:max_len]


def _atomic_write(path: Path, content: str) -> None:
    """Write file atomically via tmp+rename (POSIX)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _dump_frontmatter(data: dict) -> str:
    yaml_block = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{yaml_block}---\n\n"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    fm = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5:].lstrip()
    return fm, body


def _serialize_pydantic(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return _serialize_pydantic(obj.model_dump())
    if isinstance(obj, dict):
        return {k: _serialize_pydantic(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_pydantic(x) for x in obj]
    return obj


class MarkdownMemoryStore:
    """File-based memory store with eager index regeneration."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "sessions").mkdir(exist_ok=True)
        (self.root / "agent_notes").mkdir(exist_ok=True)

    # --- Sticky ---

    def _sticky_path(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id / "sticky.md"

    def _read_sticky_frontmatter(
        self, session_id: str
    ) -> tuple[dict, str] | None:
        """Robust sticky.md read.

        Returns (frontmatter_dict, body) on success, None when:
          - file does not exist
          - file cannot be read (permissions, decoding error)
          - YAML is malformed

        Never raises. Used by all read_sticky intent branches so file-level
        robustness lives in one place.
        """
        path = self._sticky_path(session_id)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            fm, body = _parse_frontmatter(text)
        except yaml.YAMLError:
            return None
        if not isinstance(fm, dict):
            # YAML may parse to a scalar/list/None if the frontmatter is
            # syntactically valid but semantically wrong. Treat as corrupt.
            return None
        return fm, body

    def read_sticky(
        self,
        session_id: str,
        intent: StickyIntent = "full",
    ) -> StickyContext | StickyEntitiesView | StickyCitationsView | StickySummaryView | None:
        """Read sticky.md, optionally returning only a slice (spec §5.4.3).

        intent semantics:
          - "full"             → StickyContext (default, backward-compatible)
          - "entities_only"    → StickyEntitiesView (EntityState + session_id)
          - "recent_citations" → StickyCitationsView (cited_articles list)
          - "summary_only"     → StickySummaryView (compressed history summary)

        Robustness:
          - file missing → None for every intent
          - corrupt YAML / unreadable file → None for every intent
          - missing field in an otherwise-valid sticky → empty view (NOT None);
            this lets callers distinguish "session has no entities yet" from
            "session does not exist"
          - corrupt individual list items (e.g. one malformed cited_article)
            are silently skipped; valid neighbours still returned
          - unknown intent string → ValueError (programming bug, not a data bug)
        """
        if intent not in _INTENT_VALUES:
            raise ValueError(
                f"Unknown intent {intent!r}; expected one of "
                f"{sorted(_INTENT_VALUES)}"
            )

        parsed = self._read_sticky_frontmatter(session_id)
        if parsed is None:
            return None
        fm, body = parsed
        sid = fm.get("session_id") or session_id

        if intent == "full":
            try:
                fm_with_body = {**fm, "body": body}
                return StickyContext.model_validate(fm_with_body)
            except Exception:
                # Validation failure on a present-but-malformed sticky.
                # Treat as corrupt: callers using read_sticky() get None
                # rather than a half-built object.
                return None

        if intent == "entities_only":
            raw = fm.get("entity_state")
            try:
                es = EntityState.model_validate(raw or {})
            except Exception:
                es = EntityState()
            return StickyEntitiesView(session_id=sid, entity_state=es)

        if intent == "recent_citations":
            raw_list = fm.get("cited_articles") or []
            cits: list[CitedArticle] = []
            if isinstance(raw_list, list):
                for item in raw_list:
                    try:
                        cits.append(CitedArticle.model_validate(item))
                    except Exception:
                        # Skip individual corrupt entry, keep the rest
                        continue
            return StickyCitationsView(session_id=sid, cited_articles=cits)

        # intent == "summary_only"
        summary = fm.get("history_summary") or ""
        if not isinstance(summary, str):
            summary = ""
        return StickySummaryView(session_id=sid, history_summary=summary)

    def write_sticky(self, sticky: StickyContext) -> Path:
        sticky.updated_at = datetime.now()
        path = self._sticky_path(sticky.session_id)
        data = _serialize_pydantic(sticky.model_dump(exclude={"body"}))
        text = _dump_frontmatter(data) + (sticky.body or "")
        _atomic_write(path, text)
        self._regenerate_index()
        return path

    # --- Turns ---

    def _turns_dir(self, session_id: str) -> Path:
        return self.root / "sessions" / session_id / "turns"

    def append_turn(self, session_id: str, turn: Turn) -> Path:
        td = self._turns_dir(session_id)
        td.mkdir(parents=True, exist_ok=True)
        slug = _slugify(turn.question)
        fname = f"{turn.turn:03d}-{slug}.md"
        path = td / fname
        data = _serialize_pydantic(turn.model_dump())
        body = (
            f"## 问题\n{turn.question}\n\n"
            f"## 答复\n{turn.final_answer}\n"
        )
        text = _dump_frontmatter(data) + body
        _atomic_write(path, text)
        self._regenerate_index()
        return path

    def recent_turns(self, session_id: str, n: int = 5) -> list[Turn]:
        td = self._turns_dir(session_id)
        if not td.exists():
            return []
        files = sorted(td.glob("*.md"), key=lambda p: p.name, reverse=True)[:n]
        out: list[Turn] = []
        for f in files:
            fm, _ = _parse_frontmatter(f.read_text(encoding="utf-8"))
            out.append(Turn.model_validate(fm))
        return out

    # --- Agent notes ---

    def write_note(self, note: AgentNote) -> Path:
        path = self.root / "agent_notes" / f"{note.name}.md"
        data = _serialize_pydantic(note.model_dump(exclude={"body"}))
        text = _dump_frontmatter(data) + (note.body or "")
        _atomic_write(path, text)
        self._regenerate_index()
        return path

    def find_notes(
        self,
        tags: list[str] | None = None,
        produced_by: str | None = None,
        about_agent: str | None = None,
        limit: int = 10,
    ) -> list[AgentNote]:
        notes_dir = self.root / "agent_notes"
        out: list[AgentNote] = []
        for f in sorted(notes_dir.glob("*.md")):
            fm, body = _parse_frontmatter(f.read_text(encoding="utf-8"))
            fm["body"] = body
            try:
                note = AgentNote.model_validate(fm)
            except Exception:
                continue
            if tags and not any(t in note.tags for t in tags):
                continue
            if produced_by and note.produced_by != produced_by:
                continue
            if about_agent and note.about_agent != about_agent:
                continue
            out.append(note)
            if len(out) >= limit:
                break
        return out

    # --- Index ---

    def _regenerate_index(self) -> None:
        index: dict[str, Any] = {
            "version": 1,
            "regenerated_at": datetime.now().isoformat(),
            "sessions": {},
            "notes_by_tag": {},
            "notes_by_about_agent": {},
            "notes_by_name": {},
        }
        sessions_dir = self.root / "sessions"
        if sessions_dir.exists():
            for session_dir in sorted(sessions_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                sticky_path = session_dir / "sticky.md"
                turns_dir = session_dir / "turns"
                turn_count = len(list(turns_dir.glob("*.md"))) if turns_dir.exists() else 0
                tags: list[str] = []
                linked_runs: list[str] = []
                if sticky_path.exists():
                    fm, _ = _parse_frontmatter(sticky_path.read_text(encoding="utf-8"))
                    cd = fm.get("case_type", "")
                    if cd:
                        tags.append(cd)
                    linked_runs = list(fm.get("linked_runs", []))
                index["sessions"][session_dir.name] = {
                    "path": str(sticky_path.relative_to(self.root)) if sticky_path.exists() else "",
                    "turn_count": turn_count,
                    "tags": tags,
                    "linked_runs": linked_runs,
                }
        notes_dir = self.root / "agent_notes"
        if notes_dir.exists():
            for note_path in sorted(notes_dir.glob("*.md")):
                fm, _ = _parse_frontmatter(note_path.read_text(encoding="utf-8"))
                name = fm.get("name", note_path.stem)
                index["notes_by_name"][name] = {
                    "path": str(note_path.relative_to(self.root)),
                    "produced_by": fm.get("produced_by", ""),
                    "about_agent": fm.get("about_agent", ""),
                    "usage_count": fm.get("usage_count", 0),
                }
                for tag in fm.get("tags", []):
                    index["notes_by_tag"].setdefault(tag, []).append(name)
                ag = fm.get("about_agent", "")
                if ag:
                    index["notes_by_about_agent"].setdefault(ag, []).append(name)
        _atomic_write(
            self.root / "_index.json",
            json.dumps(index, ensure_ascii=False, indent=2),
        )
        lines = ["<!-- AUTO-GENERATED. DO NOT EDIT. -->",
                 f"<!-- Last regenerated: {index['regenerated_at']} -->",
                 "", "## Sessions"]
        for sid, info in index["sessions"].items():
            tag_str = f" [{','.join(info['tags'])}]" if info["tags"] else ""
            lines.append(f"- [{sid}]({info['path']}){tag_str} — {info['turn_count']} turns")
        lines.append("\n## Agent Notes by Tag")
        for tag, names in sorted(index["notes_by_tag"].items()):
            lines.append(f"- **{tag}**: {', '.join(names)}")
        _atomic_write(self.root / "MEMORY.md", "\n".join(lines) + "\n")
