# Phase 3 — Receptionist + Markdown Memory Store + Multi-Issue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan.

**Goal:** Add the file-based `MarkdownMemoryStore` (sticky/turns/agent_notes per spec §5) + `ReceptionistAgent` (triage + safety + multi-issue decomposition per spec §3.5/§3.6). Wire Receptionist → Lawyer pipeline through memory: Receptionist classifies + extracts EntityState, then Lawyer reads sticky context and handles each sub_case in turn.

**Architecture:** New `memory/` subpackage holds the MD-based store (no DB). Receptionist is a new lightweight agent — outputs `ReceptionistOutput` with `sub_cases` per spec §3.6. Lawyer base class gains `pre_react_hook` so sub-classes can inject memory context into messages without rewriting `_react_loop`. Multi-issue V1: Receptionist outputs sub_cases, Lawyer processes them sequentially (per ADR-17), V2 will fan-out.

**Spec reference:** §3.5 (Receptionist), §3.6 (Multi-Issue), §5 (Memory MD), ADR-17, ADR-23.

**Phase 2d starting point:** Tag `phase2d-cases-collection`. 144 tests pass + 1 skipped.

---

## Out of scope (defer to later)

- `ma_user_history` Qdrant collection (Phase 3b — needs memory_store to be populated first)
- Cross-turn compression (Phase 3b — `history_summary` field design)
- WorkingMemory threading into agents (Phase 3b — schema already exists from Phase 1)
- `find_notes` API on memory store (Phase 4 Supervisor will write agent_notes)
- Conversational follow-ups across multiple turns in real-time UI (out of scope — file-based persistence only)

---

## File Structure (Phase 3 additions)

```
experiments/multi_agent/
├── multi_agent/
│   ├── schemas/
│   │   ├── memory.py                       # NEW: StickyContext + EntityState + Turn + AgentNote
│   │   └── receptionist.py                 # NEW: ReceptionistOutput + SubCase
│   ├── memory/
│   │   ├── __init__.py
│   │   └── store.py                        # NEW: MarkdownMemoryStore (sticky/turns/notes + indexes)
│   └── agents/
│       ├── receptionist.py                 # NEW: ReceptionistAgent
│       └── lawyer.py                       # MODIFY: pre_react_hook + sub_cases handling
├── prompts/
│   └── receptionist/
│       ├── __init__.py
│       └── system.md                       # NEW
└── tests/
    ├── unit/
    │   ├── test_memory_schemas.py
    │   ├── test_memory_store.py
    │   ├── test_receptionist_output.py
    │   ├── test_receptionist.py
    │   └── test_lawyer_multi_issue.py
    └── integration/
        └── test_receptionist_lawyer_e2e.py
```

**Working directory:** `/home/xxm/rag/experiments/multi_agent/`
**Test command:** `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && <pytest>"`

---

## Task 1: Memory Schemas

**Files:**
- Create: `multi_agent/schemas/memory.py`
- Create: `tests/unit/test_memory_schemas.py`

Spec §5.4: sticky.md has EntityState. Spec §5.5: turns/NNN.md links to run_id. Spec §5.6: agent_notes have produced_by/about_agent/tags.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_memory_schemas.py
import pytest
from datetime import datetime
from multi_agent.schemas.memory import (
    EntityState, ActiveSubject, KeyFact, RejectedPath,
    StickyContext, Turn, AgentNote,
)


def test_entity_state_minimal():
    es = EntityState()
    assert es.active_subjects == []
    assert es.key_facts == []
    assert es.open_questions == []
    assert es.rejected_paths == []
    assert es.legal_objectives == []


def test_entity_state_full():
    es = EntityState(
        active_subjects=[
            ActiveSubject(role="原告", identifier="用户", attributes=["房屋承租人"]),
        ],
        key_facts=[KeyFact(fact="租期1年", confidence="high", source_turn=1)],
        rejected_paths=[RejectedPath(path="走刑事路径", reason="未涉及胁迫")],
    )
    assert es.active_subjects[0].role == "原告"
    assert es.key_facts[0].fact == "租期1年"


def test_sticky_context_required_fields():
    s = StickyContext(
        session_id="s_test_2026-05-14",
        legal_domain="民事",
        case_type="租赁纠纷",
        last_law_name="民法典",
    )
    assert s.session_id.startswith("s_")
    assert s.mentioned_laws == []
    assert s.cited_articles == []
    assert s.linked_runs == []
    assert isinstance(s.entity_state, EntityState)


def test_turn_record():
    t = Turn(
        turn=1, run_id="r_abc",
        started_at=datetime(2026, 5, 14, 14, 0),
        finished_at=datetime(2026, 5, 14, 14, 1),
        question="房东涨租?", final_answer='{"answer": "..."}',
        answer_mode="evidence_grounded",
        agents_invoked=["receptionist", "lawyer"],
    )
    assert t.duration_ms == 60000


def test_agent_note():
    n = AgentNote(
        name="lawyer-misses-rental-mgmt-rules",
        description="涨租漏引租赁管理办法",
        produced_by="supervisor", about_agent="lawyer",
        tags=["涨租", "民法典-510"], triggered_by_run="r_abc",
    )
    assert n.usage_count == 0
    assert "涨租" in n.tags
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/memory.py`**

```python
"""Schemas for the file-based memory store (spec §5)."""
from __future__ import annotations
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


# --- EntityState components (spec §5.4 frontmatter) ---

class ActiveSubject(BaseModel):
    role: str                                          # e.g. "原告"
    identifier: str                                    # e.g. "用户" / "房东"
    attributes: list[str] = Field(default_factory=list)


class KeyFact(BaseModel):
    fact: str
    confidence: Literal["low", "medium", "high"] = "high"
    source_turn: int = 0


class RejectedPath(BaseModel):
    path: str                                          # e.g. "走刑事路径"
    reason: str


class EntityState(BaseModel):
    """Structured facts extracted from a session (spec §5.4)."""
    active_subjects: list[ActiveSubject] = Field(default_factory=list)
    key_facts: list[KeyFact] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    rejected_paths: list[RejectedPath] = Field(default_factory=list)
    legal_objectives: list[str] = Field(default_factory=list)


# --- Top-level memory artifacts ---

class CitedArticle(BaseModel):
    law: str                                           # e.g. "民法典"
    article: str                                       # e.g. "510"
    from_turn: int = 0


class StickyContext(BaseModel):
    """Running session state — sticky.md frontmatter (spec §5.4)."""
    session_id: str                                    # e.g. "s_abc_2026-05-14"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    legal_domain: str = ""                             # 民事 / 刑事 / ...
    case_type: str = ""
    last_law_name: str = ""
    mentioned_laws: list[str] = Field(default_factory=list)
    cited_articles: list[CitedArticle] = Field(default_factory=list)
    linked_runs: list[str] = Field(default_factory=list)
    entity_state: EntityState = Field(default_factory=EntityState)
    history_summary: str = ""                          # Phase 3b: compaction populates this
    body: str = ""                                     # human-readable narrative


class Turn(BaseModel):
    """One conversation turn — turns/NNN-slug.md (spec §5.5)."""
    turn: int
    run_id: str
    started_at: datetime
    finished_at: datetime
    question: str
    final_answer: str
    answer_mode: str = "evidence_grounded"
    supervisor_verdict: str = ""                       # Phase 4 populates
    agents_invoked: list[str] = Field(default_factory=list)
    total_tokens: int = 0
    citations: list[CitedArticle] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return int((self.finished_at - self.started_at).total_seconds() * 1000)


class AgentNote(BaseModel):
    """Cross-session learning — agent_notes/<slug>.md (spec §5.6)."""
    name: str                                          # slug, also filename
    description: str
    produced_by: str                                   # which agent wrote it
    about_agent: str                                   # which agent it's about
    verdict_that_triggered: str = ""                   # "reject" / "revise" / ""
    tags: list[str] = Field(default_factory=list)
    triggered_by_run: str = ""
    used_in_runs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    usage_count: int = 0
    body: str = ""                                     # markdown body
```

- [ ] **Step 4: Verify pass** → 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/memory.py experiments/multi_agent/tests/unit/test_memory_schemas.py
git commit -m "phase3(schemas): memory schemas (StickyContext + EntityState + Turn + AgentNote)"
```

---

## Task 2: MarkdownMemoryStore (Sticky + Turns)

**Files:**
- Create: `multi_agent/memory/__init__.py`
- Create: `multi_agent/memory/store.py`
- Create: `tests/unit/test_memory_store.py`

Implements the file-based store. Frontmatter via PyYAML (already in deps via pydantic). Sticky atomic write; turns numbered NNN with slug.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_memory_store.py
import pytest
from datetime import datetime
from pathlib import Path
from multi_agent.memory.store import MarkdownMemoryStore
from multi_agent.schemas.memory import (
    StickyContext, EntityState, ActiveSubject, KeyFact,
    Turn, AgentNote,
)


@pytest.fixture
def store(tmp_path):
    return MarkdownMemoryStore(root=tmp_path / "memory_store")


def test_sticky_initially_absent(store):
    assert store.read_sticky("s_test") is None


def test_sticky_write_and_read_roundtrip(store):
    s = StickyContext(
        session_id="s_test_2026", legal_domain="民事", case_type="租赁",
        last_law_name="民法典", mentioned_laws=["民法典", "商品房屋租赁管理办法"],
        entity_state=EntityState(
            active_subjects=[ActiveSubject(role="原告", identifier="用户")],
            key_facts=[KeyFact(fact="租期1年", confidence="high", source_turn=1)],
        ),
        body="租房涨租案",
    )
    store.write_sticky(s)
    loaded = store.read_sticky("s_test_2026")
    assert loaded is not None
    assert loaded.legal_domain == "民事"
    assert loaded.mentioned_laws == ["民法典", "商品房屋租赁管理办法"]
    assert loaded.entity_state.active_subjects[0].role == "原告"
    assert loaded.body == "租房涨租案"


def test_append_turn_creates_numbered_file(store):
    s = StickyContext(session_id="s_x")
    store.write_sticky(s)
    t = Turn(
        turn=1, run_id="r_001",
        started_at=datetime(2026, 5, 14, 14, 0),
        finished_at=datetime(2026, 5, 14, 14, 1),
        question="房东涨租?", final_answer='{"answer": "..."}',
        agents_invoked=["lawyer"],
    )
    path = store.append_turn("s_x", t)
    assert path.exists()
    assert "001" in path.name


def test_recent_turns_sorted_descending(store):
    s = StickyContext(session_id="s_y")
    store.write_sticky(s)
    for i in range(1, 4):
        store.append_turn("s_y", Turn(
            turn=i, run_id=f"r_{i:03d}",
            started_at=datetime(2026, 5, 14, 14, 0),
            finished_at=datetime(2026, 5, 14, 14, i),
            question=f"q{i}", final_answer=f"a{i}",
        ))
    recent = store.recent_turns("s_y", n=2)
    assert len(recent) == 2
    assert recent[0].turn == 3
    assert recent[1].turn == 2


def test_agent_note_write_and_find(store):
    note = AgentNote(
        name="test-note", description="x",
        produced_by="supervisor", about_agent="lawyer",
        tags=["涨租", "民法典-510"],
        triggered_by_run="r_abc",
    )
    store.write_note(note)
    found = store.find_notes(tags=["涨租"])
    assert len(found) == 1
    assert found[0].name == "test-note"
    # Tag miss
    assert store.find_notes(tags=["unrelated_tag"]) == []


def test_index_regenerated_after_writes(store):
    s = StickyContext(session_id="s_z")
    store.write_sticky(s)
    store.append_turn("s_z", Turn(
        turn=1, run_id="r_x",
        started_at=datetime.now(), finished_at=datetime.now(),
        question="q", final_answer="a",
    ))
    import json as _j
    index = _j.loads((store.root / "_index.json").read_text(encoding="utf-8"))
    assert "s_z" in index["sessions"]
    assert index["sessions"]["s_z"]["turn_count"] == 1
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/memory/__init__.py` and `multi_agent/memory/store.py`**

```python
# multi_agent/memory/__init__.py
```

```python
# multi_agent/memory/store.py
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
from multi_agent.schemas.memory import StickyContext, Turn, AgentNote


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
    """Render frontmatter block."""
    yaml_block = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{yaml_block}---\n\n"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown doc into (frontmatter_dict, body)."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    fm = yaml.safe_load(text[4:end]) or {}
    body = text[end + 5:].lstrip()
    return fm, body


def _serialize_pydantic(obj: Any) -> Any:
    """Convert datetimes / models to YAML-friendly values."""
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

    def read_sticky(self, session_id: str) -> StickyContext | None:
        path = self._sticky_path(session_id)
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(text)
        fm["body"] = body
        return StickyContext.model_validate(fm)

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
        self, tags: list[str] | None = None,
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
        """Eagerly rebuild _index.json and MEMORY.md."""
        index: dict[str, Any] = {
            "version": 1,
            "regenerated_at": datetime.now().isoformat(),
            "sessions": {},
            "notes_by_tag": {},
            "notes_by_about_agent": {},
            "notes_by_name": {},
        }
        # Sessions
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
        # Notes
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
        # MEMORY.md (human-readable)
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
```

- [ ] **Step 4: Verify pass** → 6 passed.

Full suite check: `pytest -v 2>&1 | tail -5` → 155 passed + 1 skipped (144 + 11 new across Tasks 1+2).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/memory/ experiments/multi_agent/tests/unit/test_memory_store.py
git commit -m "phase3(memory): MarkdownMemoryStore with sticky/turns/notes + eager index"
```

---

## Task 3: ReceptionistOutput Schema

**Files:**
- Create: `multi_agent/schemas/receptionist.py`
- Create: `tests/unit/test_receptionist_output.py`

Per spec §3.5/§3.6: SubCase + ReceptionistOutput. Used by Receptionist to declare specialty routing + multi-issue decomposition.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_receptionist_output.py
import pytest
from multi_agent.schemas.receptionist import ReceptionistOutput, SubCase


def test_subcase_required_fields():
    sc = SubCase(
        issue="责任归属确认",
        specialty="侵权",
        priority=1,
    )
    assert sc.priority == 1
    assert sc.requires_separate_retrieval is True   # default


def test_receptionist_output_single_issue():
    out = ReceptionistOutput(
        primary_specialty="民事",
        case_type="租赁纠纷",
        urgency="中",
        is_multi_issue=False,
        sub_cases=[],
        initial_facts=["合同期一年", "涨幅30%"],
        normalized_query="房东合同期内涨租 30% 是否合法",
    )
    assert out.primary_specialty == "民事"
    assert out.is_multi_issue is False
    assert out.risk_flag is None


def test_receptionist_output_multi_issue():
    out = ReceptionistOutput(
        primary_specialty="家事",
        case_type="离婚+保护令",
        urgency="高",
        is_multi_issue=True,
        sub_cases=[
            SubCase(issue="离婚诉讼", specialty="家事", priority=2),
            SubCase(issue="人身安全保护令", specialty="治安", priority=1),
        ],
        initial_facts=["原告不想出庭", "被告威胁"],
        normalized_query="离婚诉讼 + 人身保护令",
    )
    assert out.is_multi_issue is True
    assert len(out.sub_cases) == 2
    assert out.sub_cases[1].issue == "人身安全保护令"


def test_receptionist_output_safety_refusal():
    out = ReceptionistOutput(
        primary_specialty="(safety)",
        case_type="safety_refusal",
        urgency="高",
        risk_flag="safety_refusal",
        is_multi_issue=False, sub_cases=[],
        initial_facts=[], normalized_query="",
    )
    assert out.risk_flag == "safety_refusal"


def test_receptionist_output_rejects_unknown_urgency():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ReceptionistOutput(
            primary_specialty="x", case_type="y", urgency="bogus",
            is_multi_issue=False, sub_cases=[],
            initial_facts=[], normalized_query="",
        )
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/receptionist.py`**

```python
"""Receptionist output schema (spec §3.5)."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class SubCase(BaseModel):
    issue: str                                       # human description
    specialty: str                                   # 民事 / 家事 / 治安 / ...
    priority: int = 1                                # 1=must-answer, 2=secondary, 3=optional
    requires_separate_retrieval: bool = True


class ReceptionistOutput(BaseModel):
    """Triage + decomposition output (spec §3.5)."""
    primary_specialty: str
    case_type: str
    urgency: Literal["低", "中", "高"]
    is_multi_issue: bool = False
    sub_cases: list[SubCase] = Field(default_factory=list)
    initial_facts: list[str] = Field(default_factory=list)
    normalized_query: str = ""
    need_clarification: bool = False
    clarification_q: str | None = None
    risk_flag: str | None = None                     # "safety_refusal" / "hi_risk_consult" / None
```

- [ ] **Step 4: Verify pass** → 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/receptionist.py experiments/multi_agent/tests/unit/test_receptionist_output.py
git commit -m "phase3(schemas): ReceptionistOutput + SubCase (multi-issue support)"
```

---

## Task 4: ReceptionistAgent + system prompt

**Files:**
- Create: `multi_agent/prompts/receptionist/__init__.py`
- Create: `multi_agent/prompts/receptionist/system.md`
- Create: `multi_agent/agents/receptionist.py`
- Create: `tests/unit/test_receptionist.py`

ReceptionistAgent is tool-less (Phase 3 V1 — Phase 4 might add safety_check tool). Outputs `ReceptionistOutput`. Phase 2c's tool-first enforcement bypasses cleanly because `tool_specs` will be empty.

- [ ] **Step 1: Create prompt files**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/receptionist
touch /home/xxm/rag/experiments/multi_agent/multi_agent/prompts/receptionist/__init__.py
```

Write `/home/xxm/rag/experiments/multi_agent/multi_agent/prompts/receptionist/system.md`:

```markdown
你是法律咨询的接待员(分诊员)。你的工作:

1. 理解用户咨询的法律领域
2. 提取关键事实(prior_facts)
3. 判断是否包含多个独立法律问题(multi_issue)
4. 输出 JSON 决策

# 输出 JSON 格式

```json
{
  "primary_specialty": "民事|劳动|交通|婚姻|房产|家事|治安|通用",
  "case_type": "<简短描述,如'租赁纠纷'>",
  "urgency": "低|中|高",
  "is_multi_issue": false,
  "sub_cases": [
    {"issue": "<子问题>", "specialty": "<专业>", "priority": 1, "requires_separate_retrieval": true}
  ],
  "initial_facts": ["<事实1>", "<事实2>"],
  "normalized_query": "<消解代词后的查询>",
  "need_clarification": false,
  "clarification_q": null,
  "risk_flag": null
}
```

# 多议题判断标准
- 用户一次咨询包含 ≥2 个相互独立的法律问题 → is_multi_issue=true
- 每个 sub_case 应明确写出 issue 描述、对应 specialty、优先级
- 单一议题时 sub_cases 留空 []

# 风险标记
- 若用户在询问犯罪手法/暴力威胁等 → risk_flag="safety_refusal"
- 若涉及高风险但合法咨询(如自杀危机、未成年人受害) → risk_flag="hi_risk_consult"

# 输出约束
- 只输出 JSON,不输出其他文字
- urgency 必须是"低""中""高"三选一
```

- [ ] **Step 2: Update `pyproject.toml` package-data**

Find `[tool.setuptools.package-data]` and update:

```toml
[tool.setuptools.package-data]
multi_agent = ["prompts/lawyer/*.md", "prompts/receptionist/*.md"]
```

Reinstall: `conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pip install -e '.[dev]' 2>&1 | tail -3"`.

- [ ] **Step 3: Failing test**

```python
# tests/unit/test_receptionist.py
import pytest
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_receptionist_prompt_loads(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec)
    prompt = agent.system_prompt()
    assert "接待员" in prompt or "分诊员" in prompt
    assert "is_multi_issue" in prompt
    rec.close()


def test_receptionist_output_schema_is_correct(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec)
    assert agent.output_schema() is ReceptionistOutput
    rec.close()


@pytest.mark.asyncio
async def test_receptionist_runs_to_output(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[
        ScriptedResponse(
            text='{"primary_specialty": "民事", "case_type": "租赁", "urgency": "中",'
                 ' "is_multi_issue": false, "sub_cases": [],'
                 ' "initial_facts": ["合同一年"], "normalized_query": "涨租",'
                 ' "need_clarification": false, "clarification_q": null, "risk_flag": null}',
            finish_reason="end_turn",
        ),
    ])
    agent = ReceptionistAgent(name="receptionist", role="triage",
                             provider=p, recorder=rec, model="stub-1")
    out = await agent.run(AgentInput(payload={"query": "房东涨租"}))
    rec.close()
    assert isinstance(out.payload, ReceptionistOutput)
    assert out.payload.primary_specialty == "民事"
    assert out.payload.urgency == "中"
```

- [ ] **Step 4: Verify failure** → ImportError.

- [ ] **Step 5: Create `multi_agent/agents/receptionist.py`**

```python
"""ReceptionistAgent — triage + multi-issue decomposition (spec §3.5)."""
from __future__ import annotations
from importlib.resources import files

from multi_agent.agents.base import BaseAgent
from multi_agent.schemas.receptionist import ReceptionistOutput


class ReceptionistAgent(BaseAgent):
    """Tool-less classifier. Reads user query, outputs ReceptionistOutput."""

    def system_prompt(self) -> str:
        return files("multi_agent.prompts.receptionist").joinpath("system.md").read_text(encoding="utf-8")

    def output_schema(self) -> type[ReceptionistOutput]:
        return ReceptionistOutput
```

- [ ] **Step 6: Verify pass + full suite** → 162 passed + 1 skipped (155 + 7 new).

- [ ] **Step 7: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/prompts/receptionist/ experiments/multi_agent/multi_agent/agents/receptionist.py experiments/multi_agent/tests/unit/test_receptionist.py experiments/multi_agent/pyproject.toml
git commit -m "phase3(agents): ReceptionistAgent (tool-less) + system prompt + package data"
```

---

## Task 5: Multi-Issue Lawyer (V1 Sequential)

**Files:**
- Modify: `multi_agent/agents/lawyer.py` — accept optional `sub_cases` in input payload; if present, process each sequentially and merge
- Create: `tests/unit/test_lawyer_multi_issue.py`

V1: simple sequential processing — for each sub_case, run a sub-query through the existing ReAct loop with sub_case.issue as the focused query. Merge results into one FiveSection with combined citations.

Actually, even simpler: Phase 3 V1 just makes the Lawyer AWARE of sub_cases (passes them in the prompt context). The actual sequential fan-out is Phase 3b's concern.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_lawyer_multi_issue.py
import pytest
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.schemas.receptionist import SubCase
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.agents.base import AgentInput
from multi_agent.tracing.recorder import Recorder


def test_lawyer_accepts_sub_cases_in_input(tmp_path):
    rec = Recorder(run_id="r1", run_dir=tmp_path / "r")
    p = StubProvider(responses=[])
    lawyer = LawyerAgent(name="lawyer", role="advisor",
                       provider=p, recorder=rec)
    # Sub_cases in payload should not crash construction
    input = AgentInput(payload={
        "query": "test",
        "sub_cases": [
            SubCase(issue="A", specialty="民事", priority=1).model_dump(),
            SubCase(issue="B", specialty="家事", priority=2).model_dump(),
        ],
    })
    # Render the input into a string the lawyer can see
    rendered = lawyer.render_input(input)
    assert "sub_cases" in rendered or "子议题" in rendered
    rec.close()
```

- [ ] **Step 2: Verify failure** → AttributeError on `render_input`.

- [ ] **Step 3: Add `render_input` to `LawyerAgent`**

In `multi_agent/agents/lawyer.py`, add a method:

```python
    def render_input(self, input):
        """Render AgentInput into the user-message text the LLM will see.

        If sub_cases present (multi-issue case), inject them as a numbered list
        so the Lawyer addresses each.
        """
        payload = input.payload
        query = str(payload.get("query", ""))
        sub_cases = payload.get("sub_cases", [])
        if not sub_cases:
            return query
        # Multi-issue case: render sub_cases below the main query
        lines = [f"用户咨询: {query}", "", "本案包含以下独立子议题(请逐一回答):"]
        for i, sc in enumerate(sub_cases, 1):
            issue = sc.get("issue", "") if isinstance(sc, dict) else sc.issue
            specialty = sc.get("specialty", "") if isinstance(sc, dict) else sc.specialty
            lines.append(f"{i}. [{specialty}] {issue}")
        return "\n".join(lines)
```

Also modify `BaseAgent._react_loop` so it calls `self.render_input(input)` if defined, else falls back to current logic. Easiest:

In `agents/base.py`, find this line in `_react_loop`:

```python
            AgentMessage(role="user", content=str(input.payload.get("query", input.payload))),
```

Change to:

```python
            AgentMessage(role="user", content=self._render_input(input)),
```

And add a method to `BaseAgent`:

```python
    def _render_input(self, input: AgentInput) -> str:
        """Default: just the query text. Subclasses can override."""
        return str(input.payload.get("query", input.payload))
```

In `LawyerAgent` override `_render_input` (rename `render_input` if you want consistency, but the test uses `render_input` — use that name). Actually, to keep the test simple, expose BOTH: `render_input` (public) and `_render_input` (internal). Have `_render_input` call `render_input`.

Or: simpler — rename the public method to `_render_input` in both places. Update the test to use `_render_input`.

Let me prescribe: rename test to use `_render_input` (single underscore is the convention for "package-internal API"). The test becomes:

```python
def test_lawyer_accepts_sub_cases_in_input(tmp_path):
    ...
    rendered = lawyer._render_input(input)
    assert "子议题" in rendered or "sub_cases" in rendered
```

And `LawyerAgent` only adds `_render_input(self, input)`. `BaseAgent` has the default `_render_input`.

- [ ] **Step 4: Verify pass + full suite** → 163 passed + 1 skipped (162 + 1 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/base.py experiments/multi_agent/multi_agent/agents/lawyer.py experiments/multi_agent/tests/unit/test_lawyer_multi_issue.py
git commit -m "phase3(agents): _render_input hook + Lawyer renders sub_cases for multi-issue"
```

---

## Task 6: Receptionist → Lawyer Integration E2E (Real Qwen)

**Files:**
- Create: `tests/integration/test_receptionist_lawyer_e2e.py`

Flagship Phase 3 test: real Qwen drives Receptionist on a multi-issue prompt; the output's sub_cases are fed into the Lawyer alongside indexed retrieval.

- [ ] **Step 1: Write test**

```python
# tests/integration/test_receptionist_lawyer_e2e.py
"""Phase 3 flagship: Receptionist classifies + decomposes, then Lawyer handles
the case (multi-issue if applicable). Real Qwen, real Qdrant."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.receptionist import ReceptionistOutput
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.receptionist import ReceptionistAgent
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.agents.base import AgentInput
from multi_agent.runner import run_query
from multi_agent.tracing.recorder import Recorder
from multi_agent.tracing.ulid_gen import fresh_run_id


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def statute_index(tmp_path_factory):
    name = f"test_r2l_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照合同相关条款或者交易习惯确定。"),
            Chunk(doc_id="民法典-703", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="703",
                  text="租赁合同是出租人将租赁物交付承租人使用、收益,承租人支付租金的合同。"),
        ],
    )]
    build_index(documents=docs, collection_name=name,
                sparse_artifact_path=sparse_path, dense_encoder=DenseEncoder())
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_receptionist_then_lawyer(statute_index, tmp_path):
    """Run Receptionist first, then Lawyer with its sub_cases."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True)
    provider = OpenAICompatibleProvider()

    # Step 1: Receptionist
    rec_run_id = fresh_run_id()
    rec_recorder = Recorder(run_id=rec_run_id, run_dir=runs_root / rec_run_id)
    receptionist = ReceptionistAgent(
        name="receptionist", role="triage",
        provider=provider, recorder=rec_recorder,
        model="qwen3.5-9b", max_steps=2,
    )
    triage_input = AgentInput(payload={"query": "我租的房合同一年,房东要涨 30% 房租,合法吗?"})
    triage_out = await receptionist.run(triage_input)
    rec_recorder.close()
    assert isinstance(triage_out.payload, ReceptionistOutput)
    assert triage_out.payload.primary_specialty in ("民事", "房产", "通用")

    # Step 2: Lawyer with sub_cases from receptionist
    statute_search = StatuteSearchTool(
        collection_name=statute_index["collection"],
        sparse_artifact_path=statute_index["sparse_path"],
    )

    result = await run_query(
        query="我租的房合同一年,房东要涨 30% 房租,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search],
            model="qwen3.5-9b",
            specialty=triage_out.payload.primary_specialty if triage_out.payload.primary_specialty in ("民事", "房产") else "民事",
            max_steps=8, max_tool_calls=10,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "3", "received_from_receptionist": triage_out.payload.model_dump()},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"
    # No fabricated citations
    indexed = {"民法典-510", "民法典-703"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated: {doc_id}"
```

- [ ] **Step 2: Run + Step 3: Full suite + Step 4: Commit + tag**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_receptionist_lawyer_e2e.py -v"
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -5"

cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_receptionist_lawyer_e2e.py
git commit -m "phase3(integration): Receptionist → Lawyer pipeline E2E with real Qwen"
git tag -a phase3-receptionist-memory -m "Phase 3 complete: Receptionist + MarkdownMemoryStore + multi-issue rendering"
git tag -l "phase*"
```

Expected: full suite ~164 passed + 1 skipped.

## Acceptance Criteria

Phase 3 complete when:

1. Full pytest passes
2. `MarkdownMemoryStore` correctly reads/writes sticky + turns + agent_notes
3. `_index.json` and `MEMORY.md` regenerated on every write
4. `ReceptionistAgent` produces valid `ReceptionistOutput` against real Qwen
5. `LawyerAgent` renders `sub_cases` from input payload
6. Real-Qwen E2E proves Receptionist → Lawyer pipeline works without fabrication
7. Tag `phase3-receptionist-memory` exists

## Out of Scope (Reminder for Phase 3b/4)

- Sequential fan-out for multi-issue (V2 — current V1 just renders sub_cases in prompt)
- Cross-turn compression (>5 turns → `history_summary`)
- `WorkingMemory` threaded through agents
- `ma_user_history` Qdrant collection + `history_search` tool
- agent_notes write-by-Supervisor (Phase 4)
- Receptionist `safety_check` tool
