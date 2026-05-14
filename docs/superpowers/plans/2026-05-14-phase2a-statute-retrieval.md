# Phase 2a — Statute Retrieval (Qdrant + bge-m3 + jieba) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index the 177-law Chinese statute corpus into Qdrant with dense (bge-m3) + sparse (jieba+IDF) vectors and expose `statute_search` / `read_article` as Tools that any Phase 1 agent can call. Plus a small but critical Phase 2 prep fix: convert the trace span stack to `contextvars` so concurrent async I/O won't corrupt the parent_id chain.

**Architecture:** Builds on Phase 1 walking skeleton. New code is additive — `multi_agent/tools/retrievers/` package + corpus loader + index-builder script. Hybrid search uses Qdrant native RRF fusion via `query_points(prefetch=..., query=FusionQuery(Fusion.RRF))` — no manual fusion code.

**Qdrant deployment:** **Reuse the existing `legal-rag-qdrant` container** that's already running (`qdrant/qdrant:v1.12.4`, healthy, host ports `6433:6333` and `6434:6334`). The default Qdrant port 6333 on the host is already taken by another service; this project connects to `http://localhost:6433`. We do NOT create a new docker-compose service.

**Collection naming:** All multi_agent collections are prefixed `ma_` to coexist with the existing `legal_statutes` collection (used by legacy `legal_rag/`). Phase 2a creates `ma_statutes`.

**Tech Stack:** Python 3.10+, Pydantic 2.x, qdrant-client, sentence-transformers (bge-m3), jieba, numpy, pytest-asyncio. Optional GPU for faster index build; CPU works (slower).

**Spec reference:** `/home/xxm/rag/docs/superpowers/specs/2026-05-14-multi-agent-experiment-design.md` §4 (Tool / Retriever 层), §4.4 (Chunking), ADR-19 (concurrent span stack ContextVar).

---

## What this plan does NOT cover

Out of scope (later sub-plans of Phase 2):
- **Phase 2b**: Real LLM providers (Anthropic / OpenAI-compatible Qwen) — still using StubProvider here
- **Phase 2c**: Real Lawyer agent with five-section prompt — still using EchoStubAgent
- **Phase 2d**: cases / user_history collections — only statutes collection here
- Concepts field generation (spec §4.4 says "实验性,可选", needs local Qwen → Phase 2b dependency)
- book/chapter章节信息 (spec §4.4 方案 B — skipped in V0)

---

## File Structure (Phase 2a additions)

```
experiments/multi_agent/
├── pyproject.toml                                  # Task 1 — add deps (no docker-compose; reuse legal-rag-qdrant)
├── multi_agent/
│   ├── tracing/recorder.py                         # Task 0 — MODIFY: ContextVar for span stack
│   ├── schemas/
│   │   └── document.py                             # Task 2 — Document + Chunk schemas
│   └── tools/
│       ├── corpus.py                               # Task 3 — Corpus loader from Chinese-Laws
│       └── retrievers/
│           ├── __init__.py
│           ├── qdrant_client.py                    # Task 6 — Qdrant connection + collection mgmt
│           ├── dense_encoder.py                    # Task 4 — bge-m3 wrapper
│           ├── sparse_encoder.py                   # Task 5 — jieba + IDF
│           ├── index_builder.py                    # Task 7 — encode + upsert pipeline
│           ├── statute_search.py                   # Task 8 — search tool
│           └── exact_read.py                       # Task 9 — read by doc_id tool
├── scripts/
│   └── build_statutes_index.py                     # Task 7 — CLI entry
└── tests/
    ├── unit/
    │   ├── test_recorder_contextvars.py            # Task 0
    │   ├── test_document.py                        # Task 2
    │   ├── test_corpus.py                          # Task 3
    │   ├── test_dense_encoder.py                   # Task 4
    │   ├── test_sparse_encoder.py                  # Task 5
    │   ├── test_qdrant_client.py                   # Task 6
    │   ├── test_index_builder.py                   # Task 7
    │   ├── test_statute_search.py                  # Task 8
    │   └── test_exact_read.py                      # Task 9
    └── integration/
        └── test_retrieval_e2e.py                   # Task 10
```

**Working directory for all tasks:** `/home/xxm/rag/experiments/multi_agent/`

**Prerequisites:**
- The `legal-rag-qdrant` container is running and healthy on host ports 6433/6434. Verify with `docker ps | grep legal-rag-qdrant` and `curl -s http://localhost:6433/healthz`. If down, restart with `docker start legal-rag-qdrant`.
- Phase 1 tag `phase1-walking-skeleton` is the starting point.

---

## Task 0: ContextVar Fix for Span Stack

**Files:**
- Modify: `multi_agent/tracing/recorder.py`
- Create: `tests/unit/test_recorder_contextvars.py`

**Why:** Phase 1 final review flagged that `Recorder._span_stack` is a plain `list[str]`. Once concurrent `await` points exist (Phase 2b real providers), two coroutines pushing/popping the same list will corrupt the parent_id chain. Fix: `contextvars.ContextVar[tuple[str, ...]]` — each coroutine gets its own logical view of the stack.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_recorder_contextvars.py
"""Verify span stack is async-task-local. Without this, concurrent
spans on the same recorder corrupt the parent_id chain.
"""
import asyncio
import json
import pytest
from multi_agent.tracing.recorder import Recorder


@pytest.mark.asyncio
async def test_concurrent_spans_have_independent_parents(tmp_run_dir):
    """Two coroutines opening tool_call spans concurrently must each have
    the SAME outer parent (the agent_invoke span), not each other."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)

    async def inner_tool(name: str, delay: float):
        with rec.span("tool_call", tool_name=name, args={}, agent_name="a") as s:
            await asyncio.sleep(delay)
            return s.span_id, s.parent_id

    with rec.span("agent_invoke", agent_name="a", role="t") as outer:
        outer_id = outer.span_id
        # Two concurrent tools that overlap in time
        results = await asyncio.gather(
            inner_tool("alpha", 0.05),
            inner_tool("beta", 0.01),  # finishes first
        )
    rec.close()

    span_a, parent_a = results[0]
    span_b, parent_b = results[1]
    # Both inner spans must have the outer agent_invoke as parent
    # Without ContextVar, the second-to-exit would have its sibling as parent
    assert parent_a == outer_id, f"alpha's parent {parent_a} != outer {outer_id}"
    assert parent_b == outer_id, f"beta's parent {parent_b} != outer {outer_id}"
    assert span_a != span_b


@pytest.mark.asyncio
async def test_nested_spans_still_chain_correctly(tmp_run_dir):
    """Single-coroutine nested spans must still form a chain (no regression)."""
    rec = Recorder(run_id="r2", run_dir=tmp_run_dir)
    with rec.span("agent_invoke", agent_name="a", role="t") as outer:
        outer_id = outer.span_id
        with rec.span("tool_call", tool_name="x", args={}, agent_name="a") as inner:
            assert inner.parent_id == outer_id
            with rec.span("llm_call", provider="stub", model="m", agent_name="a") as deeper:
                assert deeper.parent_id == inner.span_id
    rec.close()
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_recorder_contextvars.py -v
```

Expected: `test_concurrent_spans_have_independent_parents` FAILS — alpha's parent will be beta's span_id (or vice versa) because list-based stack interleaves between the two coroutines.

`test_nested_spans_still_chain_correctly` should PASS already (single coroutine path).

- [ ] **Step 3: Replace `_span_stack` list with ContextVar**

Edit `multi_agent/tracing/recorder.py`. Add import at top:

```python
from contextvars import ContextVar
```

Replace the existing `__init__` lines that set `self._span_stack`:

```python
        self._span_stack: list[str] = []
```

With:

```python
        # Async-task-local stack: each coroutine gets its own view.
        # Use ContextVar at recorder level so isolated runs don't share state.
        self._span_stack_var: ContextVar[tuple[str, ...]] = ContextVar(
            f"span_stack_{run_id}", default=()
        )
```

Replace the three helper methods:

```python
    def current_parent_id(self) -> str | None:
        return self._span_stack[-1] if self._span_stack else None

    def push_span(self, span_id: str) -> None:
        self._span_stack.append(span_id)

    def pop_span(self, span_id: str) -> None:
        if not self._span_stack or self._span_stack[-1] != span_id:
            raise RuntimeError(f"span stack corrupted; expected {span_id}")
        self._span_stack.pop()
```

With:

```python
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
```

**Why tuple not list:** ContextVar values should be immutable so that contextvar's snapshot-on-task-spawn semantics work. We rebind with a new tuple each push/pop.

- [ ] **Step 4: Run tests — all pass + Phase 1 regression check**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_recorder_contextvars.py -v
pytest -v   # FULL suite — no regressions
```

Expected: 65 passed (63 from Phase 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tracing/recorder.py experiments/multi_agent/tests/unit/test_recorder_contextvars.py
git commit -m "phase2a(tracing): async-task-local span stack via ContextVar"
```

---

## Task 1: Add Dependencies + Document Qdrant Reuse

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md` (document Qdrant reuse, NOT a new container)

**Important:** We reuse the existing `legal-rag-qdrant` container (image `qdrant/qdrant:v1.12.4`, host ports `6433:6333` and `6434:6334`). Do NOT create a `docker-compose.yml` — the host's default port 6333 is already taken by another service.

- [ ] **Step 1: Verify Qdrant is reachable**

```bash
docker ps --filter "name=legal-rag-qdrant" --format "{{.Names}} {{.Status}} {{.Ports}}"
curl -s http://localhost:6433/healthz
```

Expected: container shows `Up X days (healthy)` and curl returns `healthz check passed` (or 200 status).

If the container is stopped, start it: `docker start legal-rag-qdrant`. If it doesn't exist, report BLOCKED — do not create a new one without user approval.

- [ ] **Step 2: Update `pyproject.toml`**

Read current `pyproject.toml` and add to the `dependencies` list:

```toml
dependencies = [
    "pydantic>=2.5",
    "python-ulid>=2.0",
    "aiosqlite>=0.19",
    "qdrant-client>=1.12",
    "sentence-transformers>=3.0",
    "jieba>=0.42",
    "numpy>=1.26",
]
```

The remaining keys (project metadata, optional-dependencies dev, pytest config, build-system, setuptools) stay unchanged.

- [ ] **Step 3: Install new dependencies**

```bash
cd /home/xxm/rag/experiments/multi_agent
pip install -e ".[dev]"
```

Expected: installs qdrant-client, sentence-transformers, jieba, numpy. sentence-transformers pulls in torch (~1.5 GB) as a transitive dep. This step may take several minutes.

- [ ] **Step 4: Smoke-import the new deps**

```bash
cd /home/xxm/rag/experiments/multi_agent
python -c "
import qdrant_client
import sentence_transformers
import jieba
import numpy as np
print('qdrant-client', qdrant_client.__version__)
print('sentence-transformers', sentence_transformers.__version__)
print('jieba ok')
print('numpy', np.__version__)

# Verify connection to the existing container
from qdrant_client import QdrantClient
c = QdrantClient(url='http://localhost:6433')
print('Existing collections:', [coll.name for coll in c.get_collections().collections])
"
```

Expected: all four imports succeed; existing collections include `legal_statutes` and some test artifacts. Multi_agent's own collections (prefixed `ma_`) will be added in later tasks.

- [ ] **Step 5: Update `README.md`**

Append after the existing "## Run tests" section:

```markdown

## Qdrant

This project **reuses** the existing `legal-rag-qdrant` container on host ports `6433:6333` and `6434:6334`. It does NOT manage its own container.

```bash
# Verify running
docker ps | grep legal-rag-qdrant
curl http://localhost:6433/healthz

# Restart if stopped
docker start legal-rag-qdrant

# List multi_agent's own collections (prefixed `ma_`)
curl -s http://localhost:6433/collections | python -m json.tool
```

Multi_agent collections coexist with legacy `legal_rag/` collections under the `ma_` prefix:
- `ma_statutes` — Chinese-Laws indexed by multi_agent (Phase 2a)
- `ma_cases` — laws_data Q&A (Phase 2d)
- `ma_user_history` — turn/sticky-derived index (Phase 2d)

Connection URL is configurable via the `QDRANT_URL` env var (default `http://localhost:6433`).
```

- [ ] **Step 6: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/pyproject.toml experiments/multi_agent/README.md
git commit -m "phase2a(deps): add qdrant-client/sentence-transformers/jieba; document reuse of legal-rag-qdrant"
```

---

## Task 2: Document + Chunk Schemas

**Files:**
- Create: `multi_agent/schemas/document.py`
- Create: `tests/unit/test_document.py`

**Purpose:** Pydantic types for the parsed corpus. `Document` is one law text. `Chunk` is one article inside that document — the unit of retrieval (per spec §4.4: "chunk = 1 article").

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_document.py
from multi_agent.schemas.document import Document, Chunk


def test_chunk_required_fields():
    c = Chunk(
        doc_id="民法典-510",
        law_name="中华人民共和国民法典",
        law_short="民法典",
        article_no="510",
        text="当事人就合同补充内容没有约定...",
    )
    assert c.doc_id == "民法典-510"
    assert c.metadata == {}                     # default empty
    assert c.cross_refs == []
    assert c.concepts == []


def test_chunk_with_optional_fields():
    c = Chunk(
        doc_id="d", law_name="l", law_short="L", article_no="1",
        text="t",
        book="合同编", chapter="合同的订立",
        cross_refs=["第511条"], concepts=["合同补充"],
        metadata={"source_file": "law.txt"},
    )
    assert c.book == "合同编"
    assert c.chapter == "合同的订立"
    assert c.metadata["source_file"] == "law.txt"


def test_chunk_embedding_text_includes_law_chapter_article():
    """The string used to build the dense embedding should include
    law_short + book + chapter + article_no + text, per spec §4.4."""
    c = Chunk(
        doc_id="d", law_name="l", law_short="民法典", article_no="510",
        text="正文内容",
        book="合同编", chapter="合同的订立",
    )
    et = c.embedding_text()
    assert "民法典" in et
    assert "合同编" in et
    assert "合同的订立" in et
    assert "510" in et
    assert "正文内容" in et


def test_document_holds_chunks():
    d = Document(
        law_name="民法典", law_short="民法典",
        source_path="laws/民法典.txt",
        chunks=[
            Chunk(doc_id="民法典-1", law_name="民法典", law_short="民法典", article_no="1", text="a"),
            Chunk(doc_id="民法典-2", law_name="民法典", law_short="民法典", article_no="2", text="b"),
        ],
    )
    assert len(d.chunks) == 2
    assert d.law_short == "民法典"
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_document.py -v
```

Expected: ImportError on `multi_agent.schemas.document`.

- [ ] **Step 3: Create `multi_agent/schemas/document.py`**

```python
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """One law article — the unit of retrieval (per spec §4.4).

    `text` is the raw article body. `embedding_text()` prepends
    law/book/chapter/article context to improve dense recall on
    short articles (spec §4.4 'Embedding 拼接').
    """
    doc_id: str                                 # e.g. "民法典-510"
    law_name: str                               # e.g. "中华人民共和国民法典"
    law_short: str                              # e.g. "民法典"
    article_no: str                             # e.g. "510"
    text: str                                   # article body only
    # Optional structural context (spec §4.4 says skip in V0,
    # but the field stays — populated when chapters get added later)
    book: str = ""
    chapter: str = ""
    # Optional enrichment
    cross_refs: list[str] = Field(default_factory=list)
    preceding_text: str = ""
    following_text: str = ""
    concepts: list[str] = Field(default_factory=list)
    # Free-form metadata (source file path, version, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text fed to the dense encoder. Includes structural context
        so short articles have richer embeddings."""
        parts = [f"《{self.law_short}》"]
        if self.book:
            parts.append(self.book)
        if self.chapter:
            parts.append(self.chapter)
        parts.append(f"第{self.article_no}条")
        head = "·".join(parts)
        return f"{head}: {self.text}"


class Document(BaseModel):
    """One law file (e.g. 民法典 全文). Contains many Chunks."""
    law_name: str
    law_short: str
    source_path: str
    chunks: list[Chunk] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_document.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/document.py experiments/multi_agent/tests/unit/test_document.py
git commit -m "phase2a(schemas): Document + Chunk with embedding_text()"
```

---

## Task 3: Corpus Loader (parse Chinese-Laws txt files)

**Files:**
- Create: `multi_agent/tools/corpus.py`
- Create: `tests/unit/test_corpus.py`
- Create: `tests/fixtures/sample_laws/民法典-sample.txt`

**Background:** The corpus at `/home/xxm/rag/Chinese-Laws/extracted/` contains 177 `.txt` files. Each line is one article in this format:

```
《中华人民共和国民法典》第八条规定，民事主体从事民事活动，不得违反法律，不得违背公序良俗。
《中华人民共和国民法典》第九条规定，民事主体从事民事活动，应当有利于节约资源、保护生态环境。
```

We need to parse this into Chunks with `law_name`, `article_no`, and `text`.

- [ ] **Step 1: Create fixture file**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/tests/fixtures/sample_laws
```

Write `/home/xxm/rag/experiments/multi_agent/tests/fixtures/sample_laws/民法典-sample.txt`:

```
《中华人民共和国民法典》第一条规定，为了保护民事主体的合法权益，调整民事关系。
《中华人民共和国民法典》第二条规定，民法调整平等主体之间的人身关系和财产关系。
《中华人民共和国民法典》第五百一十条规定，当事人就合同补充内容没有约定的，按照合同相关条款或者交易习惯确定。
```

- [ ] **Step 2: Write failing test**

```python
# tests/unit/test_corpus.py
from pathlib import Path
import pytest
from multi_agent.tools.corpus import load_law_file, load_corpus
from multi_agent.schemas.document import Document, Chunk


FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "sample_laws"


def test_load_law_file_returns_document():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    assert isinstance(doc, Document)
    assert doc.law_short == "民法典"
    assert doc.law_name == "中华人民共和国民法典"
    assert len(doc.chunks) == 3


def test_chunks_have_correct_article_numbers():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    article_nos = [c.article_no for c in doc.chunks]
    # "第一条" → "1", "第二条" → "2", "第五百一十条" → "510"
    assert article_nos == ["1", "2", "510"]


def test_chunks_have_clean_text():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    # First chunk text should be the body AFTER "规定，"
    assert doc.chunks[0].text.startswith("为了保护民事主体")
    # Should NOT include the law name prefix or article marker
    assert "《" not in doc.chunks[0].text
    assert "第一条" not in doc.chunks[0].text


def test_chunk_doc_ids_unique_per_law():
    doc = load_law_file(FIXTURE_DIR / "民法典-sample.txt")
    ids = {c.doc_id for c in doc.chunks}
    assert len(ids) == 3
    assert "民法典-1" in ids
    assert "民法典-510" in ids


def test_load_corpus_finds_all_files(tmp_path):
    # Make two tiny law files in tmp dir
    (tmp_path / "民法典.txt").write_text(
        "《中华人民共和国民法典》第一条规定，第一条内容。\n", encoding="utf-8"
    )
    (tmp_path / "刑法.txt").write_text(
        "《中华人民共和国刑法》第一条规定，第一条内容。\n", encoding="utf-8"
    )
    docs = load_corpus(tmp_path)
    assert len(docs) == 2
    shorts = {d.law_short for d in docs}
    assert shorts == {"民法典", "刑法"}


def test_load_corpus_skips_non_txt(tmp_path):
    (tmp_path / "民法典.txt").write_text(
        "《中华人民共和国民法典》第一条规定，正文。\n", encoding="utf-8"
    )
    (tmp_path / "readme.md").write_text("# not a law\n", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert len(docs) == 1


def test_load_law_file_skips_malformed_lines(tmp_path):
    # Mix of valid lines and garbage
    (tmp_path / "law.txt").write_text(
        "《中华人民共和国民法典》第一条规定，正文一。\n"
        "GARBAGE LINE\n"
        "《中华人民共和国民法典》第二条规定，正文二。\n",
        encoding="utf-8",
    )
    doc = load_law_file(tmp_path / "law.txt")
    assert len(doc.chunks) == 2  # garbage skipped
```

- [ ] **Step 3: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_corpus.py -v
```

Expected: ImportError on `multi_agent.tools.corpus`.

- [ ] **Step 4: Create `multi_agent/tools/corpus.py`**

```python
"""Parse Chinese-Laws .txt files into Document objects.

Source format per line:
  《<law_name>》第<article_no_cn>条规定，<text body>。
"""
from __future__ import annotations
import re
from pathlib import Path
from multi_agent.schemas.document import Document, Chunk


# Pattern: 《<law_name>》第<article_no_cn>条规定，<text>
_LINE_RE = re.compile(
    r"^《(?P<law_name>[^》]+)》第(?P<article_cn>[一二三四五六七八九十百千零\d]+)条规定[，,](?P<text>.+)$"
)

# Chinese numeral → arabic numeral
_CN_DIGIT = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def chinese_to_int(s: str) -> int:
    """Convert Chinese numeral string like '五百一十' to integer 510.
    Falls through to int() for ASCII-digit strings."""
    if s.isdigit():
        return int(s)

    total = 0
    section = 0
    current = 0
    for ch in s:
        if ch in _CN_DIGIT:
            current = _CN_DIGIT[ch]
        elif ch == "十":
            section += (current if current else 1) * 10
            current = 0
        elif ch == "百":
            section += (current if current else 1) * 100
            current = 0
        elif ch == "千":
            section += (current if current else 1) * 1000
            current = 0
        elif ch == "万":
            total += (section + current) * 10000
            section = 0
            current = 0
        else:
            raise ValueError(f"unknown Chinese numeral char: {ch}")
    return total + section + current


def _law_short_from_name(law_name: str) -> str:
    """'中华人民共和国民法典' → '民法典'."""
    prefix = "中华人民共和国"
    return law_name[len(prefix):] if law_name.startswith(prefix) else law_name


def load_law_file(path: Path) -> Document:
    """Parse one law .txt file into a Document with one Chunk per article."""
    path = Path(path)
    chunks: list[Chunk] = []
    law_name = ""
    law_short = ""

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue  # skip malformed line
            if not law_name:
                law_name = m.group("law_name")
                law_short = _law_short_from_name(law_name)
            try:
                article_no = str(chinese_to_int(m.group("article_cn")))
            except ValueError:
                continue
            text = m.group("text").rstrip("。")
            chunks.append(
                Chunk(
                    doc_id=f"{law_short}-{article_no}",
                    law_name=law_name,
                    law_short=law_short,
                    article_no=article_no,
                    text=text,
                    metadata={"source_file": str(path)},
                )
            )

    return Document(
        law_name=law_name,
        law_short=law_short,
        source_path=str(path),
        chunks=chunks,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Scan a directory for .txt law files and parse each."""
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    for path in sorted(corpus_dir.iterdir()):
        if path.is_file() and path.suffix == ".txt":
            doc = load_law_file(path)
            if doc.chunks:
                docs.append(doc)
    return docs
```

- [ ] **Step 5: Run test to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_corpus.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Smoke test against real corpus**

```bash
cd /home/xxm/rag/experiments/multi_agent
python -c "
from pathlib import Path
from multi_agent.tools.corpus import load_law_file
d = load_law_file(Path('/home/xxm/rag/Chinese-Laws/extracted/中华人民共和国民法典.txt'))
print(f'law_short: {d.law_short}')
print(f'num_chunks: {len(d.chunks)}')
print(f'first article: {d.chunks[0].doc_id} text len={len(d.chunks[0].text)}')
print(f'article 510 found:', any(c.doc_id == '民法典-510' for c in d.chunks))
"
```

Expected: `law_short: 民法典`, `num_chunks` close to 1259, `article 510 found: True`.

- [ ] **Step 7: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/corpus.py experiments/multi_agent/tests/unit/test_corpus.py experiments/multi_agent/tests/fixtures/
git commit -m "phase2a(corpus): parse Chinese-Laws .txt → Document + Chunk"
```

---

## Task 4: Dense Encoder (bge-m3)

**Files:**
- Create: `multi_agent/tools/retrievers/__init__.py`
- Create: `multi_agent/tools/retrievers/dense_encoder.py`
- Create: `tests/unit/test_dense_encoder.py`

**Note on model download:** First call to `SentenceTransformer("BAAI/bge-m3")` downloads ~2.3 GB from HuggingFace to `~/.cache/huggingface/`. This is a one-time cost.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_dense_encoder.py
import numpy as np
import pytest
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


@pytest.fixture(scope="module")
def encoder():
    return DenseEncoder()


def test_encode_single_returns_1d_vector(encoder):
    vec = encoder.encode_one("民法典第510条")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == encoder.dim  # bge-m3 is 1024


def test_encode_batch_returns_2d_matrix(encoder):
    texts = ["民法典第一条", "民法典第二条", "刑法第十三条"]
    mat = encoder.encode_batch(texts)
    assert mat.shape == (3, encoder.dim)


def test_encode_vectors_are_unit_normalized(encoder):
    vec = encoder.encode_one("测试文本")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4   # bge-m3 outputs are normalized


def test_similar_text_higher_similarity(encoder):
    a = encoder.encode_one("房东要涨房租")
    b = encoder.encode_one("房屋租金变更")
    c = encoder.encode_one("天气真好")
    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac, f"租赁相关应比天气相关更相似: {sim_ab} vs {sim_ac}"
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_dense_encoder.py -v
```

Expected: ImportError on `multi_agent.tools.retrievers.dense_encoder`.

- [ ] **Step 3: Create encoder files**

```python
# multi_agent/tools/retrievers/__init__.py
```

```python
# multi_agent/tools/retrievers/dense_encoder.py
"""bge-m3 dense embedding wrapper."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEncoder:
    """Wrap sentence-transformers bge-m3 for batch encoding.

    First instantiation downloads ~2.3 GB to HuggingFace cache.
    Subsequent calls reuse the cached model.
    """

    DEFAULT_MODEL = "BAAI/bge-m3"
    DEFAULT_DIM = 1024

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self._model = SentenceTransformer(model_name, device=device)
        # bge-m3 is 1024-dim; we read from model to be robust to other models
        self.dim = self._model.get_sentence_embedding_dimension() or self.DEFAULT_DIM

    def encode_one(self, text: str) -> np.ndarray:
        """Encode a single text into a normalized 1D vector."""
        return self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_batch(
        self,
        texts: Iterable[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode a list of texts → 2D matrix (N, dim) of unit vectors."""
        texts = list(texts)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
```

- [ ] **Step 4: Run test to verify pass (slow — first run downloads model)**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_dense_encoder.py -v
```

Expected: 4 passed. First run downloads bge-m3 (~2.3 GB, may take 5-30 minutes depending on bandwidth). Subsequent runs are fast.

If download fails (network issue), report BLOCKED with the error. Do not invent a smaller model substitution without user approval.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/__init__.py experiments/multi_agent/multi_agent/tools/retrievers/dense_encoder.py experiments/multi_agent/tests/unit/test_dense_encoder.py
git commit -m "phase2a(retrievers): DenseEncoder wrapping bge-m3 with batch encoding"
```

---

## Task 5: Sparse Encoder (jieba + IDF)

**Files:**
- Create: `multi_agent/tools/retrievers/sparse_encoder.py`
- Create: `tests/unit/test_sparse_encoder.py`

**What is a "sparse vector" in Qdrant?** A dict-like `{token_id_1: weight_1, token_id_2: weight_2, ...}`. Token IDs are arbitrary stable integers we assign per unique token. Weights are typically TF×IDF. Qdrant matches sparse query against sparse index and ranks by dot product.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_sparse_encoder.py
import pytest
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder, SparseVector


CORPUS = [
    "民法典第一条 立法目的",
    "民法典第二条 调整民事关系",
    "民法典第五百一十条 合同补充内容确定",
    "刑法第十三条 犯罪定义",
    "刑法第十四条 故意犯罪",
]


def test_fit_and_encode_returns_sparse_vector():
    enc = SparseEncoder()
    enc.fit(CORPUS)
    vec = enc.encode("民法典 合同")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) == len(vec.values)
    assert len(vec.indices) > 0


def test_idf_weights_rare_terms_higher():
    """A token appearing in 1 doc should have higher IDF than one in many."""
    enc = SparseEncoder()
    enc.fit(CORPUS)
    rare_vec = enc.encode("合同补充")     # appears in 1 doc
    common_vec = enc.encode("民法典")     # appears in 3 docs
    rare_max_val = max(rare_vec.values) if rare_vec.values else 0
    common_max_val = max(common_vec.values) if common_vec.values else 0
    assert rare_max_val > common_max_val


def test_oov_token_returns_empty_vector():
    """Query with only unseen tokens → empty vector (Qdrant treats as no match)."""
    enc = SparseEncoder()
    enc.fit(CORPUS)
    vec = enc.encode("xyz-unknown-12345")
    assert vec.indices == []
    assert vec.values == []


def test_save_and_load_roundtrip(tmp_path):
    enc = SparseEncoder()
    enc.fit(CORPUS)
    out = tmp_path / "sparse.json"
    enc.save(out)

    enc2 = SparseEncoder.load(out)
    v1 = enc.encode("民法典 合同")
    v2 = enc2.encode("民法典 合同")
    assert v1.indices == v2.indices
    assert v1.values == v2.values
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_sparse_encoder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/sparse_encoder.py`**

```python
"""jieba + IDF sparse vector encoder for Qdrant sparse retrieval.

Sparse vector format matches Qdrant's expectation: parallel lists of
integer token IDs and float weights. The vocabulary is built from
`fit(corpus_texts)` and persisted to disk for reproducible indexing.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections import Counter
from typing import Iterable
import jieba
from pydantic import BaseModel


class SparseVector(BaseModel):
    """Parallel arrays: indices[i] has weight values[i]."""
    indices: list[int]
    values: list[float]


def _tokenize(text: str) -> list[str]:
    """jieba tokenize, drop pure-whitespace and very short tokens."""
    return [tok for tok in jieba.cut(text) if tok.strip() and len(tok.strip()) > 0]


class SparseEncoder:
    """Stateful encoder. Must call fit() before encode() unless loaded from disk."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._idf: dict[int, float] = {}
        self._fitted = False

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def fit(self, corpus_texts: Iterable[str]) -> None:
        """Build vocabulary and IDF table from a corpus."""
        corpus_texts = list(corpus_texts)
        # 1) Build vocab
        for text in corpus_texts:
            for tok in _tokenize(text):
                if tok not in self._token_to_id:
                    self._token_to_id[tok] = len(self._token_to_id)
        # 2) Compute document-frequency
        n_docs = len(corpus_texts)
        df: Counter[int] = Counter()
        for text in corpus_texts:
            seen: set[int] = set()
            for tok in _tokenize(text):
                tid = self._token_to_id[tok]
                if tid not in seen:
                    seen.add(tid)
                    df[tid] += 1
        # 3) IDF = log((N + 1) / (df + 1)) + 1  (smoothed)
        for tid, count in df.items():
            self._idf[tid] = math.log((n_docs + 1) / (count + 1)) + 1.0
        self._fitted = True

    def encode(self, text: str) -> SparseVector:
        """Encode one text → sparse TF×IDF vector. Unseen tokens dropped."""
        if not self._fitted:
            raise RuntimeError("SparseEncoder.fit() must be called first")
        tf: Counter[int] = Counter()
        for tok in _tokenize(text):
            tid = self._token_to_id.get(tok)
            if tid is None:
                continue  # OOV
            tf[tid] += 1
        indices: list[int] = []
        values: list[float] = []
        for tid, freq in tf.items():
            idf = self._idf.get(tid, 0.0)
            weight = float(freq) * idf
            if weight > 0:
                indices.append(int(tid))
                values.append(weight)
        return SparseVector(indices=indices, values=values)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_id": self._token_to_id,
            "idf": {str(k): v for k, v in self._idf.items()},  # JSON keys must be str
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SparseEncoder":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls()
        enc._token_to_id = payload["token_to_id"]
        enc._idf = {int(k): v for k, v in payload["idf"].items()}
        enc._fitted = True
        return enc
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_sparse_encoder.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/sparse_encoder.py experiments/multi_agent/tests/unit/test_sparse_encoder.py
git commit -m "phase2a(retrievers): SparseEncoder with jieba + TF*IDF + save/load"
```

---

## Task 6: Qdrant Client + Collection Management

**Files:**
- Create: `multi_agent/tools/retrievers/qdrant_client.py`
- Create: `tests/unit/test_qdrant_client.py`

**Prerequisite:** The shared `legal-rag-qdrant` container must be running. Verify with `docker ps | grep legal-rag-qdrant`; if stopped, run `docker start legal-rag-qdrant`.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_qdrant_client.py
import uuid
import pytest
from multi_agent.tools.retrievers.qdrant_client import (
    get_qdrant_client, ensure_collection, drop_collection, STATUTE_COLLECTION_PARAMS,
)


@pytest.fixture
def temp_collection_name():
    """Unique collection name per test, cleaned up after."""
    name = f"test_coll_{uuid.uuid4().hex[:8]}"
    yield name
    try:
        drop_collection(name)
    except Exception:
        pass


def test_client_singleton_returns_same_instance():
    c1 = get_qdrant_client()
    c2 = get_qdrant_client()
    assert c1 is c2


def test_ensure_collection_creates_with_named_vectors(temp_collection_name):
    ensure_collection(temp_collection_name, STATUTE_COLLECTION_PARAMS)
    client = get_qdrant_client()
    info = client.get_collection(temp_collection_name)
    # Both named vectors (dense + sparse) must exist
    vecs = info.config.params.vectors
    sparse = info.config.params.sparse_vectors
    assert "dense" in vecs
    assert "sparse" in sparse


def test_ensure_collection_idempotent(temp_collection_name):
    ensure_collection(temp_collection_name, STATUTE_COLLECTION_PARAMS)
    ensure_collection(temp_collection_name, STATUTE_COLLECTION_PARAMS)  # no error
    client = get_qdrant_client()
    info = client.get_collection(temp_collection_name)
    assert info is not None


def test_drop_collection_removes_it(temp_collection_name):
    ensure_collection(temp_collection_name, STATUTE_COLLECTION_PARAMS)
    drop_collection(temp_collection_name)
    client = get_qdrant_client()
    names = {c.name for c in client.get_collections().collections}
    assert temp_collection_name not in names
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_qdrant_client.py -v
```

Expected: ImportError (or, if file exists, connection error).

- [ ] **Step 3: Create `multi_agent/tools/retrievers/qdrant_client.py`**

```python
"""Qdrant connection singleton + collection helpers.

Default points at the shared `legal-rag-qdrant` container on host port 6433
(NOT the standard 6333 — that port is occupied by another service).
Override via QDRANT_URL env var.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from qdrant_client import QdrantClient, models


# Singleton (lazy)
_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL", "http://localhost:6433")
        _client = QdrantClient(url=url, timeout=30)
    return _client


@dataclass(frozen=True)
class CollectionParams:
    """Shape of one Qdrant collection: dense dim + which sparse name."""
    dense_dim: int
    dense_name: str = "dense"
    sparse_name: str = "sparse"


# Shape used for `statutes`, `cases`, `user_history` (all spec §4.2).
STATUTE_COLLECTION_PARAMS = CollectionParams(dense_dim=1024)


def ensure_collection(name: str, params: CollectionParams) -> None:
    """Create the collection if missing. Idempotent."""
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            params.dense_name: models.VectorParams(
                size=params.dense_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            params.sparse_name: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )


def drop_collection(name: str) -> None:
    """Delete a collection. No-op if missing."""
    client = get_qdrant_client()
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass
```

- [ ] **Step 4: Verify Qdrant is reachable, then run tests**

```bash
docker ps | grep legal-rag-qdrant   # confirm container running
curl -s http://localhost:6433/healthz | head -1
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_qdrant_client.py -v
```

Expected: container line shows "Up X (healthy)", `healthz check passed`, then 4 tests pass.

If Qdrant unreachable: `docker start legal-rag-qdrant`. If that fails, report BLOCKED — do not stub the client.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/qdrant_client.py experiments/multi_agent/tests/unit/test_qdrant_client.py
git commit -m "phase2a(retrievers): Qdrant client singleton + ensure/drop collection"
```

---

## Task 7: Index Builder

**Files:**
- Create: `multi_agent/tools/retrievers/index_builder.py`
- Create: `scripts/build_statutes_index.py`
- Create: `tests/unit/test_index_builder.py`

**Purpose:** Given a list of `Document`s, encode each `Chunk` with dense + sparse vectors and upsert into Qdrant. Also persists the sparse encoder's vocab/IDF so future searches use the same tokens.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_index_builder.py
import uuid
import pytest
from multi_agent.schemas.document import Document, Chunk
from multi_agent.tools.retrievers.index_builder import build_index, IndexArtifacts
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client, drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


@pytest.fixture
def temp_collection_name():
    name = f"test_idx_{uuid.uuid4().hex[:8]}"
    yield name
    drop_collection(name)


def _docs():
    return [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典",
            source_path="t",
            chunks=[
                Chunk(doc_id="民法典-1", law_name="中华人民共和国民法典", law_short="民法典",
                      article_no="1", text="为了保护民事主体的合法权益。"),
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典", law_short="民法典",
                      article_no="510", text="当事人就合同补充内容没有约定的，按照合同相关条款确定。"),
            ],
        ),
    ]


def test_build_index_creates_points(temp_collection_name, tmp_path):
    artifacts = build_index(
        documents=_docs(),
        collection_name=temp_collection_name,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    assert isinstance(artifacts, IndexArtifacts)
    client = get_qdrant_client()
    count = client.count(temp_collection_name).count
    assert count == 2


def test_build_index_persists_sparse_encoder(temp_collection_name, tmp_path):
    out = tmp_path / "sparse.json"
    build_index(
        documents=_docs(),
        collection_name=temp_collection_name,
        sparse_artifact_path=out,
        dense_encoder=DenseEncoder(),
    )
    assert out.exists()
    enc = SparseEncoder.load(out)
    assert enc.vocab_size > 0


def test_build_index_idempotent_upsert(temp_collection_name, tmp_path):
    """Running twice should keep count at 2 (upsert by doc_id, not append)."""
    build_index(
        documents=_docs(),
        collection_name=temp_collection_name,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    build_index(
        documents=_docs(),
        collection_name=temp_collection_name,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    client = get_qdrant_client()
    assert client.count(temp_collection_name).count == 2
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_index_builder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/index_builder.py`**

```python
"""Build a Qdrant collection from a list of Documents.

For each Chunk: encode with dense + sparse encoders, then upsert as
a point with payload = chunk metadata.

Sparse encoder is fit on the entire corpus first (to compute IDF),
then persisted next to the index so search queries use the same vocab.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from qdrant_client import models

from multi_agent.schemas.document import Document
from multi_agent.tools.retrievers.qdrant_client import (
    get_qdrant_client, ensure_collection, STATUTE_COLLECTION_PARAMS,
)
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


def _point_id_from_doc_id(doc_id: str) -> int:
    """Stable integer ID from doc_id string. Qdrant accepts uuid or unsigned int.
    Use truncated sha256 for determinism and idempotent upserts."""
    h = hashlib.sha256(doc_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") >> 1  # 63-bit positive


@dataclass
class IndexArtifacts:
    """Returned by build_index() — paths the search tool needs at query time."""
    collection_name: str
    sparse_artifact_path: Path
    dense_dim: int


def build_index(
    *,
    documents: Sequence[Document],
    collection_name: str,
    sparse_artifact_path: Path,
    dense_encoder: DenseEncoder,
    batch_size: int = 64,
) -> IndexArtifacts:
    """Encode every chunk in `documents` and upsert into Qdrant.

    Steps:
      1. Flatten chunks
      2. Fit SparseEncoder on chunk texts
      3. Encode dense + sparse for every chunk
      4. Upsert in batches
      5. Persist sparse encoder to disk
    """
    sparse_artifact_path = Path(sparse_artifact_path)

    chunks = [c for doc in documents for c in doc.chunks]
    if not chunks:
        raise ValueError("no chunks to index")

    # 1. Sparse encoder fits on raw article text (not the enriched embedding text —
    #    spec §4.4 says sparse encodes article body only).
    sparse_enc = SparseEncoder()
    sparse_enc.fit(c.text for c in chunks)

    # 2. Ensure collection exists with the right shape.
    ensure_collection(collection_name, STATUTE_COLLECTION_PARAMS)
    client = get_qdrant_client()

    # 3. Encode + upsert in batches.
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        dense_vecs = dense_encoder.encode_batch([c.embedding_text() for c in batch])
        points = []
        for chunk, dense_vec in zip(batch, dense_vecs):
            sparse_vec = sparse_enc.encode(chunk.text)
            points.append(
                models.PointStruct(
                    id=_point_id_from_doc_id(chunk.doc_id),
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices,
                            values=sparse_vec.values,
                        ),
                    },
                    payload={
                        "doc_id": chunk.doc_id,
                        "law_name": chunk.law_name,
                        "law_short": chunk.law_short,
                        "article_no": chunk.article_no,
                        "text": chunk.text,
                        "book": chunk.book,
                        "chapter": chunk.chapter,
                        "concepts": chunk.concepts,
                        "cross_refs": chunk.cross_refs,
                        "metadata": chunk.metadata,
                    },
                )
            )
        client.upsert(collection_name=collection_name, points=points)

    # 4. Persist sparse encoder vocabulary.
    sparse_enc.save(sparse_artifact_path)

    return IndexArtifacts(
        collection_name=collection_name,
        sparse_artifact_path=sparse_artifact_path,
        dense_dim=dense_encoder.dim,
    )
```

- [ ] **Step 4: Create CLI script `scripts/build_statutes_index.py`**

```bash
mkdir -p /home/xxm/rag/experiments/multi_agent/scripts
```

```python
# scripts/build_statutes_index.py
"""CLI: build the `statutes` collection from /home/xxm/rag/Chinese-Laws/extracted/.

Usage:
    cd /home/xxm/rag/experiments/multi_agent
    python -m scripts.build_statutes_index [--limit N] [--collection NAME]
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

from multi_agent.tools.corpus import load_corpus
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("/home/xxm/rag/Chinese-Laws/extracted"),
    )
    parser.add_argument("--collection", default="ma_statutes")
    parser.add_argument(
        "--sparse-out",
        type=Path,
        default=Path("indexes/ma_statutes_sparse.json"),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only index first N law files (0 = all). Useful for smoke testing.",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    docs = load_corpus(args.corpus_dir)
    if args.limit:
        docs = docs[: args.limit]
    n_chunks = sum(len(d.chunks) for d in docs)
    print(f"Loaded {len(docs)} laws, {n_chunks} chunks. Encoding...")

    encoder = DenseEncoder()
    artifacts = build_index(
        documents=docs,
        collection_name=args.collection,
        sparse_artifact_path=args.sparse_out,
        dense_encoder=encoder,
    )
    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s. Sparse vocab saved to {artifacts.sparse_artifact_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run unit tests to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_index_builder.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Smoke run the CLI on 2 laws**

```bash
docker ps | grep legal-rag-qdrant   # confirm running
cd /home/xxm/rag/experiments/multi_agent
python -m scripts.build_statutes_index --limit 2 --collection ma_statutes_smoke --sparse-out indexes/ma_statutes_smoke_sparse.json
```

Expected output: `Loaded 2 laws, ~N chunks. Encoding... Done in X.Xs. Sparse vocab saved to indexes/ma_statutes_smoke_sparse.json`

Verify:

```bash
curl -s http://localhost:6433/collections/ma_statutes_smoke | python -m json.tool | head
```

Should show non-zero `points_count`.

- [ ] **Step 7: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/index_builder.py experiments/multi_agent/scripts/build_statutes_index.py experiments/multi_agent/tests/unit/test_index_builder.py
git commit -m "phase2a(retrievers): IndexBuilder + CLI script for statutes collection"
```

---

## Task 8: statute_search Tool

**Files:**
- Create: `multi_agent/tools/retrievers/statute_search.py`
- Create: `tests/unit/test_statute_search.py`

**Purpose:** A `Tool` (per Phase 1 Tool ABC) that runs hybrid search (dense + sparse + RRF) against the `statutes` collection and returns `Evidence` objects.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_statute_search.py
import uuid
import pytest
from pathlib import Path

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool, StatuteSearchArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    """Build a fresh 4-chunk collection for the entire module."""
    name = f"test_statute_{uuid.uuid4().hex[:8]}"
    tmp_dir = tmp_path_factory.mktemp("idx")
    sparse_path = tmp_dir / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典",
            source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照合同相关条款或者交易习惯确定。"),
                Chunk(doc_id="民法典-563", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="563",
                      text="当事人一方违约时，对方可以解除合同。"),
            ],
        ),
        Document(
            law_name="中华人民共和国刑法", law_short="刑法",
            source_path="t",
            chunks=[
                Chunk(doc_id="刑法-13", law_name="中华人民共和国刑法",
                      law_short="刑法", article_no="13",
                      text="一切危害国家主权的行为依照法律应当受刑罚处罚的，都是犯罪。"),
                Chunk(doc_id="刑法-14", law_name="中华人民共和国刑法",
                      law_short="刑法", article_no="14",
                      text="明知自己的行为会发生危害社会的结果，并且希望或者放任这种结果发生，因而构成犯罪的，是故意犯罪。"),
            ],
        ),
    ]
    build_index(
        documents=docs,
        collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_search_returns_evidence_list(small_index, tmp_run_dir):
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(StatuteSearchArgs(query="合同补充约定", k=3), rec)
    rec.close()
    assert result.error is None
    hits = result.payload["evidences"]
    assert isinstance(hits, list)
    assert len(hits) >= 1
    # Top hit must be the contract article, not the criminal one
    top = Evidence.model_validate(hits[0])
    assert top.law_short == "民法典"
    assert top.retriever == "hybrid"


@pytest.mark.asyncio
async def test_search_respects_k(small_index, tmp_run_dir):
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(StatuteSearchArgs(query="违约", k=1), rec)
    rec.close()
    assert len(result.payload["evidences"]) == 1


@pytest.mark.asyncio
async def test_search_filter_by_law_short(small_index, tmp_run_dir):
    """Filter narrows results to just one law."""
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        StatuteSearchArgs(query="犯罪", k=5, law_short="刑法"),
        rec,
    )
    rec.close()
    hits = result.payload["evidences"]
    assert len(hits) >= 1
    assert all(Evidence.model_validate(h).law_short == "刑法" for h in hits)
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_statute_search.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/statute_search.py`**

```python
"""Hybrid (dense+sparse, RRF-fused) search over the `statutes` collection.

Implemented as a Tool so Phase 2c Lawyer can call it via the standard
ReAct dispatch path.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


class StatuteSearchArgs(BaseModel):
    query: str
    k: int = 10
    law_short: str | None = None     # filter: only this law (e.g. "民法典")


class StatuteSearchTool(Tool):
    name: str = "statute_search"
    description: str = (
        "Search Chinese statutes using hybrid retrieval "
        "(dense BAAI/bge-m3 + sparse jieba+IDF, fused via RRF). "
        "Returns up to k Evidence objects. Optional law_short filter."
    )
    args_schema: type[BaseModel] = StatuteSearchArgs
    # Runtime config — not LLM-visible
    collection_name: str
    sparse_artifact_path: Path

    # Lazy-initialized state
    _dense: Any = None
    _sparse: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _ensure_encoders(self) -> None:
        if self._dense is None:
            object.__setattr__(self, "_dense", DenseEncoder())
        if self._sparse is None:
            object.__setattr__(
                self, "_sparse", SparseEncoder.load(self.sparse_artifact_path)
            )

    async def call(self, args: StatuteSearchArgs, recorder: Recorder) -> ToolResult:
        self._ensure_encoders()
        client = get_qdrant_client()

        dense_vec = self._dense.encode_one(args.query).tolist()
        sparse_vec = self._sparse.encode(args.query)

        # Build optional filter
        query_filter = None
        if args.law_short:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_short",
                        match=models.MatchValue(value=args.law_short),
                    )
                ]
            )

        # Native hybrid via prefetch + RRF
        result = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=max(args.k * 2, 20),
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices,
                        values=sparse_vec.values,
                    ),
                    using="sparse",
                    limit=max(args.k * 2, 20),
                    filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=args.k,
            with_payload=True,
        )

        evidences: list[dict] = []
        for point in result.points:
            payload = point.payload or {}
            ev = Evidence(
                doc_id=payload.get("doc_id", ""),
                law_name=payload.get("law_name", ""),
                article_no=payload.get("article_no", ""),
                text=payload.get("text", ""),
                score=float(point.score) if point.score is not None else 0.0,
                retriever="hybrid",
                metadata={
                    "law_short": payload.get("law_short", ""),
                    "book": payload.get("book", ""),
                    "chapter": payload.get("chapter", ""),
                    "concepts": payload.get("concepts", []),
                },
            )
            evidences.append(ev.model_dump())

        return ToolResult(
            tool_use_id="",            # filled by BaseAgent._dispatch_tool
            payload={"evidences": evidences, "count": len(evidences)},
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_statute_search.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/statute_search.py experiments/multi_agent/tests/unit/test_statute_search.py
git commit -m "phase2a(retrievers): StatuteSearchTool with hybrid RRF + law filter"
```

---

## Task 9: exact_read Tool

**Files:**
- Create: `multi_agent/tools/retrievers/exact_read.py`
- Create: `tests/unit/test_exact_read.py`

**Purpose:** Retrieve a specific article by its `doc_id` (e.g. `民法典-510`) — for queries like "what does article 510 of the Civil Code say?" — no vector search needed.

- [ ] **Step 1: Write failing test**

```python
# tests/unit/test_exact_read.py
import uuid
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.exact_read import ExactReadTool, ExactReadArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    name = f"test_exact_{uuid.uuid4().hex[:8]}"
    tmp_dir = tmp_path_factory.mktemp("idx")
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典",
            source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=tmp_dir / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    yield name
    drop_collection(name)


@pytest.mark.asyncio
async def test_exact_read_finds_article(small_index, tmp_run_dir):
    tool = ExactReadTool(collection_name=small_index)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        ExactReadArgs(law_short="民法典", article_no="510"),
        rec,
    )
    rec.close()
    assert result.error is None
    ev = Evidence.model_validate(result.payload["evidence"])
    assert ev.doc_id == "民法典-510"
    assert ev.retriever == "exact"
    assert "合同补充" in ev.text


@pytest.mark.asyncio
async def test_exact_read_missing_returns_error(small_index, tmp_run_dir):
    tool = ExactReadTool(collection_name=small_index)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        ExactReadArgs(law_short="民法典", article_no="9999"),
        rec,
    )
    rec.close()
    assert result.error is not None
    assert "not found" in result.error.lower()
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_exact_read.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/exact_read.py`**

```python
"""Look up a specific article by (law_short, article_no).

Uses Qdrant's scroll with a payload filter — no embedding required.
"""
from __future__ import annotations
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client


class ExactReadArgs(BaseModel):
    law_short: str
    article_no: str


class ExactReadTool(Tool):
    name: str = "read_article"
    description: str = (
        "Look up the full text of a specific article by law name and number. "
        "Use when the user asks 'what does Article X of Law Y say'."
    )
    args_schema: type[BaseModel] = ExactReadArgs
    collection_name: str

    async def call(self, args: ExactReadArgs, recorder: Recorder) -> ToolResult:
        client = get_qdrant_client()
        result, _ = client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_short",
                        match=models.MatchValue(value=args.law_short),
                    ),
                    models.FieldCondition(
                        key="article_no",
                        match=models.MatchValue(value=args.article_no),
                    ),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not result:
            return ToolResult(
                tool_use_id="",
                payload=None,
                error=f"article not found: {args.law_short} 第 {args.article_no} 条",
            )
        payload = result[0].payload or {}
        ev = Evidence(
            doc_id=payload.get("doc_id", ""),
            law_name=payload.get("law_name", ""),
            article_no=payload.get("article_no", ""),
            text=payload.get("text", ""),
            score=1.0,                  # exact match
            retriever="exact",
            metadata={
                "law_short": payload.get("law_short", ""),
                "book": payload.get("book", ""),
                "chapter": payload.get("chapter", ""),
            },
        )
        return ToolResult(tool_use_id="", payload={"evidence": ev.model_dump()})
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest tests/unit/test_exact_read.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/exact_read.py experiments/multi_agent/tests/unit/test_exact_read.py
git commit -m "phase2a(retrievers): ExactReadTool for (law, article_no) lookup"
```

---

## Task 10: Integration Test — Retrieval E2E via Stub Agent

**Files:**
- Create: `tests/integration/test_retrieval_e2e.py`

**Purpose:** Phase 2a acceptance test. Builds a small index, wires `StatuteSearchTool` into a stub agent, runs a query end-to-end, and asserts:
1. The trace contains AgentInvoked → LLMRequested → ToolCalled (statute_search) → ToolReturned → LLMResponded → AgentResponded chain
2. The tool actually returned evidence
3. The recorder file has the correct parent_id chain (validating ContextVar fix in real flow)

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_retrieval_e2e.py
"""Phase 2a acceptance test: stub agent uses statute_search tool against
a real (small) Qdrant index, full trace is consistent.
"""
import json
import uuid
from pathlib import Path
import pytest
from pydantic import BaseModel

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.messages import ToolCallRequest
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.runner import run_query


class _Output(BaseModel):
    summary: str


class _SearchAgent(BaseAgent):
    def system_prompt(self) -> str:
        return "Use statute_search then summarize."

    def output_schema(self):
        return _Output


@pytest.fixture(scope="module")
def populated_index(tmp_path_factory):
    name = f"test_e2e_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    sparse_path = tmp / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照交易习惯确定。"),
                Chunk(doc_id="民法典-563", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="563",
                      text="一方违约时，对方可以解除合同。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_retrieval_e2e_trace_invariants(populated_index, tmp_path):
    runs_root = tmp_path / "runs"
    search_tool = StatuteSearchTool(
        collection_name=populated_index["collection"],
        sparse_artifact_path=populated_index["sparse_path"],
    )

    # Scripted: first response calls statute_search, second produces final answer
    provider = StubProvider(responses=[
        ScriptedResponse(
            text="",
            tool_calls=[ToolCallRequest(
                tool_use_id="t1", tool_name="statute_search",
                args={"query": "合同补充", "k": 2},
            )],
            finish_reason="tool_use",
        ),
        ScriptedResponse(
            text='{"summary": "found contract supplementation rules"}',
            finish_reason="end_turn",
        ),
    ])

    result = await run_query(
        query="搜索合同补充相关法条",
        agent_factory=lambda p, r: _SearchAgent(
            name="searcher", role="lookup",
            provider=p, recorder=r,
            tools=[search_tool],
        ),
        provider=provider,
        runs_root=runs_root,
        config={"profile": "stub+retrieval"},
    )

    assert result["status"] == "ok"
    run_dir = runs_root / result["run_id"]
    events = [json.loads(l) for l in (run_dir / "events.jsonl").read_text().splitlines()]
    types = [e["event_type"] for e in events]

    # Required chain: RunStarted → AgentInvoked → LLMRequested → LLMResponded
    # → ToolCalled → ToolReturned → LLMRequested → LLMResponded
    # → AgentResponded → RunFinished
    assert types[0] == "RunStarted"
    assert types[-1] == "RunFinished"
    assert "ToolCalled" in types
    assert "ToolReturned" in types

    # Tool call must be statute_search and produce a non-empty result
    tool_called = next(e for e in events if e["event_type"] == "ToolCalled")
    assert tool_called["tool_name"] == "statute_search"

    tool_returned = next(e for e in events if e["event_type"] == "ToolReturned")
    assert tool_returned["error"] is None
    assert tool_returned["result"]["count"] >= 1
    # At least one of the returned evidences is the contract article
    doc_ids = [e["doc_id"] for e in tool_returned["result"]["evidences"]]
    assert "民法典-510" in doc_ids

    # ContextVar invariant: ToolCalled.parent_id must be the agent_invoke span_id
    agent_invoked = next(e for e in events if e["event_type"] == "AgentInvoked")
    assert tool_called["parent_id"] == agent_invoked["event_id"]
```

- [ ] **Step 2: Run integration test**

```bash
docker ps | grep legal-rag-qdrant   # confirm running
cd /home/xxm/rag/experiments/multi_agent
pytest tests/integration/test_retrieval_e2e.py -v
```

Expected: 1 passed (the test). Will take ~10-20s because of bge-m3 inference time on a small batch.

- [ ] **Step 3: Run the FULL test suite as Phase 2a acceptance**

```bash
cd /home/xxm/rag/experiments/multi_agent
pytest -v
```

Expected: all tests pass. Phase 1 had 63 tests; Phase 2a adds approximately 30 more, so total should be ~93. Last line of pytest output should say `N passed in X.Xs`.

- [ ] **Step 4: Commit and tag**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_retrieval_e2e.py
git commit -m "phase2a(integration): retrieval E2E via stub agent + statute_search"
git tag -a phase2a-statute-retrieval -m "Phase 2a complete: Qdrant + bge-m3 + jieba + statutes collection + search tools"
```

---

## Acceptance Criteria

Phase 2a is complete when:

1. `pytest -v` from `experiments/multi_agent/` runs all tests green (Phase 1 + Phase 2a, ~93 total)
2. `python -m scripts.build_statutes_index --limit 2` succeeds against the shared `legal-rag-qdrant` container (port 6433) and creates a `ma_statutes_smoke` collection
3. The integration test `test_retrieval_e2e_trace_invariants` proves a stub agent can call `statute_search` and the trace chain stays consistent
4. ContextVar fix verified by `test_concurrent_spans_have_independent_parents`
5. Tag `phase2a-statute-retrieval` exists
6. `legal_statutes` collection (legacy from `legal_rag/`) is untouched

## Out-of-Scope (Reminder)

These DO NOT need to work after Phase 2a:
- Real LLM providers (Anthropic / Qwen) — Phase 2b
- Real Lawyer agent with five-section prompt — Phase 2c
- cases / user_history collections — Phase 2d
- Streaming output via `complete_stream` — Phase 2b
- Concepts field auto-generation — needs Phase 2b's local LLM

## Notes for Implementing Engineer

- **First bge-m3 download is slow.** ~2.3 GB from HuggingFace. Plan for 10-30 minutes on first encoder instantiation. The model is cached at `~/.cache/huggingface/` and reused.
- **Qdrant is the shared `legal-rag-qdrant` container** on host port 6433 — NOT a new container, NOT default port 6333. If it isn't running: `docker start legal-rag-qdrant`.
- **All multi_agent collections use `ma_` prefix** (`ma_statutes`, future `ma_cases`, `ma_user_history`). The existing `legal_statutes` collection from legacy code is read-only territory for this project — never drop it, never write to it.
- **Each integration/unit test builds a unique temporary collection** (`test_<uuid>`) and drops it on teardown. Persistent collections (`ma_statutes`, `ma_statutes_smoke`) created by the CLI script are separate — tests must not drop them.
- **If `pip install -e ".[dev]"` pulls torch over slow internet**, the whole bootstrap may take 30+ minutes. Pre-download torch into pip cache if possible.
- **Don't add concepts generation here** — even though spec §4.4 mentions it as "V0 跑", running it requires the local Qwen (Phase 2b). Phase 2a leaves the `concepts` field empty; Phase 2b or later fills it.
