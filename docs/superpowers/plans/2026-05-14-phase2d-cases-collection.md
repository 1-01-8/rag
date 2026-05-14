# Phase 2d — Cases Collection + Multi-Source Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add `ma_cases` Qdrant collection populated from `laws_data` Q&A pairs (after LLM extraction of cited articles), plus `case_search` and `all_sources_search` tools that Lawyer can use alongside `statute_search`. Resolves Phase 2c review items (budget rejections, multi-tool tool-first message). Defers `ma_user_history` to Phase 3 (needs MarkdownMemoryStore).

**Architecture:** Additive over Phase 2c. `case_search` returns Evidence with `retriever="case"`. `all_sources_search` fuses statutes + cases via Qdrant RRF over two collections (Qdrant 1.12 doesn't support cross-collection prefetch natively — we'll call each collection then RRF locally).

**Spec reference:** §4.2.1 (collections), §4.5 (query rewriting), ADR-22.

**Phase 2c starting point:** Tag `phase2c-real-lawyer`. 130 tests pass + 1 skipped.

---

## Out of scope

- `ma_user_history` (needs MarkdownMemoryStore — Phase 3)
- LLM-driven query rewriting (HyDE) — Phase 2c+ enhancement
- Concept-weighted sparse search — Phase 2c+ enhancement
- LLM extraction at full scale (23k records) — Phase 2d only does small batch (100-500 records) for proof of concept; full extraction is operational task

---

## File Structure (Phase 2d additions)

```
experiments/multi_agent/
├── multi_agent/
│   ├── agents/base.py                          # MODIFY: max_pre_tool_rejections; parameterize tool name
│   ├── tools/
│   │   ├── retrievers/
│   │   │   ├── case_search.py                  # NEW
│   │   │   └── all_sources_search.py           # NEW
│   │   └── laws_data_loader.py                 # NEW: parse laws_data zip
│   └── schemas/case.py                         # NEW: CaseQA schema
├── scripts/
│   ├── extract_case_citations.py               # NEW: LLM batch extraction
│   └── build_cases_index.py                    # NEW: cases collection builder
└── tests/
    ├── unit/
    │   ├── test_case_schema.py
    │   ├── test_laws_data_loader.py
    │   ├── test_case_search.py
    │   ├── test_all_sources_search.py
    │   └── test_base_pre_tool_budget.py
    └── integration/
        └── test_lawyer_multi_source_e2e.py
```

---

## Task 0: Phase 2c Follow-Up (Pre-Tool Rejection Budget + Tool Name Parameterization)

**Files:**
- Modify: `multi_agent/agents/base.py`
- Create: `tests/unit/test_base_pre_tool_budget.py`

Phase 2c review flagged: a pathological model that ignores tool-first instructions loops up to `max_steps` burning LLM calls. Add `max_pre_tool_rejections` (default 2) that fails fast.

Also: the tool-first redirect message hardcodes "statute_search" — make it use the first available tool name.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_base_pre_tool_budget.py
import pytest
from pydantic import BaseModel
from multi_agent.agents.base import BaseAgent, AgentInput
from multi_agent.providers.stub import StubProvider, ScriptedResponse
from multi_agent.tools.base import Tool
from multi_agent.schemas.messages import ToolResult
from multi_agent.tracing.recorder import Recorder
from multi_agent.errors import BudgetExceeded


class _Args(BaseModel):
    q: str


class _Tool(Tool):
    name: str = "echo"
    description: str = "echo"
    args_schema: type[BaseModel] = _Args

    async def call(self, args, recorder):
        return ToolResult(tool_use_id="x", payload={"echo": args.q})


class _Out(BaseModel):
    a: str


class _Agent(BaseAgent):
    def system_prompt(self) -> str:
        return "test"
    def output_schema(self):
        return _Out


@pytest.mark.asyncio
async def test_pre_tool_rejection_budget_fires(tmp_run_dir):
    """If model keeps answering without calling tools, BudgetExceeded should fire
    on max_pre_tool_rejections — not silently loop until max_steps."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    # 5 responses that all give final answer without tool calls
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"a": "fake"}', finish_reason="end_turn")
        for _ in range(5)
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   tools=[_Tool()], max_pre_tool_rejections=2, max_steps=10)
    with pytest.raises(BudgetExceeded) as exc:
        await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert exc.value.budget == "max_pre_tool_rejections"
    assert exc.value.limit == 2


@pytest.mark.asyncio
async def test_no_tools_no_rejection_budget(tmp_run_dir):
    """Agent with no tools accepts direct answer (no budget applies)."""
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    provider = StubProvider(responses=[
        ScriptedResponse(text='{"a": "ok"}', finish_reason="end_turn"),
    ])
    agent = _Agent(name="a", role="t", provider=provider, recorder=rec,
                   max_pre_tool_rejections=2)
    out = await agent.run(AgentInput(payload={"query": "hi"}))
    rec.close()
    assert out.payload.a == "ok"
```

- [ ] **Step 2: Verify failure**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_base_pre_tool_budget.py -v"
```

Expected: FAIL — either field doesn't exist or budget doesn't fire.

- [ ] **Step 3: Modify `BaseAgent`**

In `multi_agent/agents/base.py`:

a) Add field to `BaseAgent`:
```python
    max_pre_tool_rejections: int = 2
```

b) In `_react_loop`, find the existing tool-first enforcement block. Find this code (search for the section that handles the case where tools exist but model gave final answer without calling them):

The existing code probably looks like:
```python
        if tool_specs and tool_calls_made == 0:
            # silently discard the answer, inject redirect
            messages.append(AgentMessage(role="user", content="⚠️ ... statute_search ..."))
            continue
```

Modify to track rejection count:

```python
        pre_tool_rejections = 0   # add this initialization before the loop
```

And in the rejection branch:
```python
        if tool_specs and tool_calls_made == 0:
            pre_tool_rejections += 1
            if pre_tool_rejections > self.max_pre_tool_rejections:
                from multi_agent.errors import BudgetExceeded
                raise BudgetExceeded(
                    self.name, "max_pre_tool_rejections", self.max_pre_tool_rejections,
                )
            # Use the first available tool name in the redirect rather than hardcoded statute_search
            first_tool_name = self.tools[0].name if self.tools else "<a tool>"
            messages.append(AgentMessage(
                role="user",
                content=f"⚠️ 错误:你必须先调用 {first_tool_name} 工具检索后才能回答。请立即调用 {first_tool_name}。",
            ))
            continue
```

Read the existing code to understand the exact structure before editing. Match the variable names and indentation.

- [ ] **Step 4: Verify pass + full suite**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/unit/test_base_pre_tool_budget.py -v && pytest -v 2>&1 | tail -5"
```

Expected: 132 passed + 1 skipped (130 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/agents/base.py experiments/multi_agent/tests/unit/test_base_pre_tool_budget.py
git commit -m "phase2d(agents): max_pre_tool_rejections budget + parameterize tool name in redirect"
```

---

## Task 1: CaseQA Schema

**Files:**
- Create: `multi_agent/schemas/case.py`
- Create: `tests/unit/test_case_schema.py`

`CaseQA` represents one row from `laws_data` after LLM extraction of cited articles.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_case_schema.py
from multi_agent.schemas.case import CaseQA


def test_caseqa_required_fields():
    c = CaseQA(
        case_id="train_001234",
        cause="房产纠纷",
        question="房东要涨房租怎么办?",
        answer="可以与房东协商,不成可起诉。",
        extracted_cite_ids=["民法典-510", "民法典-563"],
    )
    assert c.case_id == "train_001234"
    assert c.cause == "房产纠纷"
    assert len(c.extracted_cite_ids) == 2


def test_caseqa_optional_fields_default():
    c = CaseQA(
        case_id="x", cause="y", question="q", answer="a",
        extracted_cite_ids=[],
    )
    assert c.candidate_answers == []
    assert c.extraction_confidence == 0.0
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/schemas/case.py`**

```python
"""Case Q&A schema — one entry from laws_data after LLM extraction."""
from __future__ import annotations
from pydantic import BaseModel, Field


class CaseQA(BaseModel):
    """A single legal Q&A pair with extracted citations.

    Sourced from laws_data train/*.json. The `extracted_cite_ids` field is
    populated by Task 2's extraction script — list of doc_ids (e.g.
    "民法典-510") that the lawyer answer references.
    """
    case_id: str                                    # e.g. "train_001234"
    cause: str                                      # 5 categories: 交通事故 / 婚姻家庭 / 债权债务 / 劳动纠纷 / 房产纠纷
    question: str
    answer: str                                     # primary lawyer answer
    candidate_answers: list[str] = Field(default_factory=list)
    extracted_cite_ids: list[str] = Field(default_factory=list)  # ["民法典-510", ...]
    extraction_confidence: float = 0.0              # 0..1, from extraction LLM
```

- [ ] **Step 4: Verify pass** → 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/schemas/case.py experiments/multi_agent/tests/unit/test_case_schema.py
git commit -m "phase2d(schemas): CaseQA schema for laws_data Q&A pairs"
```

---

## Task 2: laws_data Loader

**Files:**
- Create: `multi_agent/tools/laws_data_loader.py`
- Create: `tests/unit/test_laws_data_loader.py`

Reads `/home/xxm/rag/laws_data/*.zip`, iterates the `train/*.json` files, filters out `cause=劳动纠纷` (per ADR-15 corpus gap), and yields `CaseQA` objects (with empty `extracted_cite_ids` — Task 3 populates).

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_laws_data_loader.py
import pytest
import zipfile
import json
from pathlib import Path
from multi_agent.tools.laws_data_loader import iter_laws_data, filter_unsupported_causes
from multi_agent.schemas.case import CaseQA


@pytest.fixture
def fake_zip(tmp_path):
    """Create a tiny zip mimicking the laws_data train/ structure."""
    zpath = tmp_path / "fake.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("train/000001.json", json.dumps({
            "question": "房东涨租", "answer": "协商不成可起诉",
            "candidate_answer": ["协商", "起诉"], "cause": "房产纠纷",
        }))
        z.writestr("train/000002.json", json.dumps({
            "question": "被开除", "answer": "申请劳动仲裁",
            "candidate_answer": [], "cause": "劳动纠纷",
        }))
        z.writestr("train/000003.json", json.dumps({
            "question": "撞人了", "answer": "保险公司先赔",
            "candidate_answer": [], "cause": "交通事故",
        }))
    return zpath


def test_iter_yields_all_records(fake_zip):
    records = list(iter_laws_data(fake_zip))
    assert len(records) == 3
    for r in records:
        assert isinstance(r, CaseQA)
        assert r.extracted_cite_ids == []   # not extracted yet


def test_iter_records_have_correct_case_ids(fake_zip):
    records = list(iter_laws_data(fake_zip))
    assert {r.case_id for r in records} == {
        "train_000001", "train_000002", "train_000003",
    }


def test_filter_drops_unsupported_causes(fake_zip):
    """Drop 劳动纠纷 per ADR-15."""
    records = filter_unsupported_causes(iter_laws_data(fake_zip))
    causes = {r.cause for r in records}
    assert "劳动纠纷" not in causes
    assert "房产纠纷" in causes
    assert "交通事故" in causes
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/tools/laws_data_loader.py`**

```python
"""Iterate laws_data zip files into CaseQA objects.

Skips records where cause is in the unsupported set (per ADR-15: 劳动纠纷
because 劳动合同法 not in corpus). Extraction of cited articles is Task 3.
"""
from __future__ import annotations
import json
import zipfile
from pathlib import Path
from typing import Iterator
from multi_agent.schemas.case import CaseQA


UNSUPPORTED_CAUSES: frozenset[str] = frozenset({"劳动纠纷"})


def iter_laws_data(zip_path: Path) -> Iterator[CaseQA]:
    """Yield CaseQA from every train/*.json or test/*.json in the zip."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if not info.filename.endswith(".json"):
                continue
            # Extract the numeric ID from path like "train/000001.json"
            stem = Path(info.filename).stem            # "000001"
            split = Path(info.filename).parts[0]       # "train" / "test"
            case_id = f"{split}_{stem}"
            with z.open(info) as fh:
                raw = json.loads(fh.read())
            yield CaseQA(
                case_id=case_id,
                cause=raw.get("cause", "未知"),
                question=raw.get("question", ""),
                answer=raw.get("answer", ""),
                candidate_answers=raw.get("candidate_answer", []),
                extracted_cite_ids=[],
            )


def filter_unsupported_causes(records: Iterator[CaseQA]) -> Iterator[CaseQA]:
    """Filter out causes the corpus can't handle."""
    for r in records:
        if r.cause not in UNSUPPORTED_CAUSES:
            yield r
```

- [ ] **Step 4: Verify pass** → 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/laws_data_loader.py experiments/multi_agent/tests/unit/test_laws_data_loader.py
git commit -m "phase2d(tools): laws_data zip loader + 劳动 filter"
```

---

## Task 3: LLM Citation Extraction Script

**Files:**
- Create: `scripts/extract_case_citations.py`

A CLI script that takes N case records (or all), calls Qwen via OpenAICompatibleProvider to extract cited articles from each `answer`, and writes the enriched records to a JSONL file.

NOT a unit-tested module — it's a one-shot operational script. We just verify it runs end-to-end on 5 records.

- [ ] **Step 1: Write the script**

`/home/xxm/rag/experiments/multi_agent/scripts/extract_case_citations.py`:

```python
"""CLI: extract cited articles from laws_data Q&A using Qwen.

Usage:
    cd /home/xxm/rag/experiments/multi_agent
    python -m scripts.extract_case_citations \\
        --zip /home/xxm/rag/laws_data/<train.zip> \\
        --output indexes/cases_extracted.jsonl \\
        --limit 100
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

from multi_agent.tools.laws_data_loader import iter_laws_data, filter_unsupported_causes
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.providers.json_robust import parse_json_robust
from multi_agent.schemas.messages import AgentMessage
from multi_agent.tracing.recorder import Recorder


EXTRACTION_PROMPT = """你是法律文本分析器。从下面的律师答复中提取被引用的法条。

要求:
1. 仅提取明确引用的"法律名+条号"组合,如"民法典 第510条"、"道路交通安全法 第76条"
2. 标准化为 doc_id 格式: "<law_short>-<arabic_number>",如 "民法典-510"
3. 若无明确引用,返回空列表
4. 不要从案件描述中推测引用,只看律师答复

输出 JSON:
```json
{{
  "doc_ids": ["民法典-510", "道路交通安全法-76"],
  "confidence": 0.85
}}
```

律师答复:
{answer}
"""


async def extract_one(provider, answer: str, rec: Recorder, agent_name: str) -> tuple[list[str], float]:
    """Returns (doc_ids, confidence)."""
    try:
        resp = await provider.complete(
            messages=[AgentMessage(role="user", content=EXTRACTION_PROMPT.format(answer=answer))],
            model="qwen3.5-9b",
            max_tokens=256,
            temperature=0,
            recorder=rec,
            agent_name=agent_name,
        )
        parsed = parse_json_robust(resp.text)
        return parsed.get("doc_ids", []), float(parsed.get("confidence", 0.0))
    except Exception as e:
        print(f"  extraction failed: {e}")
        return [], 0.0


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="0 = all")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/extraction"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(run_id="extract", run_dir=args.run_dir)
    rec.set_meta(zip=str(args.zip), output=str(args.output))

    provider = OpenAICompatibleProvider()
    n_total = 0
    n_with_cites = 0

    with args.output.open("w", encoding="utf-8") as out:
        records = filter_unsupported_causes(iter_laws_data(args.zip))
        t0 = time.monotonic()
        for record in records:
            if args.limit and n_total >= args.limit:
                break
            doc_ids, conf = await extract_one(provider, record.answer, rec, "extractor")
            record = record.model_copy(update={
                "extracted_cite_ids": doc_ids,
                "extraction_confidence": conf,
            })
            out.write(record.model_dump_json() + "\n")
            out.flush()
            n_total += 1
            if doc_ids:
                n_with_cites += 1
            if n_total % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = n_total / elapsed
                print(f"  {n_total} done ({n_with_cites} with cites, {rate:.1f}/s)")

    rec.close()
    print(f"Done: {n_total} records, {n_with_cites} with citations, {n_with_cites/max(n_total,1)*100:.1f}% hit rate")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Smoke test on 5 records**

```bash
# Find the train zip
ls /home/xxm/rag/laws_data/*.zip
# Use the first one with substantial data; per Phase 2a-3 we observed train.zip has 16209 entries
# Pick whichever exists
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && python -m scripts.extract_case_citations --zip /home/xxm/rag/laws_data/9d3f0f41-c7df-4211-a32a-9e1fc74f8a68.zip --output indexes/cases_smoke.jsonl --limit 5"
```

Expected: prints "Done: 5 records, N with citations, X% hit rate"

Verify the output file:

```bash
wc -l /home/xxm/rag/experiments/multi_agent/indexes/cases_smoke.jsonl
head -1 /home/xxm/rag/experiments/multi_agent/indexes/cases_smoke.jsonl | python -m json.tool | head -15
```

Each line should be a valid CaseQA JSON. Some entries may have empty `extracted_cite_ids` (lawyer answer didn't cite anything) — that's expected.

- [ ] **Step 3: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/scripts/extract_case_citations.py
git commit -m "phase2d(scripts): laws_data citation extraction CLI"
```

The smoke output file `indexes/cases_smoke.jsonl` is gitignored.

---

## Task 4: Build Cases Qdrant Index

**Files:**
- Create: `scripts/build_cases_index.py`
- Create: `tests/unit/test_cases_index_build.py` (uses tiny fixture, no real Qwen)

Reads the extracted JSONL from Task 3, encodes each `question` with dense+sparse, upserts into `ma_cases` collection.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_cases_index_build.py
import json
import uuid
import pytest
from multi_agent.schemas.case import CaseQA
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client, drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


@pytest.fixture
def cases_jsonl(tmp_path):
    """Write a tiny extracted-cases JSONL."""
    p = tmp_path / "cases.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租怎么办?",
            answer="协商不成可起诉。", extracted_cite_ids=["民法典-510"],
            extraction_confidence=0.9,
        ).model_dump_json() + "\n")
        f.write(CaseQA(
            case_id="train_002", cause="交通事故",
            question="撞了人住院要赔多少?",
            answer="参照伤残等级和实际损失。", extracted_cite_ids=["道路交通安全法-76"],
            extraction_confidence=0.85,
        ).model_dump_json() + "\n")
    return p


@pytest.fixture
def temp_collection():
    name = f"test_cases_{uuid.uuid4().hex[:8]}"
    yield name
    drop_collection(name)


def test_build_cases_index_creates_points(cases_jsonl, temp_collection, tmp_path):
    from scripts.build_cases_index import build_cases_index
    artifacts = build_cases_index(
        jsonl_path=cases_jsonl,
        collection_name=temp_collection,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    client = get_qdrant_client()
    count = client.count(temp_collection).count
    assert count == 2


def test_build_cases_index_payload_has_extracted_cites(cases_jsonl, temp_collection, tmp_path):
    from scripts.build_cases_index import build_cases_index
    build_cases_index(
        jsonl_path=cases_jsonl,
        collection_name=temp_collection,
        sparse_artifact_path=tmp_path / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    client = get_qdrant_client()
    # Scroll all and verify payload structure
    points, _ = client.scroll(collection_name=temp_collection, limit=10, with_payload=True)
    for pt in points:
        assert "case_id" in pt.payload
        assert "cause" in pt.payload
        assert "extracted_cite_ids" in pt.payload
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `scripts/build_cases_index.py`**

```python
"""Build ma_cases Qdrant collection from extracted JSONL.

Encodes each `question` with dense (bge-m3 on enriched text) + sparse (jieba+IDF).
Payload preserves cause, question, answer, extracted_cite_ids for later use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from qdrant_client import models

from multi_agent.schemas.case import CaseQA
from multi_agent.tools.retrievers.qdrant_client import (
    get_qdrant_client, ensure_collection, STATUTE_COLLECTION_PARAMS,
)
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


def _point_id(case_id: str) -> int:
    h = hashlib.sha256(case_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") >> 1


@dataclass
class CasesIndexArtifacts:
    collection_name: str
    sparse_artifact_path: Path
    n_indexed: int


def _read_cases(jsonl_path: Path) -> Iterable[CaseQA]:
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield CaseQA.model_validate_json(line)


def _embedding_text(case: CaseQA) -> str:
    """Text used for dense embedding: question + cause for context."""
    return f"[{case.cause}] {case.question}"


def build_cases_index(
    *,
    jsonl_path: Path,
    collection_name: str,
    sparse_artifact_path: Path,
    dense_encoder: DenseEncoder,
    batch_size: int = 64,
) -> CasesIndexArtifacts:
    cases = list(_read_cases(jsonl_path))
    if not cases:
        raise ValueError(f"no cases in {jsonl_path}")

    sparse_enc = SparseEncoder()
    sparse_enc.fit(c.question for c in cases)

    ensure_collection(collection_name, STATUTE_COLLECTION_PARAMS)
    client = get_qdrant_client()

    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        dense_vecs = dense_encoder.encode_batch([_embedding_text(c) for c in batch])
        points = []
        for case, dense_vec in zip(batch, dense_vecs):
            sparse_vec = sparse_enc.encode(case.question)
            points.append(models.PointStruct(
                id=_point_id(case.case_id),
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values,
                    ),
                },
                payload={
                    "case_id": case.case_id,
                    "cause": case.cause,
                    "question": case.question,
                    "answer": case.answer,
                    "candidate_answers": case.candidate_answers,
                    "extracted_cite_ids": case.extracted_cite_ids,
                    "extraction_confidence": case.extraction_confidence,
                },
            ))
        client.upsert(collection_name=collection_name, points=points)

    Path(sparse_artifact_path).parent.mkdir(parents=True, exist_ok=True)
    sparse_enc.save(sparse_artifact_path)
    return CasesIndexArtifacts(
        collection_name=collection_name,
        sparse_artifact_path=sparse_artifact_path,
        n_indexed=len(cases),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--collection", default="ma_cases")
    parser.add_argument("--sparse-out", type=Path,
                        default=Path("indexes/ma_cases_sparse.json"))
    args = parser.parse_args()
    artifacts = build_cases_index(
        jsonl_path=args.jsonl, collection_name=args.collection,
        sparse_artifact_path=args.sparse_out, dense_encoder=DenseEncoder(),
    )
    print(f"Indexed {artifacts.n_indexed} cases into {artifacts.collection_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify pass** → 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/scripts/build_cases_index.py experiments/multi_agent/tests/unit/test_cases_index_build.py
git commit -m "phase2d(scripts): build_cases_index for ma_cases collection"
```

---

## Task 5: CaseSearchTool

**Files:**
- Create: `multi_agent/tools/retrievers/case_search.py`
- Create: `tests/unit/test_case_search.py`

Mirrors `StatuteSearchTool` but queries `ma_cases` and returns Evidence with `retriever="case"`. Payload exposes `question` and `answer` differently — we render them into Evidence.text.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_case_search.py
import json
import uuid
import pytest
from pathlib import Path

from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.case_search import CaseSearchTool, CaseSearchArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def case_index(tmp_path_factory):
    name = f"test_case_search_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    jsonl = tmp / "cases.jsonl"
    sparse_path = tmp / "sparse.json"
    with jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租 30%,合不合法?",
            answer="一般不合法,可拒绝并起诉。",
            extracted_cite_ids=["民法典-510"],
            extraction_confidence=0.9,
        ).model_dump_json() + "\n")
        f.write(CaseQA(
            case_id="train_002", cause="交通事故",
            question="撞了人住院要赔多少钱?",
            answer="参照伤残等级和实际损失,先走保险。",
            extracted_cite_ids=["道路交通安全法-76"],
            extraction_confidence=0.85,
        ).model_dump_json() + "\n")
    from scripts.build_cases_index import build_cases_index
    build_cases_index(
        jsonl_path=jsonl, collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_case_search_returns_evidence(case_index, tmp_run_dir):
    tool = CaseSearchTool(
        collection_name=case_index["collection"],
        sparse_artifact_path=case_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(CaseSearchArgs(query="房东涨租", k=2), rec)
    rec.close()
    assert result.error is None
    evidences = result.payload["evidences"]
    assert len(evidences) >= 1
    top = Evidence.model_validate(evidences[0])
    assert top.retriever == "case"
    # Evidence.text should contain the question and answer
    assert "涨房租" in top.text or "涨租" in top.text


@pytest.mark.asyncio
async def test_case_search_filter_by_cause(case_index, tmp_run_dir):
    tool = CaseSearchTool(
        collection_name=case_index["collection"],
        sparse_artifact_path=case_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        CaseSearchArgs(query="撞人", k=5, cause="交通事故"),
        rec,
    )
    rec.close()
    for h in result.payload["evidences"]:
        assert Evidence.model_validate(h).metadata.get("cause") == "交通事故"
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/case_search.py`**

```python
"""Search ma_cases collection (laws_data Q&A pairs).

Returns Evidence whose .text combines question + answer for downstream
agents to ingest as 'similar case' context.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from qdrant_client import models

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.qdrant_client import get_qdrant_client
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder


class CaseSearchArgs(BaseModel):
    query: str
    k: int = 5
    cause: str | None = None


class CaseSearchTool(Tool):
    name: str = "case_search"
    description: str = (
        "Search past legal Q&A cases. Returns similar real-world cases "
        "(question + lawyer answer + extracted citations). Optional cause filter."
    )
    args_schema: type[BaseModel] = CaseSearchArgs
    collection_name: str
    sparse_artifact_path: Path

    _dense: Any = None
    _sparse: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def _ensure_encoders(self) -> None:
        if self._dense is None:
            object.__setattr__(self, "_dense", DenseEncoder())
        if self._sparse is None:
            object.__setattr__(self, "_sparse", SparseEncoder.load(self.sparse_artifact_path))

    async def call(self, args: CaseSearchArgs, recorder: Recorder) -> ToolResult:
        self._ensure_encoders()
        client = get_qdrant_client()
        dense_vec = self._dense.encode_one(args.query).tolist()
        sparse_vec = self._sparse.encode(args.query)

        query_filter = None
        if args.cause:
            query_filter = models.Filter(must=[
                models.FieldCondition(
                    key="cause", match=models.MatchValue(value=args.cause),
                )
            ])

        result = client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense",
                                limit=max(args.k * 2, 20), filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices, values=sparse_vec.values,
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
            text = f"[问题] {payload.get('question', '')}\n[律师答复] {payload.get('answer', '')}"
            ev = Evidence(
                doc_id=payload.get("case_id", ""),
                law_name="(case)",
                law_short="",
                article_no=payload.get("case_id", ""),
                text=text,
                score=float(point.score) if point.score is not None else 0.0,
                retriever="case",
                metadata={
                    "cause": payload.get("cause", ""),
                    "extracted_cite_ids": payload.get("extracted_cite_ids", []),
                    "extraction_confidence": payload.get("extraction_confidence", 0.0),
                },
            )
            evidences.append(ev.model_dump())

        return ToolResult(
            tool_use_id="",
            payload={"evidences": evidences, "count": len(evidences)},
        )
```

- [ ] **Step 4: Verify pass** → 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/case_search.py experiments/multi_agent/tests/unit/test_case_search.py
git commit -m "phase2d(retrievers): CaseSearchTool with cause filter + hybrid RRF over ma_cases"
```

---

## Task 6: AllSourcesSearchTool

**Files:**
- Create: `multi_agent/tools/retrievers/all_sources_search.py`
- Create: `tests/unit/test_all_sources_search.py`

Queries both `statutes` and `cases` collections, merges results via local RRF (Qdrant 1.12 doesn't support cross-collection prefetch). Returns merged Evidence list.

- [ ] **Step 1: Failing test**

```python
# tests/unit/test_all_sources_search.py
import uuid
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.all_sources_search import (
    AllSourcesSearchTool, AllSourcesArgs,
)
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def both_indexes(tmp_path_factory):
    statutes_name = f"test_s_{uuid.uuid4().hex[:8]}"
    cases_name = f"test_c_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    s_sparse = tmp / "s_sparse.json"
    c_sparse = tmp / "c_sparse.json"

    encoder = DenseEncoder()

    # Build statutes
    stat_docs = [Document(
        law_name="中华人民共和国民法典", law_short="民法典", source_path="t",
        chunks=[
            Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                  law_short="民法典", article_no="510",
                  text="当事人就合同补充内容没有约定的,按照交易习惯确定。"),
        ],
    )]
    build_index(documents=stat_docs, collection_name=statutes_name,
                sparse_artifact_path=s_sparse, dense_encoder=encoder)

    # Build cases
    cases_jsonl = tmp / "cases.jsonl"
    with cases_jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东要涨房租 30%,合不合法?",
            answer="一般不合法,可拒绝。",
            extracted_cite_ids=["民法典-510"],
        ).model_dump_json() + "\n")
    from scripts.build_cases_index import build_cases_index
    build_cases_index(jsonl_path=cases_jsonl, collection_name=cases_name,
                     sparse_artifact_path=c_sparse, dense_encoder=encoder)

    yield {
        "statutes": statutes_name, "statutes_sparse": s_sparse,
        "cases": cases_name, "cases_sparse": c_sparse,
    }
    drop_collection(statutes_name)
    drop_collection(cases_name)


@pytest.mark.asyncio
async def test_all_sources_returns_mixed_evidence(both_indexes, tmp_run_dir):
    tool = AllSourcesSearchTool(
        statutes_collection=both_indexes["statutes"],
        statutes_sparse=both_indexes["statutes_sparse"],
        cases_collection=both_indexes["cases"],
        cases_sparse=both_indexes["cases_sparse"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        AllSourcesArgs(query="房东涨租 民法典 第510条", k=5),
        rec,
    )
    rec.close()
    assert result.error is None
    evidences = result.payload["evidences"]
    retrievers = {Evidence.model_validate(e).retriever for e in evidences}
    # Both statute and case results should appear
    assert "hybrid" in retrievers or "case" in retrievers
    # If both indexes contribute, we expect both kinds
    assert len(evidences) >= 1
```

- [ ] **Step 2: Verify failure** → ImportError.

- [ ] **Step 3: Create `multi_agent/tools/retrievers/all_sources_search.py`**

```python
"""Cross-collection retrieval: statutes + cases, merged via local RRF.

Qdrant 1.12 doesn't support cross-collection prefetch, so we query each
collection separately and merge.
"""
from __future__ import annotations
from pathlib import Path
from pydantic import BaseModel

from multi_agent.schemas.messages import ToolResult
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.base import Tool
from multi_agent.tracing.recorder import Recorder
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool, StatuteSearchArgs
from multi_agent.tools.retrievers.case_search import CaseSearchTool, CaseSearchArgs


class AllSourcesArgs(BaseModel):
    query: str
    k: int = 8
    law_short: str | None = None
    cause: str | None = None


def _rrf_merge(lists: list[list[Evidence]], k_constant: int = 60, top_k: int = 8) -> list[Evidence]:
    """Reciprocal Rank Fusion across multiple ranked Evidence lists.

    Keeps evidences keyed by doc_id; sums 1/(k_constant + rank) across all
    input lists. Returns top_k by fused score.
    """
    fused: dict[str, tuple[Evidence, float]] = {}
    for lst in lists:
        for rank, ev in enumerate(lst):
            score_contribution = 1.0 / (k_constant + rank)
            if ev.doc_id in fused:
                existing_ev, existing_score = fused[ev.doc_id]
                fused[ev.doc_id] = (existing_ev, existing_score + score_contribution)
            else:
                fused[ev.doc_id] = (ev, score_contribution)
    # Sort by fused score descending
    ranked = sorted(fused.values(), key=lambda x: -x[1])[:top_k]
    # Override Evidence.score with fused score for transparency
    return [ev.model_copy(update={"score": float(score)}) for ev, score in ranked]


class AllSourcesSearchTool(Tool):
    name: str = "all_sources_search"
    description: str = (
        "Search across BOTH statutes and case law (Q&A pairs) simultaneously. "
        "Results are fused via reciprocal rank fusion. Optional filters: "
        "law_short (limits statute results), cause (limits case results)."
    )
    args_schema: type[BaseModel] = AllSourcesArgs
    statutes_collection: str
    statutes_sparse: Path
    cases_collection: str
    cases_sparse: Path

    async def call(self, args: AllSourcesArgs, recorder: Recorder) -> ToolResult:
        # Query both collections in parallel? For simplicity, sequential here;
        # asyncio.gather adds complexity around span ordering. Sequential is fine.
        statute_tool = StatuteSearchTool(
            collection_name=self.statutes_collection,
            sparse_artifact_path=self.statutes_sparse,
        )
        case_tool = CaseSearchTool(
            collection_name=self.cases_collection,
            sparse_artifact_path=self.cases_sparse,
        )

        stat_result = await statute_tool.call(
            StatuteSearchArgs(query=args.query, k=args.k, law_short=args.law_short),
            recorder,
        )
        case_result = await case_tool.call(
            CaseSearchArgs(query=args.query, k=args.k, cause=args.cause),
            recorder,
        )

        stat_evs = [Evidence.model_validate(e) for e in (stat_result.payload or {}).get("evidences", [])]
        case_evs = [Evidence.model_validate(e) for e in (case_result.payload or {}).get("evidences", [])]

        fused = _rrf_merge([stat_evs, case_evs], top_k=args.k)
        return ToolResult(
            tool_use_id="",
            payload={
                "evidences": [e.model_dump() for e in fused],
                "count": len(fused),
                "stats": {"statutes": len(stat_evs), "cases": len(case_evs)},
            },
        )
```

- [ ] **Step 4: Verify pass** → 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/multi_agent/tools/retrievers/all_sources_search.py experiments/multi_agent/tests/unit/test_all_sources_search.py
git commit -m "phase2d(retrievers): AllSourcesSearchTool with local RRF across statutes+cases"
```

---

## Task 7: Multi-Source E2E (Lawyer with statute_search + case_search)

**Files:**
- Create: `tests/integration/test_lawyer_multi_source_e2e.py`

The flagship Phase 2d test: LawyerAgent with both `statute_search` and `case_search` tools. Verifies Lawyer can choose retrieval strategy autonomously.

- [ ] **Step 1: Write test** (similar shape to Phase 2c civil E2E but uses both collections)

```python
# tests/integration/test_lawyer_multi_source_e2e.py
"""Phase 2d flagship: LawyerAgent uses statute_search + case_search."""
import json
import uuid
import httpx
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.case import CaseQA
from multi_agent.schemas.lawyer import LawyerOutput
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool
from multi_agent.tools.retrievers.case_search import CaseSearchTool
from multi_agent.providers.openai_compatible import OpenAICompatibleProvider
from multi_agent.agents.lawyer import LawyerAgent
from multi_agent.runner import run_query


def _qwen_reachable() -> bool:
    try:
        with httpx.Client(timeout=2.0) as c:
            return c.get("http://localhost:8000/v1/models").status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _qwen_reachable(), reason="Qwen vLLM not running")


@pytest.fixture(scope="module")
def both_indexes(tmp_path_factory):
    stat = f"test_multi_s_{uuid.uuid4().hex[:8]}"
    case = f"test_multi_c_{uuid.uuid4().hex[:8]}"
    tmp = tmp_path_factory.mktemp("idx")
    encoder = DenseEncoder()

    stat_docs = [Document(
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
    s_sparse = tmp / "s_sparse.json"
    build_index(documents=stat_docs, collection_name=stat,
                sparse_artifact_path=s_sparse, dense_encoder=encoder)

    cases_jsonl = tmp / "cases.jsonl"
    with cases_jsonl.open("w", encoding="utf-8") as f:
        f.write(CaseQA(
            case_id="train_001", cause="房产纠纷",
            question="房东合同期内单方涨租,能拒绝吗?",
            answer="可以拒绝。合同期内租金条款受约束,涨租属于变更条款,需双方协商一致。",
            extracted_cite_ids=["民法典-510", "民法典-703"],
            extraction_confidence=0.92,
        ).model_dump_json() + "\n")
    c_sparse = tmp / "c_sparse.json"
    from scripts.build_cases_index import build_cases_index
    build_cases_index(jsonl_path=cases_jsonl, collection_name=case,
                     sparse_artifact_path=c_sparse, dense_encoder=encoder)

    yield {"stat": stat, "case": case, "s_sparse": s_sparse, "c_sparse": c_sparse}
    drop_collection(stat)
    drop_collection(case)


@pytest.mark.asyncio
async def test_lawyer_uses_both_tools(both_indexes, tmp_path):
    statute_search = StatuteSearchTool(
        collection_name=both_indexes["stat"],
        sparse_artifact_path=both_indexes["s_sparse"],
    )
    case_search = CaseSearchTool(
        collection_name=both_indexes["case"],
        sparse_artifact_path=both_indexes["c_sparse"],
    )
    provider = OpenAICompatibleProvider()

    runs_root = tmp_path / "runs"
    result = await run_query(
        query="我租房合同里没写能涨租,房东突然要涨 30%,合法吗?",
        agent_factory=lambda p, r: LawyerAgent(
            name="lawyer", role="advisor",
            provider=p, recorder=r,
            tools=[statute_search, case_search],
            model="qwen3.5-9b",
            specialty="房产",
            max_steps=10, max_tool_calls=12,
        ),
        provider=provider,
        runs_root=runs_root,
        config={"phase": "2d"},
    )

    assert result["status"] == "ok"
    out = LawyerOutput.model_validate(json.loads(result["final_answer"]))
    assert out.mode == "consultation"
    assert out.five_section is not None

    # No fabricated citations — must be from statute index
    indexed = {"民法典-510", "民法典-703"}
    for cit in out.citations:
        doc_id = f"{cit.law_short}-{cit.article_no}"
        assert doc_id in indexed, f"Fabricated: {doc_id}"

    # Verify at least one tool was called
    events = [json.loads(l) for l in (runs_root / result["run_id"] / "events.jsonl").read_text().splitlines()]
    tool_calls = [e for e in events if e["event_type"] == "ToolCalled"]
    assert len(tool_calls) >= 1
```

- [ ] **Step 2: Run**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest tests/integration/test_lawyer_multi_source_e2e.py -v"
```

Expected: PASS (60-180s).

- [ ] **Step 3: Full suite**

```bash
conda run -n qwen35 bash -c "cd /home/xxm/rag/experiments/multi_agent && pytest -v 2>&1 | tail -10"
```

Expected: ~140 passed + 1 skipped.

- [ ] **Step 4: Commit + tag**

```bash
cd /home/xxm/rag
git add experiments/multi_agent/tests/integration/test_lawyer_multi_source_e2e.py
git commit -m "phase2d(integration): LawyerAgent uses statute_search + case_search together"
git tag -a phase2d-cases-collection -m "Phase 2d complete: ma_cases collection + multi-source retrieval"
```

---

## Acceptance Criteria

Phase 2d complete when:

1. Full pytest passes (~140 tests)
2. `test_lawyer_uses_both_tools` proves Lawyer can use both retrieval tools without fabricating
3. `extract_case_citations.py` smoke-runs on 5 records successfully
4. `build_cases_index.py` smoke-creates `ma_cases_smoke` collection in Qdrant
5. Tag `phase2d-cases-collection` exists
6. Phase 2c review items resolved: `max_pre_tool_rejections` budget enforced + tool-first redirect uses first available tool name

## Out of Scope (Reminder)

- `ma_user_history` collection (needs MarkdownMemoryStore — Phase 3)
- Full 23k laws_data extraction (operational, not phase-task)
- HyDE / concept-weighted sparse (Phase 2c+ enhancement)
- Anthropic provider gotcha around consecutive user messages (Phase 4 concern when Supervisor lands)
