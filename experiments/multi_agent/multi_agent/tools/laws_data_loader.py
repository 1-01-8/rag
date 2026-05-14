"""Iterate laws_data zip files into CaseQA objects.

Skips records where cause is in the unsupported set (per ADR-15: 劳动纠纷
because 劳动合同法 not in corpus). Extraction of cited articles is Phase 2d Task 3.
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
            stem = Path(info.filename).stem
            split = Path(info.filename).parts[0]
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
