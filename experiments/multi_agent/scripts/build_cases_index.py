"""Build ma_cases Qdrant collection from extracted JSONL."""
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
