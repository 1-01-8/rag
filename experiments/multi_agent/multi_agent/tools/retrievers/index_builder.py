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
        # Pass batch_size to encode_batch so sentence-transformers doesn't
        # split into a larger internal batch and OOM the GPU.
        dense_vecs = dense_encoder.encode_batch(
            [c.embedding_text() for c in batch], batch_size=batch_size,
        )
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
