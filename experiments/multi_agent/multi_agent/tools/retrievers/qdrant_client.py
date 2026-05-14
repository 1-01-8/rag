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
        _client = QdrantClient(url=url, timeout=30, check_compatibility=False)
    return _client


@dataclass(frozen=True)
class CollectionParams:
    """Shape of one Qdrant collection: dense dim + which sparse name."""
    dense_dim: int
    dense_name: str = "dense"
    sparse_name: str = "sparse"


# Shape used for `ma_statutes`, `ma_cases`, `ma_user_history` (all spec §4.2).
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
