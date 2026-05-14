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
