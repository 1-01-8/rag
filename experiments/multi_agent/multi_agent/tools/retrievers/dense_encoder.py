"""bge-m3 dense embedding wrapper.

Default loads from a local model snapshot at /home/xxm/models/bge-m3
onto GPU card 1 (cuda:1). Override via constructor args or env vars
(BGE_M3_PATH, BGE_M3_DEVICE).
"""
from __future__ import annotations
import os
from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEncoder:
    """Wrap sentence-transformers bge-m3 for batch encoding."""

    DEFAULT_MODEL_PATH = "/home/xxm/models/bge-m3"
    DEFAULT_DEVICE = "cuda:1"
    DEFAULT_DIM = 1024

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ):
        model_name = (
            model_name
            or os.environ.get("BGE_M3_PATH")
            or self.DEFAULT_MODEL_PATH
        )
        device = (
            device
            or os.environ.get("BGE_M3_DEVICE")
            or self.DEFAULT_DEVICE
        )
        self._model = SentenceTransformer(model_name, device=device)
        # `get_embedding_dimension` is the current API; fall back to the
        # deprecated alias for older sentence-transformers versions.
        _dim_fn = getattr(
            self._model,
            "get_embedding_dimension",
            None,
        ) or getattr(self._model, "get_sentence_embedding_dimension", None)
        self.dim = (_dim_fn() if _dim_fn is not None else None) or self.DEFAULT_DIM

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
        """Encode a list of texts -> 2D matrix (N, dim) of unit vectors."""
        texts = list(texts)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
