"""TurnIndexer — embed Turn into ma_user_history (Phase 3d §4.2)."""
from __future__ import annotations
import uuid
from qdrant_client import models

from multi_agent.schemas.memory import Turn
from multi_agent.tools.retrievers.qdrant_client import (
    get_qdrant_client, ensure_collection, CollectionParams,
)
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


HISTORY_COLLECTION_PARAMS = CollectionParams(dense_dim=1024)

_NAMESPACE = uuid.UUID("e6bf57c0-3a8e-4d22-9b8f-5ac4f6e3b7d1")  # arbitrary fixed namespace


def _point_id(session_id: str, turn_no: int) -> str:
    """Deterministic UUID5 from (session_id, turn_no) — idempotent upserts."""
    return str(uuid.uuid5(_NAMESPACE, f"{session_id}:{turn_no}"))


class TurnIndexer:
    """Embed a completed Turn and upsert it into a Qdrant history collection."""

    def __init__(self, *, collection_name: str, dense_encoder: DenseEncoder):
        self.collection_name = collection_name
        self.dense_encoder = dense_encoder
        ensure_collection(collection_name, HISTORY_COLLECTION_PARAMS)

    async def index_turn(self, *, session_id: str, turn: Turn) -> None:
        """Dense-encode the turn and upsert into Qdrant.

        Text input: ``"<question>\\n<final_answer>"`` (trimmed).
        Point ID is deterministic via uuid5 — re-indexing the same turn just
        overwrites the existing point (idempotent).
        """
        text = f"{turn.question}\n{turn.final_answer or ''}".strip()
        # encode_batch returns shape (N, dim); slice row 0 for the single text
        vec = self.dense_encoder.encode_batch([text])[0]
        client = get_qdrant_client()
        client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=_point_id(session_id, turn.turn),
                    vector={"dense": vec.tolist()},
                    payload={
                        "session_id": session_id,
                        "turn_no": turn.turn,
                        "run_id": turn.run_id,
                        "question_preview": (turn.question or "")[:200],
                        "answer_preview": (turn.final_answer or "")[:300],
                        "started_at": (
                            turn.started_at.isoformat() if turn.started_at else None
                        ),
                    },
                ),
            ],
        )
