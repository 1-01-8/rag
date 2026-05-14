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
