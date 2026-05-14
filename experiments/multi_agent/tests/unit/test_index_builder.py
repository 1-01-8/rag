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
