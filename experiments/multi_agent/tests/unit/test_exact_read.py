import uuid
import pytest

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.exact_read import ExactReadTool, ExactReadArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    name = f"test_exact_{uuid.uuid4().hex[:8]}"
    tmp_dir = tmp_path_factory.mktemp("idx")
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典",
            source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定。"),
            ],
        ),
    ]
    build_index(
        documents=docs, collection_name=name,
        sparse_artifact_path=tmp_dir / "sparse.json",
        dense_encoder=DenseEncoder(),
    )
    yield name
    drop_collection(name)


@pytest.mark.asyncio
async def test_exact_read_finds_article(small_index, tmp_run_dir):
    tool = ExactReadTool(collection_name=small_index)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        ExactReadArgs(law_short="民法典", article_no="510"),
        rec,
    )
    rec.close()
    assert result.error is None
    ev = Evidence.model_validate(result.payload["evidence"])
    assert ev.doc_id == "民法典-510"
    assert ev.retriever == "exact"
    assert "合同补充" in ev.text


@pytest.mark.asyncio
async def test_exact_read_missing_returns_error(small_index, tmp_run_dir):
    tool = ExactReadTool(collection_name=small_index)
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        ExactReadArgs(law_short="民法典", article_no="9999"),
        rec,
    )
    rec.close()
    assert result.error is not None
    assert "not found" in result.error.lower()
