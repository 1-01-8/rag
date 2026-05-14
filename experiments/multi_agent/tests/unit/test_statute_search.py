import uuid
import pytest
from pathlib import Path

from multi_agent.schemas.document import Document, Chunk
from multi_agent.schemas.evidence import Evidence
from multi_agent.tools.retrievers.qdrant_client import drop_collection
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder
from multi_agent.tools.retrievers.index_builder import build_index
from multi_agent.tools.retrievers.statute_search import StatuteSearchTool, StatuteSearchArgs
from multi_agent.tracing.recorder import Recorder


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    """Build a fresh 4-chunk collection for the entire module."""
    name = f"test_statute_{uuid.uuid4().hex[:8]}"
    tmp_dir = tmp_path_factory.mktemp("idx")
    sparse_path = tmp_dir / "sparse.json"
    docs = [
        Document(
            law_name="中华人民共和国民法典", law_short="民法典",
            source_path="t",
            chunks=[
                Chunk(doc_id="民法典-510", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="510",
                      text="当事人就合同补充内容没有约定的，按照合同相关条款或者交易习惯确定。"),
                Chunk(doc_id="民法典-563", law_name="中华人民共和国民法典",
                      law_short="民法典", article_no="563",
                      text="当事人一方违约时，对方可以解除合同。"),
            ],
        ),
        Document(
            law_name="中华人民共和国刑法", law_short="刑法",
            source_path="t",
            chunks=[
                Chunk(doc_id="刑法-13", law_name="中华人民共和国刑法",
                      law_short="刑法", article_no="13",
                      text="一切危害国家主权的行为依照法律应当受刑罚处罚的，都是犯罪。"),
                Chunk(doc_id="刑法-14", law_name="中华人民共和国刑法",
                      law_short="刑法", article_no="14",
                      text="明知自己的行为会发生危害社会的结果，并且希望或者放任这种结果发生，因而构成犯罪的，是故意犯罪。"),
            ],
        ),
    ]
    build_index(
        documents=docs,
        collection_name=name,
        sparse_artifact_path=sparse_path,
        dense_encoder=DenseEncoder(),
    )
    yield {"collection": name, "sparse_path": sparse_path}
    drop_collection(name)


@pytest.mark.asyncio
async def test_search_returns_evidence_list(small_index, tmp_run_dir):
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(StatuteSearchArgs(query="合同补充约定", k=3), rec)
    rec.close()
    assert result.error is None
    hits = result.payload["evidences"]
    assert isinstance(hits, list)
    assert len(hits) >= 1
    # Top hit must be the contract article, not the criminal one
    top = Evidence.model_validate(hits[0])
    assert top.law_short == "民法典"
    assert top.retriever == "hybrid"


@pytest.mark.asyncio
async def test_search_respects_k(small_index, tmp_run_dir):
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(StatuteSearchArgs(query="违约", k=1), rec)
    rec.close()
    assert len(result.payload["evidences"]) == 1


@pytest.mark.asyncio
async def test_search_filter_by_law_short(small_index, tmp_run_dir):
    """Filter narrows results to just one law."""
    tool = StatuteSearchTool(
        collection_name=small_index["collection"],
        sparse_artifact_path=small_index["sparse_path"],
    )
    rec = Recorder(run_id="r1", run_dir=tmp_run_dir)
    result = await tool.call(
        StatuteSearchArgs(query="犯罪", k=5, law_short="刑法"),
        rec,
    )
    rec.close()
    hits = result.payload["evidences"]
    assert len(hits) >= 1
    assert all(Evidence.model_validate(h).law_short == "刑法" for h in hits)
