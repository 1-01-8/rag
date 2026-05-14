import pytest
from multi_agent.tools.retrievers.sparse_encoder import SparseEncoder, SparseVector


CORPUS = [
    "民法典第一条 立法目的",
    "民法典第二条 调整民事关系",
    "民法典第五百一十条 合同补充内容确定",
    "刑法第十三条 犯罪定义",
    "刑法第十四条 故意犯罪",
]


def test_fit_and_encode_returns_sparse_vector():
    enc = SparseEncoder()
    enc.fit(CORPUS)
    vec = enc.encode("民法典 合同")
    assert isinstance(vec, SparseVector)
    assert len(vec.indices) == len(vec.values)
    assert len(vec.indices) > 0


def test_idf_weights_rare_terms_higher():
    """A token appearing in 1 doc should have higher IDF than one in many."""
    enc = SparseEncoder()
    enc.fit(CORPUS)
    rare_vec = enc.encode("合同补充")     # appears in 1 doc
    common_vec = enc.encode("民法典")     # appears in 3 docs
    rare_max_val = max(rare_vec.values) if rare_vec.values else 0
    common_max_val = max(common_vec.values) if common_vec.values else 0
    assert rare_max_val > common_max_val


def test_oov_token_returns_empty_vector():
    """Query with only unseen tokens → empty vector (Qdrant treats as no match)."""
    enc = SparseEncoder()
    enc.fit(CORPUS)
    vec = enc.encode("xyz-unknown-12345")
    assert vec.indices == []
    assert vec.values == []


def test_save_and_load_roundtrip(tmp_path):
    enc = SparseEncoder()
    enc.fit(CORPUS)
    out = tmp_path / "sparse.json"
    enc.save(out)

    enc2 = SparseEncoder.load(out)
    v1 = enc.encode("民法典 合同")
    v2 = enc2.encode("民法典 合同")
    assert v1.indices == v2.indices
    assert v1.values == v2.values
