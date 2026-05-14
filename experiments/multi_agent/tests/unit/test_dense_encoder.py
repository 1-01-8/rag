import numpy as np
import pytest
from multi_agent.tools.retrievers.dense_encoder import DenseEncoder


@pytest.fixture(scope="module")
def encoder():
    return DenseEncoder()


def test_encode_single_returns_1d_vector(encoder):
    vec = encoder.encode_one("民法典第510条")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == encoder.dim  # bge-m3 is 1024


def test_encode_batch_returns_2d_matrix(encoder):
    texts = ["民法典第一条", "民法典第二条", "刑法第十三条"]
    mat = encoder.encode_batch(texts)
    assert mat.shape == (3, encoder.dim)


def test_encode_vectors_are_unit_normalized(encoder):
    vec = encoder.encode_one("测试文本")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-4   # bge-m3 outputs are normalized


def test_similar_text_higher_similarity(encoder):
    a = encoder.encode_one("房东要涨房租")
    b = encoder.encode_one("房屋租金变更")
    c = encoder.encode_one("天气真好")
    sim_ab = float(np.dot(a, b))
    sim_ac = float(np.dot(a, c))
    assert sim_ab > sim_ac, f"租赁相关应比天气相关更相似: {sim_ab} vs {sim_ac}"
