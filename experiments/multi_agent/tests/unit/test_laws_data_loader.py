import pytest
import zipfile
import json
from pathlib import Path
from multi_agent.tools.laws_data_loader import iter_laws_data, filter_unsupported_causes
from multi_agent.schemas.case import CaseQA


@pytest.fixture
def fake_zip(tmp_path):
    """Create a tiny zip mimicking the laws_data train/ structure."""
    zpath = tmp_path / "fake.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("train/000001.json", json.dumps({
            "question": "房东涨租", "answer": "协商不成可起诉",
            "candidate_answer": ["协商", "起诉"], "cause": "房产纠纷",
        }))
        z.writestr("train/000002.json", json.dumps({
            "question": "被开除", "answer": "申请劳动仲裁",
            "candidate_answer": [], "cause": "劳动纠纷",
        }))
        z.writestr("train/000003.json", json.dumps({
            "question": "撞人了", "answer": "保险公司先赔",
            "candidate_answer": [], "cause": "交通事故",
        }))
    return zpath


def test_iter_yields_all_records(fake_zip):
    records = list(iter_laws_data(fake_zip))
    assert len(records) == 3
    for r in records:
        assert isinstance(r, CaseQA)
        assert r.extracted_cite_ids == []   # not extracted yet


def test_iter_records_have_correct_case_ids(fake_zip):
    records = list(iter_laws_data(fake_zip))
    assert {r.case_id for r in records} == {
        "train_000001", "train_000002", "train_000003",
    }


def test_filter_drops_unsupported_causes(fake_zip):
    """Drop 劳动纠纷 per ADR-15."""
    records = filter_unsupported_causes(iter_laws_data(fake_zip))
    causes = {r.cause for r in records}
    assert "劳动纠纷" not in causes
    assert "房产纠纷" in causes
    assert "交通事故" in causes
