# tests/unit/test_queryset.py
from pathlib import Path
import pytest
from multi_agent.eval.queryset import QuerySet, Query


def test_queryset_loads_seed_yaml():
    path = Path(__file__).parents[2] / "evals" / "querysets" / "synthetic_seed_v1.yaml"
    qs = QuerySet.from_yaml(path)
    assert qs.meta.name == "synthetic_seed_v1"
    assert len(qs.queries) >= 5
    assert qs.queries[0].id == "q001"
    assert qs.queries[0].text.startswith("房东")
    assert "民法典-510" in qs.queries[0].expected.should_cite_any


def test_query_has_required_fields():
    q = Query(id="qX", text="t", jurisdiction="CN", cause="c", source="s")
    assert q.tags == []
    assert q.expected.should_cite_any == []
