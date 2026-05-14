"""Tests for Comparator + render_md (Phase 5c §7.8)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_row(
    query_id: str,
    latency_ms: int,
    groundedness_score: float | None = None,
    helpfulness_score: float | None = None,
    citation_hit: bool = True,
) -> dict:
    row: dict = {
        "query_id": query_id,
        "status": "ok",
        "metrics": {
            "total_latency_ms": latency_ms,
            "total_input_tokens": 800,
            "total_output_tokens": 200,
            "cache_read_tokens": 0,
            "cache_hit_rate": 0,
            "agent_invocations": 1,
            "tool_calls_total": 2,
            "react_steps_total": 0,
            "errors": 0,
            "final_answer_mode": "evidence_grounded",
            "citation_count": 1,
            "supervisor_verdict": None,
        },
        "citation_judge": {
            "hit": citation_hit,
            "matched": ["民法典-510"],
            "expected": ["民法典-510"],
            "actual": ["民法典-510"],
            "skipped": False,
            "reason": "",
        },
    }
    judges: dict = {}
    if groundedness_score is not None:
        judges["groundedness"] = {
            "judge": "groundedness",
            "score": groundedness_score,
            "parsed": None,
            "raw": "",
            "error": None,
        }
    if helpfulness_score is not None:
        judges["helpfulness"] = {
            "judge": "helpfulness",
            "score": helpfulness_score,
            "parsed": None,
            "raw": "",
            "error": None,
        }
    if judges:
        row["judges"] = judges
    return row


def test_comparator_diffs_two_groups(tmp_path: Path) -> None:
    ga = tmp_path / "ga"
    gb = tmp_path / "gb"
    ga.mkdir()
    gb.mkdir()

    row_a = _make_row("q1", latency_ms=1000, groundedness_score=0.9,
                      helpfulness_score=0.8, citation_hit=True)
    row_b = {
        **_make_row("q1", latency_ms=2000, groundedness_score=0.7,
                    helpfulness_score=0.6, citation_hit=False),
    }

    (ga / "results.jsonl").write_text(json.dumps(row_a, ensure_ascii=False) + "\n", encoding="utf-8")
    (gb / "results.jsonl").write_text(json.dumps(row_b, ensure_ascii=False) + "\n", encoding="utf-8")

    from multi_agent.eval.comparator import Comparator

    report = Comparator().compare(group_a_dir=ga, group_b_dir=gb)
    assert report.n_common == 1
    assert report.n_a == 1
    assert report.n_b == 1
    assert report.per_query[0].query_id == "q1"
    # b is 1000 ms slower
    assert report.per_query[0].latency_delta_ms == 1000
    # b groundedness is 0.2 lower
    assert report.per_query[0].groundedness_delta == pytest.approx(-0.2)
    # b helpfulness is 0.2 lower
    assert report.per_query[0].helpfulness_delta == pytest.approx(-0.2)
    # citation hit flags
    assert report.per_query[0].citation_hit_a is True
    assert report.per_query[0].citation_hit_b is False


def test_render_md_produces_nonempty_file(tmp_path: Path) -> None:
    ga = tmp_path / "group_alpha"
    gb = tmp_path / "group_beta"
    ga.mkdir()
    gb.mkdir()

    row_a = _make_row("q1", latency_ms=500, groundedness_score=0.85)
    row_b = _make_row("q1", latency_ms=750, groundedness_score=0.75)
    (ga / "results.jsonl").write_text(json.dumps(row_a, ensure_ascii=False) + "\n", encoding="utf-8")
    (gb / "results.jsonl").write_text(json.dumps(row_b, ensure_ascii=False) + "\n", encoding="utf-8")

    from multi_agent.eval.comparator import Comparator

    cmp = Comparator()
    report = cmp.compare(group_a_dir=ga, group_b_dir=gb)
    out_path = cmp.render_md(report=report, out_dir=tmp_path)

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert len(content) > 50, "Expected non-trivial markdown output"
    # Table header present
    assert "Query" in content
    assert "group_alpha" in content
    assert "group_beta" in content
    # Row for q1 present
    assert "q1" in content


def test_comparator_no_common_queries(tmp_path: Path) -> None:
    """Edge case: no overlapping query_ids → n_common=0, empty per_query."""
    ga = tmp_path / "ga"
    gb = tmp_path / "gb"
    ga.mkdir()
    gb.mkdir()

    (ga / "results.jsonl").write_text(
        json.dumps(_make_row("only_in_a", latency_ms=100), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (gb / "results.jsonl").write_text(
        json.dumps(_make_row("only_in_b", latency_ms=200), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    from multi_agent.eval.comparator import Comparator

    report = Comparator().compare(group_a_dir=ga, group_b_dir=gb)
    assert report.n_common == 0
    assert report.per_query == []
    assert report.n_a == 1
    assert report.n_b == 1
