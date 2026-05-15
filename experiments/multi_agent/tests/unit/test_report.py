import json
from pathlib import Path
import pytest
from multi_agent.eval.report import render_summary_md


def _make_ok_row(query_id: str, latency_ms: int) -> dict:
    return {
        "query_id": query_id, "run_id": query_id, "status": "ok",
        "metrics": {
            "total_latency_ms": latency_ms, "total_input_tokens": 100,
            "total_output_tokens": 50, "agent_invocations": 1,
            "tool_calls_total": 0, "cache_hit_rate": 0.0,
            "errors": 0, "final_answer_mode": "evidence_grounded",
            "cache_read_tokens": 0, "react_steps_total": 0,
            "supervisor_verdict": None, "citation_count": 0,
        },
        "citation_judge": {"hit": False, "matched": [], "expected": [],
                           "actual": [], "skipped": True, "reason": ""},
    }


def test_render_summary_md(tmp_path):
    group_dir = tmp_path / "g"
    group_dir.mkdir()
    rows = [
        {"query_id": "q1", "run_id": "r1", "status": "ok",
         "metrics": {"total_latency_ms": 1000, "total_input_tokens": 800,
                     "total_output_tokens": 200, "agent_invocations": 1,
                     "tool_calls_total": 2, "cache_hit_rate": 0.5,
                     "errors": 0, "final_answer_mode": "evidence_grounded",
                     "cache_read_tokens": 400, "react_steps_total": 0,
                     "supervisor_verdict": None, "citation_count": 0},
         "citation_judge": {"hit": True, "matched": ["民法典-510"], "expected": ["民法典-510"],
                            "actual": ["民法典-510"], "skipped": False, "reason": ""}},
        {"query_id": "q2", "run_id": "r2", "status": "ok",
         "metrics": {"total_latency_ms": 2000, "total_input_tokens": 1000,
                     "total_output_tokens": 300, "agent_invocations": 1,
                     "tool_calls_total": 1, "cache_hit_rate": 0.4,
                     "errors": 0, "final_answer_mode": "evidence_grounded",
                     "cache_read_tokens": 400, "react_steps_total": 0,
                     "supervisor_verdict": None, "citation_count": 0},
         "citation_judge": {"hit": False, "matched": [], "expected": ["民法典-999"],
                            "actual": [], "skipped": False, "reason": "no expected"}},
    ]
    results = group_dir / "results.jsonl"
    results.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summary_path = render_summary_md(group_dir)
    assert summary_path.exists()
    md = summary_path.read_text(encoding="utf-8")
    assert "总计" in md or "Total" in md
    assert "1/2" in md or "50" in md  # citation hit rate
    assert "q1" in md and "q2" in md


def test_p95_uses_max_when_fewer_than_20_samples(tmp_path):
    """Fix 5: with <20 samples p95 should equal max(latencies), not quantiles()."""
    group_dir = tmp_path / "g_small"
    group_dir.mkdir()
    # 5 samples: max is 5000 ms
    rows = [_make_ok_row(f"q{i}", (i + 1) * 1000) for i in range(5)]
    (group_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    render_summary_md(group_dir)
    md = (group_dir / "summary.md").read_text()
    # p95 should be max(latencies) = 5000 ms
    assert "p95=5000ms" in md


def test_p95_uses_quantiles_when_20_or_more_samples(tmp_path):
    """Fix 5: with >=20 samples quantiles() is used (p95 <= max)."""
    group_dir = tmp_path / "g_large"
    group_dir.mkdir()
    # 20 uniform samples from 1000..20000
    rows = [_make_ok_row(f"q{i}", (i + 1) * 1000) for i in range(20)]
    (group_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )
    render_summary_md(group_dir)
    md = (group_dir / "summary.md").read_text()
    # p95 must be present and less than max (20000)
    import re
    m = re.search(r"p95=(\d+)ms", md)
    assert m is not None
    p95_val = int(m.group(1))
    assert p95_val < 20000, f"p95={p95_val} should be < max=20000 for uniform distribution"
