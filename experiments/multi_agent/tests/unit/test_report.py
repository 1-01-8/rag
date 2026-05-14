import json
from pathlib import Path
import pytest
from multi_agent.eval.report import render_summary_md


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
