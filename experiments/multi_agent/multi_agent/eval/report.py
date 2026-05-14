"""Render summary.md from a RunGroup's results.jsonl (Phase 5b §7.5)."""
from __future__ import annotations
import json
import statistics
from pathlib import Path


def render_summary_md(group_dir: Path) -> Path:
    group_dir = Path(group_dir)
    results = [
        json.loads(l)
        for l in (group_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    n = len(results)
    ok = [r for r in results if r.get("status") == "ok"]
    errs = [r for r in results if r.get("status") != "ok"]
    latencies = [r["metrics"]["total_latency_ms"] for r in ok if "metrics" in r]
    in_tok = sum(r["metrics"].get("total_input_tokens", 0) for r in ok)
    out_tok = sum(r["metrics"].get("total_output_tokens", 0) for r in ok)
    cache_in = sum(r["metrics"].get("cache_read_tokens", 0) for r in ok)
    citation_hits = sum(1 for r in ok if r.get("citation_judge", {}).get("hit"))
    citation_scored = sum(
        1 for r in ok if not r.get("citation_judge", {}).get("skipped", True)
    )
    p50 = int(statistics.median(latencies)) if latencies else 0
    p95 = (
        int(statistics.quantiles(latencies, n=20)[18])
        if len(latencies) >= 5
        else (max(latencies) if latencies else 0)
    )

    lines = [
        f"# RunGroup `{group_dir.name}` 汇总\n",
        f"- 总计 Total: **{n}** queries (ok={len(ok)}, error={len(errs)})",
        f"- 延迟 latency: p50={p50}ms, p95={p95}ms",
        (
            f"- Tokens: input={in_tok}, output={out_tok}, cache_read={cache_in},"
            f" hit_rate={cache_in / in_tok if in_tok else 0:.2f}"
        ),
        (
            f"- Citation accuracy: **{citation_hits}/{citation_scored}**"
            f" ({100 * citation_hits / citation_scored if citation_scored else 0:.0f}%)"
        ),
        "",
        "## 逐 Query",
        "",
        "| Query | Status | Latency | Tokens (in/out) | Citation | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['query_id']} | error | — | — | — |"
                f" {r.get('error', '')[:40]} |"
            )
            continue
        m = r["metrics"]
        cj = r.get("citation_judge", {})
        cit = "skip" if cj.get("skipped") else ("✓" if cj.get("hit") else "✗")
        lines.append(
            f"| {r['query_id']} | ok | {m['total_latency_ms']}ms |"
            f" {m['total_input_tokens']}/{m['total_output_tokens']} | {cit} |"
            f" {m.get('final_answer_mode') or ''} |"
        )
    out = group_dir / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
