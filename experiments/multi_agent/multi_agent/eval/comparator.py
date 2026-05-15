"""Comparator (Phase 5c §7.8).

Loads two RunGroup ``results.jsonl`` files, computes per-query deltas
(latency, token counts, citation hits, judge-score diffs), and renders
a diff Markdown table.

Pure-Python, no LLM calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class PerQueryDelta(BaseModel):
    """Deltas for a single query between group B and group A (B − A)."""

    query_id: str
    latency_delta_ms: int = 0
    in_tokens_delta: int = 0
    out_tokens_delta: int = 0
    cost_delta_usd: float | None = None
    citation_hit_a: bool = False
    citation_hit_b: bool = False
    groundedness_delta: float | None = None
    helpfulness_delta: float | None = None
    winner: Literal["A", "B", "tie"] = "tie"


class ComparisonReport(BaseModel):
    """Summary of a group-vs-group comparison."""

    group_a: str
    group_b: str
    n_a: int
    n_b: int
    n_common: int
    per_query: list[PerQueryDelta] = Field(default_factory=list)
    winners_a: int = 0
    winners_b: int = 0
    ties: int = 0


def _pick_winner(
    groundedness_delta: float | None,
    helpfulness_delta: float | None,
    citation_hit_a: bool,
    citation_hit_b: bool,
) -> Literal["A", "B", "tie"]:
    """Heuristic per-query winner.

    Priority order: groundedness > citation_hit > helpfulness. A meaningful
    difference in groundedness (|Δ| >= 0.05) wins; otherwise an exclusive
    citation hit wins; otherwise a meaningful helpfulness gap wins; else tie.
    """
    if groundedness_delta is not None and abs(groundedness_delta) >= 0.05:
        return "B" if groundedness_delta > 0 else "A"
    if citation_hit_a != citation_hit_b:
        return "A" if citation_hit_a else "B"
    if helpfulness_delta is not None and abs(helpfulness_delta) >= 0.05:
        return "B" if helpfulness_delta > 0 else "A"
    return "tie"


class Comparator:
    """Compare two RunGroup result directories.

    Usage::

        report = Comparator().compare(group_a_dir=Path("runs/ga"),
                                      group_b_dir=Path("runs/gb"))
        out_path = Comparator().render_md(report=report, out_dir=Path("reports"))
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(group_dir: Path) -> dict[str, dict]:
        """Read results.jsonl and return a {query_id → row} mapping."""
        text = (Path(group_dir) / "results.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {r["query_id"]: r for r in rows}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        *,
        group_a_dir: Path,
        group_b_dir: Path,
    ) -> ComparisonReport:
        """Compute per-query deltas between two groups.

        All numeric deltas are computed as ``B − A``, so a positive
        ``latency_delta_ms`` means group B was *slower*.
        """
        a = self._load(Path(group_a_dir))
        b = self._load(Path(group_b_dir))
        common = sorted(set(a) & set(b))

        per_q: list[PerQueryDelta] = []
        for qid in common:
            ra, rb = a[qid], b[qid]
            ma: dict = ra.get("metrics") or {}
            mb: dict = rb.get("metrics") or {}

            # Judge scores (optional fields)
            ja_g = ((ra.get("judges") or {}).get("groundedness") or {})
            jb_g = ((rb.get("judges") or {}).get("groundedness") or {})
            ja_h = ((ra.get("judges") or {}).get("helpfulness") or {})
            jb_h = ((rb.get("judges") or {}).get("helpfulness") or {})

            groundedness_delta: float | None = None
            if ja_g.get("score") is not None and jb_g.get("score") is not None:
                groundedness_delta = float(jb_g["score"]) - float(ja_g["score"])

            helpfulness_delta: float | None = None
            if ja_h.get("score") is not None and jb_h.get("score") is not None:
                helpfulness_delta = float(jb_h["score"]) - float(ja_h["score"])

            cost_a = ma.get("cost_usd")
            cost_b = mb.get("cost_usd")
            cost_delta: float | None = None
            if cost_a is not None and cost_b is not None:
                cost_delta = round(float(cost_b) - float(cost_a), 6)

            cit_a = bool((ra.get("citation_judge") or {}).get("hit"))
            cit_b = bool((rb.get("citation_judge") or {}).get("hit"))
            winner = _pick_winner(groundedness_delta, helpfulness_delta, cit_a, cit_b)

            per_q.append(
                PerQueryDelta(
                    query_id=qid,
                    latency_delta_ms=(
                        mb.get("total_latency_ms", 0) - ma.get("total_latency_ms", 0)
                    ),
                    in_tokens_delta=(
                        mb.get("total_input_tokens", 0) - ma.get("total_input_tokens", 0)
                    ),
                    out_tokens_delta=(
                        mb.get("total_output_tokens", 0) - ma.get("total_output_tokens", 0)
                    ),
                    cost_delta_usd=cost_delta,
                    citation_hit_a=cit_a,
                    citation_hit_b=cit_b,
                    groundedness_delta=groundedness_delta,
                    helpfulness_delta=helpfulness_delta,
                    winner=winner,
                )
            )

        winners_a = sum(1 for p in per_q if p.winner == "A")
        winners_b = sum(1 for p in per_q if p.winner == "B")
        ties = sum(1 for p in per_q if p.winner == "tie")
        return ComparisonReport(
            group_a=Path(group_a_dir).name,
            group_b=Path(group_b_dir).name,
            n_a=len(a),
            n_b=len(b),
            n_common=len(common),
            per_query=per_q,
            winners_a=winners_a,
            winners_b=winners_b,
            ties=ties,
        )

    def render_md(self, report: ComparisonReport, out_dir: Path) -> Path:
        """Render a diff Markdown table to *out_dir* and return the Path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            f"# Comparison `{report.group_a}` vs `{report.group_b}`",
            "",
            f"- **Group A:** `{report.group_a}` ({report.n_a} queries)",
            f"- **Group B:** `{report.group_b}` ({report.n_b} queries)",
            f"- **Common queries:** {report.n_common}",
            "",
            "| Query | Δlat ms | Δin tok | Δout tok | Δcost $ | Cite A | Cite B"
            " | Δgrounded | Δhelpful | Winner |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]

        for p in report.per_query:
            g_str = f"{p.groundedness_delta:+.2f}" if p.groundedness_delta is not None else "—"
            h_str = f"{p.helpfulness_delta:+.2f}" if p.helpfulness_delta is not None else "—"
            c_str = f"{p.cost_delta_usd:+.4f}" if p.cost_delta_usd is not None else "—"
            cite_a = "✓" if p.citation_hit_a else "✗"
            cite_b = "✓" if p.citation_hit_b else "✗"
            lines.append(
                f"| {p.query_id} | {p.latency_delta_ms:+d} | {p.in_tokens_delta:+d} |"
                f" {p.out_tokens_delta:+d} | {c_str} | {cite_a} | {cite_b} | {g_str} | {h_str}"
                f" | {p.winner} |"
            )

        # Summary stats (averages over non-None deltas)
        if report.per_query:
            lat_vals = [p.latency_delta_ms for p in report.per_query]
            g_vals = [p.groundedness_delta for p in report.per_query if p.groundedness_delta is not None]
            h_vals = [p.helpfulness_delta for p in report.per_query if p.helpfulness_delta is not None]

            avg_lat = sum(lat_vals) / len(lat_vals)
            lines += [
                "",
                "## Summary",
                f"- Avg Δlatency: {avg_lat:+.0f} ms",
            ]
            if g_vals:
                lines.append(f"- Avg Δgroundedness: {sum(g_vals)/len(g_vals):+.3f}")
            if h_vals:
                lines.append(f"- Avg Δhelpfulness: {sum(h_vals)/len(h_vals):+.3f}")

        out_path = out_dir / f"{report.group_a}_vs_{report.group_b}.md"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out_path
