#!/usr/bin/env python
"""Render the LatencyProfiler flame graph + aggregates for a single run.

Usage:
    python scripts/profile_run.py <run_dir>
    python scripts/profile_run.py runs/01KRJZ...
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from multi_agent.eval.latency import LatencyProfiler


def main() -> int:
    parser = argparse.ArgumentParser(description="Render latency flame for a run directory.")
    parser.add_argument("run_dir", type=Path,
                       help="Path to a run directory containing events.jsonl")
    parser.add_argument("--json", action="store_true",
                       help="Output the full LatencyProfile as JSON instead of the flame")
    args = parser.parse_args()

    if not (args.run_dir / "events.jsonl").exists():
        print(f"error: {args.run_dir}/events.jsonl not found", file=sys.stderr)
        return 1

    profile = LatencyProfiler().profile(args.run_dir)

    if args.json:
        print(profile.model_dump_json(indent=2))
    else:
        print(f"# Latency profile — run_id={profile.run_id} total={profile.total_ms}ms")
        print()
        print(LatencyProfiler.render_flame(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
