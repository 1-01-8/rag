"""Smoke tests for scripts/run_eval.py CLI.

These tests only invoke --help and a missing-arg error path, so they
do not require a running vLLM instance or Qdrant.
"""
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "run_eval.py"


def test_run_eval_help_text():
    """Verify the CLI parses and --help emits expected argument names."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--queryset" in out.stdout
    assert "--statutes-collection" in out.stdout
    assert "--group-name" in out.stdout


def test_run_eval_missing_required_arg_exits_nonzero():
    """Invoking the script with no args should exit non-zero and mention an error."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
    assert "required" in out.stderr.lower() or "error" in out.stderr.lower()
