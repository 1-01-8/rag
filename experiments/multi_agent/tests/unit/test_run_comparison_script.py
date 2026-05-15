"""Smoke tests for scripts/run_comparison.py — validates CLI shape only."""
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_comparison.py"


def test_run_comparison_help_text():
    """--help exits 0 and lists the expected flags."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--profile-a" in out.stdout
    assert "--profile-b" in out.stdout
    assert "--queryset" in out.stdout
    assert "Comparator" in out.stdout or "comparison" in out.stdout.lower()


def test_run_comparison_missing_required_exits_nonzero():
    """Invoking the script without required args exits non-zero."""
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert out.returncode != 0
