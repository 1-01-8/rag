import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def tmp_run_dir(tmp_path: Path) -> Path:
    """Fresh run directory for each test."""
    d = tmp_path / "runs" / "test-run-0001"
    d.mkdir(parents=True, exist_ok=True)
    return d
