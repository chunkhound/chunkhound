"""Import invariants for parser mappings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_import_writes_nothing_to_stdout() -> None:
    """Parser imports must not corrupt stdout-based protocols such as MCP."""
    completed = subprocess.run(
        [sys.executable, "-c", "import chunkhound.parsers.mappings"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
