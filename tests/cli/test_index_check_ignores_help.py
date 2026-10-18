from __future__ import annotations

from tests.helpers.cli import run_cli


def test_index_help_mentions_check_ignores() -> None:
    proc = run_cli(["index", "--help"], timeout=20)
    assert proc.returncode == 0, proc.stderr
    # Ensure the help lists the new flag
    assert "check-ignores" in proc.stdout
