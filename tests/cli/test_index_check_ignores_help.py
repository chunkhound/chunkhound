from __future__ import annotations

from tests.helpers.cli import run_cli_in_process


def test_index_help_mentions_check_ignores() -> None:
    proc = run_cli_in_process(["index", "--help"])
    assert proc.returncode == 0, proc.stderr
    # Ensure the help lists the new flag
    assert "check-ignores" in proc.stdout
