from __future__ import annotations

from tests.helpers.cli import run_cli_in_process


def test_research_help_mentions_path_filter() -> None:
    proc = run_cli_in_process(["research", "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "--path-filter" in proc.stdout
