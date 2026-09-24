"""CLI completion summary must break skip counts into user-visible categories."""

import io

from rich.console import Console

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter


def test_completion_summary_includes_timeout_line() -> None:
    buf = io.StringIO()
    formatter = RichOutputFormatter()
    formatter.console = Console(
        file=buf, force_terminal=True, color_system=None, width=80
    )
    formatter.completion_summary(
        {
            "files_processed": 1,
            "files_skipped": 19,
            "skipped_unchanged": 17,
            "skipped_filtered": 1,
            "skipped_due_to_timeout": ["src/db/duckdb_backend.rs"],
            "files_errors": 0,
            "chunks_created": 0,
        },
        1.23,
    )
    out = buf.getvalue()
    assert "Timeout:" in out
    assert "Unchanged:" in out
    assert "Filtered:" in out
