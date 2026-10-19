"""Capture the CLI's production Rich columns for native progress callbacks."""

import io

from rich.console import Console

from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.services.rust_pipeline_runner import RustProgressBridge


class ProgressOutput:
    def __init__(self, monkeypatch, *, enabled: bool = True) -> None:
        monkeypatch.delenv("CHUNKHOUND_NO_RICH", raising=False)
        monkeypatch.setenv("TERM", "xterm")
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        self.stream = io.StringIO()
        formatter = RichOutputFormatter()
        formatter.console = Console(file=self.stream, width=200, force_terminal=False)
        self.progress = formatter.create_progress_display().progress
        self.frames: list[tuple[str, str]] = []
        self._create_bridge(enabled)

    def _create_bridge(self, enabled: bool) -> None:
        add = self.progress.add_task
        self.bridge = RustProgressBridge(
            progress=self.progress,
            parse_task=add("Parsing", total=1, start=False),
            data_task=add("Writing data", total=1, start=False),
            index_task=add(
                "Vector index (HNSW)", total=1, start=False, visible=enabled
            ),
            compact_task=add("Compacting", total=1, start=False),
            diff_task=None,
            embed_task=None,
            compact_db_file="/nonexistent/progress-contract/chunks.db",
            hnsw_enabled=enabled,
        )

    def __call__(self, phase: str, current: int, total: int, chunks: int = 0) -> None:
        self.bridge(phase, current, total, chunks)
        self.frames.append((phase, self.render()))

    def render(self) -> str:
        self.stream.seek(0)
        self.stream.truncate()
        self.progress.console.print(self.progress)
        return self.stream.getvalue()

    def during(self, phase: str) -> str:
        return next(output for name, output in self.frames if name == phase)


def assert_completed_rows(output: str) -> None:
    """Completed write rows must show a stopped spinner, not an active glyph."""
    for label in ("Writing data", "Vector index (HNSW)", "Compacting"):
        for row in output.splitlines():
            if label in row:
                assert row.lstrip().startswith(label), row
                assert "0:00:" in row, row
