"""User-visible HNSW progress labels must match the Rust writer's real work.

`duckdb_hnsw_enabled=false` must never render `building...` or
`included in compaction`, and an enabled run that changed nothing must not
claim to be building. These tests drive the real Rich progress bridge with
the phase names Rust emits.
"""

from __future__ import annotations

import io

from rich.console import Console
from rich.progress import Progress

from chunkhound.services.rust_pipeline_runner import RustProgressBridge


def _bridge(hnsw_enabled: bool) -> tuple[Progress, RustProgressBridge, int, int]:
    progress = Progress(
        console=Console(file=io.StringIO(), force_terminal=False, width=120)
    )
    parse_task = progress.add_task("parse")
    data_task = progress.add_task("data")
    index_task = progress.add_task(
        "Vector index (HNSW)",
        total=1,
        start=False,
        visible=hnsw_enabled,
    )
    compact_task = progress.add_task("compacting", total=1, start=False)
    bridge = RustProgressBridge(
        progress=progress,
        parse_task=parse_task,
        data_task=data_task,
        index_task=index_task,
        compact_task=compact_task,
        diff_task=None,
        embed_task=None,
        compact_db_file="/nonexistent/chunks.db",
        hnsw_enabled=hnsw_enabled,
    )
    return progress, bridge, index_task, compact_task


def _info(progress: Progress, task: int) -> str:
    return progress.tasks[task].fields["info"]


def _visible(progress: Progress, task: int) -> bool:
    return progress.tasks[task].visible


def _finished(progress: Progress, task: int) -> bool:
    return progress.tasks[task].finished


def _elapsed(progress: Progress, task: int) -> float | None:
    return progress.tasks[task].elapsed


def test_disabled_row_is_hidden_from_start() -> None:
    progress, _bridge_, index_task, _compact_task = _bridge(hnsw_enabled=False)

    assert not _visible(progress, index_task)


def test_disabled_compaction_does_not_claim_index_rebuild() -> None:
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=False)

    bridge("write-compact-disabled", 0, 1)

    assert not _visible(progress, index_task)
    assert _info(progress, index_task) == "not built (HNSW disabled)"
    assert _finished(progress, index_task)
    assert "rebuild" not in _info(progress, compact_task)
    assert bridge.compact_ran is True
    assert not _finished(progress, compact_task)

    bridge("write-done", 1, 1)
    assert _info(progress, index_task) == "not built (HNSW disabled)"
    assert _finished(progress, compact_task)
    assert _elapsed(progress, compact_task) is not None


def test_disabled_non_compaction_stays_disabled() -> None:
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=False)

    bridge("write-index-disabled", 1, 1)

    assert not _visible(progress, index_task)
    assert _info(progress, compact_task) == "not needed"
    assert _finished(progress, index_task)
    assert _finished(progress, compact_task)
    assert _elapsed(progress, compact_task) is not None

    bridge("write-done", 1, 1)
    assert _info(progress, index_task) == "disabled (not built)"


def test_vss_unavailable_is_reported_as_skipped() -> None:
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=True)

    bridge("write-index-unavailable", 1, 1)

    assert not _visible(progress, index_task)
    assert _info(progress, index_task) == "skipped (VSS unavailable)"
    assert _finished(progress, index_task)
    assert _finished(progress, compact_task)


def test_enabled_build_reaches_done() -> None:
    progress, bridge, index_task, _compact_task = _bridge(hnsw_enabled=True)

    bridge("write-index", 0, 1)
    assert _visible(progress, index_task)
    assert _info(progress, index_task) == "building..."
    assert not _finished(progress, index_task)

    bridge("write-done", 1, 1)
    assert _info(progress, index_task) == "done"
    assert _finished(progress, index_task)
    assert _elapsed(progress, index_task) is not None


def test_enabled_empty_run_says_up_to_date() -> None:
    progress, bridge, index_task, _compact_task = _bridge(hnsw_enabled=True)

    bridge("write-index", 1, 1)
    assert _visible(progress, index_task)
    assert _info(progress, index_task) == "up to date (no changes)"
    assert _finished(progress, index_task)
    assert _elapsed(progress, index_task) is not None

    bridge("write-done", 1, 1)
    assert _info(progress, index_task) == "up to date (no changes)"
    assert _finished(progress, index_task)


def test_enabled_compaction_reports_index_maintenance() -> None:
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=True)

    bridge("write-compact", 0, 1)

    assert _visible(progress, index_task)
    assert _info(progress, index_task) == "maintained during compaction"
    assert _finished(progress, index_task)
    assert _elapsed(progress, index_task) is not None
    assert _info(progress, compact_task) == "compacting (with index)..."
    assert not _finished(progress, compact_task)
    assert bridge.compact_ran is True

    bridge("write-done", 1, 1)
    assert _finished(progress, compact_task)
    assert _elapsed(progress, compact_task) is not None


def test_vss_unavailable_compaction_does_not_claim_rebuild() -> None:
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=True)

    bridge("write-compact-unavailable", 0, 1)

    assert not _visible(progress, index_task)
    assert _info(progress, index_task) == "not built (VSS unavailable)"
    assert _finished(progress, index_task)
    assert "rebuild" not in _info(progress, compact_task)
    assert bridge.compact_ran is True
    assert not _finished(progress, compact_task)

    bridge("write-done", 1, 1)
    assert _finished(progress, compact_task)


def test_legacy_write_phase_finishes_bars() -> None:
    """Old extensions emit one `write` phase; both rows must still finish."""
    progress, bridge, index_task, compact_task = _bridge(hnsw_enabled=True)

    bridge("write", 1, 1)

    assert _finished(progress, index_task)
    assert _finished(progress, compact_task)
    assert _elapsed(progress, index_task) is not None
    assert _elapsed(progress, compact_task) is not None
