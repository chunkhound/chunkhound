"""Disabled HNSW must report itself as disabled, never as building.

The Rust writer already skips index creation when
``duckdb_hnsw_enabled=false``. These contract tests pin the progress phase
that reaches the CLI, so a future change cannot resurrect the misleading
``included in compaction`` label without failing here.
"""

from __future__ import annotations

from pathlib import Path

from tests.contracts.pipeline_harness import hnsw_index_names, index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _run(db_dir: Path, **kwargs) -> list[str]:
    phases: list[str] = []
    result = index_with_rust(
        FIXTURE_DIR,
        db_dir,
        skip_embeddings=False,
        duckdb_hnsw_enabled=False,
        progress_callback=lambda phase, current, total: phases.append(phase),
        **kwargs,
    )
    assert not result.errors, f"Unexpected errors: {result.errors}"
    assert result.chunks_written > 0, "the fixture run should write chunks"
    return phases


def test_disabled_run_reports_disabled_and_creates_no_hnsw(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"

    phases = _run(
        db_dir,
        compaction_threshold=1.0,
        compaction_min_size_mb=10**9,
    )

    assert "write-index-disabled" in phases
    assert "write-index" not in phases
    assert "write-compact" not in phases
    assert hnsw_index_names(db_dir) == set()


def test_disabled_compaction_reports_no_index_rebuild(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"

    phases = _run(
        db_dir,
        compaction_threshold=0.0,
        compaction_min_size_mb=0,
    )

    assert "write-compact-disabled" in phases
    assert "write-compact" not in phases
    assert "write-index" not in phases
    assert hnsw_index_names(db_dir) == set()
