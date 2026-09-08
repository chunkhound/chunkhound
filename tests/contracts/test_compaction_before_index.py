"""Contract test: compaction runs before (and folds in) the HNSW index build.

`run_compaction()`'s EXPORT/IMPORT rewrite copies `files`/`chunks`/
`embeddings_*` into a fresh database with no indexes, then rebuilds them via
`reopen()` — on both its success and failure-fallback paths. So whenever
compaction triggers, the store thread must NOT also call
`ensure_all_hnsw_indexes()` separately, or the (CPU-intensive) HNSW index
gets built twice for one run. This test forces compaction to trigger on
every run (`compaction_threshold=0.0`, `compaction_min_size_mb=0`) and
verifies the mutual exclusion, plus that data and the index both survive
the compact-then-reindex sequence intact.
"""

import tempfile
from pathlib import Path

import duckdb
import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _index_with_rust_forced_compaction(
    fixture_dir: Path, db_dir: Path
) -> tuple[IndexResult, list[str]]:
    """Index *fixture_dir* with the Rust pipeline, forcing compaction to run.

    Returns ``(IndexResult, phases_seen)`` where ``phases_seen`` is every
    phase string the progress callback observed, in call order.

    ``compaction_threshold=0.0`` means any non-negative free/waste ratio
    (always true) clears the bar, and ``compaction_min_size_mb=0`` means
    even a tiny reclaimable amount clears the min-size gate — together
    these force ``needs_compaction()`` to always return true.
    """
    phases_seen: list[str] = []

    def progress_callback(phase: str, current: int, total: int) -> None:
        phases_seen.append(phase)

    result = index_with_rust(
        fixture_dir,
        db_dir,
        skip_embeddings=False,
        compaction_threshold=0.0,
        compaction_min_size_mb=0,
        progress_callback=progress_callback,
    )
    return result, phases_seen


def _hnsw_index_names(db_dir: Path) -> list[str]:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        rows = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name LIKE 'embeddings_%'"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows if "hnsw" in r[0].lower()]


class TestCompactionBeforeIndex:
    """Compaction must run before, and fold in, the HNSW index rebuild."""

    @pytest.mark.asyncio
    async def test_compaction_forces_reindex_not_double_build(self):
        """When compaction triggers, "write-compact" fires and "write-index"
        does not — compaction rebuilds the index internally via reopen(), so
        a separate index-build step would silently duplicate that work.
        """
        with tempfile.TemporaryDirectory() as tmp_rs:
            db_rs = Path(tmp_rs) / "db"

            result_rs, phases = _index_with_rust_forced_compaction(FIXTURE_DIR, db_rs)

            assert not result_rs.errors, f"Unexpected errors: {result_rs.errors}"
            assert "write-compact" in phases, (
                "Expected compaction to trigger with compaction_threshold=0.0"
            )
            assert "write-index" not in phases, (
                "write-index should not fire in the same run as write-compact — "
                "compaction already rebuilds the HNSW index internally"
            )

    @pytest.mark.asyncio
    async def test_compaction_preserves_data_and_index(self):
        """Data and the HNSW index both survive the compact-then-reindex path."""
        with tempfile.TemporaryDirectory() as tmp_py, tempfile.TemporaryDirectory() as tmp_rs:
            db_py = Path(tmp_py) / "db"
            db_py.mkdir(parents=True, exist_ok=True)
            db_rs = Path(tmp_rs) / "db"

            result_py = await index_with_python(
                FIXTURE_DIR, db_py, skip_embeddings=False,
                embedding_provider=MockEmbeddingProvider(),
            )

            result_rs, _phases = _index_with_rust_forced_compaction(FIXTURE_DIR, db_rs)

            assert_identical(result_py, result_rs)

            hnsw_names = _hnsw_index_names(db_rs)
            assert hnsw_names, (
                "Expected an HNSW index on the embeddings table after a "
                "compaction-triggering run, found none"
            )
