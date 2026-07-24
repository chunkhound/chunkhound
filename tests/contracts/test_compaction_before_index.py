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

import duckdb
import pytest
import tempfile
from pathlib import Path

from tests.contracts.pipeline_harness import (
    IndexResult,
    assert_identical,
    index_with_python,
)
from tests.contracts.mock_embed import (
    embed_texts,
    MOCK_PROVIDER,
    MOCK_MODEL,
    MockEmbeddingProvider,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


async def _index_with_rust_forced_compaction(
    fixture_dir: Path, db_dir: Path
) -> tuple[IndexResult, list[str]]:
    """Index *fixture_dir* with the Rust pipeline, forcing compaction to run.

    Returns ``(IndexResult, phases_seen)`` where ``phases_seen`` is every
    phase string the progress callback observed, in call order.
    """
    from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]

    db_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "project_root": str(fixture_dir.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        # Force needs_compaction() to always return true: threshold=0.0
        # means any non-negative free/waste ratio (always true) clears the
        # bar, and min_size_mb=0 means even a tiny reclaimable amount clears
        # the min-size gate.
        "compaction_threshold": 0.0,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 0,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "skip_cleanup": False,
        "skip_embeddings": False,
        "per_file_timeout_secs": 3.0,
        "per_file_timeout_min_size_kb": 128,
        "detect_embedded_sql": True,
        "config_file_size_threshold_kb": 20,
        "embedding_provider": MOCK_PROVIDER,
        "embedding_model": MOCK_MODEL,
    }

    pipeline = IndexingPipeline(config_dict)

    files = sorted(fixture_dir.resolve().glob("*"))
    file_paths = [str(f) for f in files if f.is_file()]

    from chunkhound.pipeline_bridge import parse_batch_callback

    phases_seen: list[str] = []

    def progress_callback(phase: str, current: int, total: int) -> None:
        phases_seen.append(phase)

    report = pipeline.run(
        files=file_paths,
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=embed_texts,
        progress_callback=progress_callback,
    )

    result = IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=_collect_tuples(db_dir),
        embedding_tuples=_collect_embedding_tuples(db_dir),
        errors=list(report.errors) if report.errors else [],
    )
    return result, phases_seen


def _collect_tuples(db_dir: Path) -> list[tuple[str, str, str, str, int, int]]:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    rows = conn.execute(
        """
        SELECT f.path, c.chunk_type, c.symbol, c.code, c.start_line, c.end_line
        FROM chunks c JOIN files f ON f.id = c.file_id
        ORDER BY f.path, c.start_line, c.symbol
        """
    ).fetchall()
    conn.close()
    return [
        (str(r[0]), str(r[1]), str(r[2] or ""), str(r[3] or ""), int(r[4] or 0), int(r[5] or 0))
        for r in rows
    ]


def _collect_embedding_tuples(
    db_dir: Path,
) -> list[tuple[str, str, str, str, str, int, tuple[float, ...]]]:
    db_file = db_dir / "chunks.db"
    if not db_file.exists():
        return []
    conn = duckdb.connect(str(db_file))
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'embeddings_%'"
    ).fetchall()
    tuples: list[tuple[str, str, str, str, str, int, tuple[float, ...]]] = []
    for (table_name,) in tables:
        rows = conn.execute(
            f"""
            SELECT f.path, c.chunk_type, c.symbol, e.provider, e.model, e.dims, e.embedding
            FROM {table_name} e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN files f ON f.id = c.file_id
            ORDER BY f.path, c.start_line, c.symbol
            """
        ).fetchall()
        for row in rows:
            vec = row[6]
            prefix = tuple(float(v) for v in vec[:8])
            tuples.append(
                (str(row[0]), str(row[1]), str(row[2] or ""), str(row[3]), str(row[4]), int(row[5]), prefix)
            )
    conn.close()
    return tuples


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

            result_rs, phases = await _index_with_rust_forced_compaction(FIXTURE_DIR, db_rs)

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

            result_rs, _phases = await _index_with_rust_forced_compaction(FIXTURE_DIR, db_rs)

            assert_identical(result_py, result_rs)

            hnsw_names = _hnsw_index_names(db_rs)
            assert hnsw_names, (
                "Expected an HNSW index on the embeddings table after a "
                "compaction-triggering run, found none"
            )
