"""Phase 3 contract test — incremental updates.

Tests that re-indexing a directory only re-processes files that
actually changed, while producing the same final state as a full
re-index.
"""

import asyncio
import shutil
from pathlib import Path

import duckdb
import pytest

from tests.contracts.pipeline_harness import (
    IndexResult,
    _collect_chunk_tuples,
    _collect_embedding_tuples,
    index_with_python,
)
from tests.contracts.mock_embed import (
    MockEmbeddingProvider,
    embed_texts,
    MOCK_PROVIDER,
    MOCK_MODEL,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _disconnect_registry_db() -> None:
    """Force-disconnect the registry's DuckDB provider to clear per-process cache.

    DuckDB maintains a process-level cache of opened databases. When the Python
    indexing pipeline opens a DuckDB database, subsequent duckdb.connect() calls
    to the same path (even from Rust via py.allow_threads in the same process)
    reuse the cached state, returning stale data.

    Disconnecting the provider and unregistering it forces duckdb to release
    the cached state, so the next connection gets fresh data from disk.
    """
    try:
        from chunkhound.registry import get_registry
        registry = get_registry()
        db = registry.get_provider("database")
        if db is not None and hasattr(db, "disconnect"):
            db.disconnect()
        registry._providers.pop("database", None)
    except Exception:
        pass  # best-effort — test should still pass even if cleanup fails




@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


def _index_with_rust(
    fixture_dir: Path,
    db_dir: Path,
    *,
    skip_embeddings: bool = False,
    incremental: bool = False,
) -> IndexResult:
    """Index *fixture_dir* using the Rust pipeline."""
    try:
        from chunkhound_native import IndexingPipeline  # type: ignore[import-untyped]
    except ImportError:
        raise NotImplementedError(
            "Rust IndexingPipeline is not yet available in chunkhound_native."
        ) from None

    db_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "project_root": str(fixture_dir.resolve()),
        "db_path": str(db_dir.resolve()),
        "db_batch_size": 100,
        "compaction_threshold": 0.60,
        "compaction_batch_threshold": 10,
        "compaction_min_size_mb": 10,
        "parse_batch_size": 200,
        "parse_thread_pool_size": 4,
        "embed_batch_size": 200,
        "force_reindex": False,
        "mtime_epsilon_seconds": 0.01,
        "skip_cleanup": False,
        "skip_embeddings": skip_embeddings,
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

    report = pipeline.run(
        files=file_paths,
        parse_batch_callback=parse_batch_callback,
        embed_batch_callback=embed_texts if not skip_embeddings else None,
        progress_callback=None,
        incremental=incremental,
    )

    chunk_tuples = _collect_tuples(db_dir)
    embedding_tuples = _collect_embedding_tuples(db_dir)

    return IndexResult(
        files_processed=report.files_processed,
        chunks_written=report.chunks_written,
        embeddings_generated=report.embeddings_generated,
        chunk_tuples=chunk_tuples,
        embedding_tuples=embedding_tuples,
        errors=list(report.errors) if report.errors else [],
    )


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
        (
            str(r[0]),
            str(r[1]),
            str(r[2] or ""),
            str(r[3] or ""),
            int(r[4] or 0),
            int(r[5] or 0),
        )
        for r in rows
    ]


class TestIncrementalUpdates:
    """Verify incremental re-indexing produces same final state as full re-index."""

    def test_incremental_updates(self, fixture_dir: Path, tmp_path: Path):
        """Re-index after modifying one file — only changed file's chunks change.

        Contract:
        - Copy fixtures to temp dir (isolate from source)
        - Initial index → baseline state (Python)
        - Modify main.py in the copy
        - Python full re-index on fresh DB → reference final state
        - Rust incremental on original DB → must match reference
        - Rust must process fewer files than full re-index
        """
        # ── Isolate fixtures to a temp copy ────────────────────
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)

        # ── Step 1: Initial full index (Python) ────────────────
        db_initial = tmp_path / "db_initial"
        db_initial.mkdir()

        provider = MockEmbeddingProvider()
        result_initial = asyncio.run(
            index_with_python(
                work_dir,
                db_initial,
                skip_embeddings=False,
                embedding_provider=provider,
            )
        )
        assert result_initial.chunks_written > 0, "initial index should produce chunks"

        # IMPORTANT: Disconnect the DuckDB provider so that when Rust DuckDB writes
        # to the same DB file later (in the same process via py.allow_threads), the
        # Python DuckDB process-level cache does not return stale data (Invariant 18).
        # DuckDB maintains per-path connection state; if the provider still holds a
        # live connection, any subsequent duckdb.connect() to the same path reuses
        # the cached state from before Rust's writes.
        _disconnect_registry_db()

        # ── Step 2: Modify one fixture file ────────────────────
        main_py = work_dir / "main.py"
        original_content = main_py.read_text()
        modified_content = (
            original_content
            + "\n\ndef incremental_test_func():\n    return 'added in phase 3'\n"
        )
        main_py.write_text(modified_content)

        # ── Step 3: Python full re-index on fresh DB (reference) ──
        db_py_full = tmp_path / "db_py_full"
        db_py_full.mkdir()
        result_py_full = asyncio.run(
            index_with_python(
                work_dir,
                db_py_full,
                skip_embeddings=False,
                embedding_provider=provider,
            )
        )

        # ── Step 4: Rust incremental re-index (same DB as step 1) ─
        result_rs_inc = _index_with_rust(
            work_dir,
            db_initial,  # reuse the initial DB
            skip_embeddings=False,
            incremental=True,
        )

        # ── Step 5: Assert identical final chunk set ──────────
        # The Rust report.chunks_written counts only newly written chunks,
        # while Python counts all chunks. Compare the actual DB content.
        py_chunks = set(result_py_full.chunk_tuples)
        rs_chunks = set(result_rs_inc.chunk_tuples)
        missing = py_chunks - rs_chunks
        extra = rs_chunks - py_chunks
        assert not missing, f"Chunks missing from Rust incremental ({len(missing)}): {sorted(missing)[:3]}"
        assert not extra, f"Extra chunks in Rust incremental ({len(extra)}): {sorted(extra)[:3]}"

        # ── Step 6: Verify Rust was actually incremental ───────
        assert result_rs_inc.files_processed < result_py_full.files_processed, (
            f"Incremental should process fewer files: "
            f"Rust={result_rs_inc.files_processed} < Python={result_py_full.files_processed}"
        )