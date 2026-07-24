"""Phase 3 contract test — incremental updates.

Tests that re-indexing a directory only re-processes files that
actually changed, while producing the same final state as a full
re-index.
"""

import asyncio
import shutil
from pathlib import Path

import pytest

from tests.contracts.pipeline_harness import (
    disconnect_registry_db,
    index_with_python,
    index_with_rust,
)
from tests.contracts.mock_embed import MockEmbeddingProvider

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


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
        disconnect_registry_db()

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
        result_rs_inc = index_with_rust(
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