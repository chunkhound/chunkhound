"""Content-hash fallback contract test — mtime-touched, content-unchanged files.

Gap: the Rust pipeline's incremental diff was mtime-only. Any operation that
bumps a file's mtime without changing its bytes (git checkout, CI artifact
restore, container rebuild, a plain `touch`) made the pipeline treat the file
as genuinely changed — full re-embed and full chunk delete+reinsert — even
though nothing changed. This test asserts a touch-only mtime bump is confirmed
unchanged via content hash and skipped entirely: no re-embedding, no chunk
rewrite.
"""

import asyncio
import shutil
from pathlib import Path

import duckdb
import pytest

from tests.contracts.mock_embed import MockEmbeddingProvider
from tests.contracts.pipeline_harness import (
    disconnect_registry_db,
    index_with_python,
    index_with_rust,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


def _chunk_ids_for_path(db_dir: Path, rel_path: str) -> list[int]:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        rows = conn.execute(
            """
            SELECT c.id
            FROM chunks c JOIN files f ON f.id = c.file_id
            WHERE f.path = ?
            ORDER BY c.id
            """,
            [rel_path],
        ).fetchall()
    finally:
        conn.close()
    return [int(r[0]) for r in rows]


class TestTouchMtimeUnchanged:
    """Verify a touch-only mtime bump (no content change) is skipped entirely."""

    def test_touch_mtime_without_content_change_skips_reembed(
        self, fixture_dir: Path, tmp_path: Path
    ):
        """Re-index after touching main.py's mtime only — zero re-embeds, zero rewrite.

        Contract:
        - Copy fixtures to temp dir (isolate from source)
        - Full baseline index (Python, with real embeddings) → establish chunk rows
        - Touch main.py's mtime only — no byte changes
        - Rust incremental re-index on the same DB
        - Assert: zero embeddings generated, zero chunks written, and the
          existing chunk rows for main.py were not deleted+reinserted (ids
          stable) — content-hash confirms the file is unchanged despite the
          mtime bump.
        """
        # ── Isolate fixtures to a temp copy ────────────────────
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)

        # ── Step 1: Full baseline index (Python, with embeddings) ──
        db_dir = tmp_path / "db"
        db_dir.mkdir()

        provider = MockEmbeddingProvider()
        baseline = asyncio.run(
            index_with_python(
                work_dir,
                db_dir,
                skip_embeddings=False,
                embedding_provider=provider,
            )
        )
        assert baseline.chunks_written > 0, "baseline index should produce chunks"

        # IMPORTANT: Disconnect the DuckDB provider so that when Rust DuckDB
        # writes to the same DB file later (in the same process via
        # py.allow_threads), the Python DuckDB process-level cache does not
        # return stale data.
        disconnect_registry_db()

        before_ids = _chunk_ids_for_path(db_dir, "main.py")
        assert before_ids, "main.py should have chunk rows after the baseline index"

        # ── Step 2: Touch main.py's mtime only — no byte changes ──
        main_py = work_dir / "main.py"
        new_mtime = main_py.stat().st_mtime + 100.0  # well outside mtime_epsilon_seconds=0.01
        import os

        os.utime(main_py, (new_mtime, new_mtime))

        # ── Step 3: Rust incremental re-index (same DB) ────────
        result = index_with_rust(
            work_dir,
            db_dir,
            skip_embeddings=False,
            incremental=True,
        )

        # ── Step 4: Assert the touch-only file was skipped entirely ──
        assert result.embeddings_generated == 0, (
            f"touch-only mtime bump must not trigger re-embedding, "
            f"got {result.embeddings_generated}"
        )
        assert result.chunks_written == 0, (
            f"touch-only mtime bump must not trigger chunk rewrite, "
            f"got {result.chunks_written}"
        )

        after_ids = _chunk_ids_for_path(db_dir, "main.py")
        assert after_ids == before_ids, (
            "main.py's chunk rows must not be deleted+reinserted for a "
            f"touch-only mtime bump: before={before_ids}, after={after_ids}"
        )
