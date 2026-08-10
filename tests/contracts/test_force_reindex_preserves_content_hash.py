"""Force-reindex content-hash regression test.

Gap: the force-reindex path (`incremental=False`) ran the diff step only to
find deleted files for orphan cleanup, then discarded the content hashes,
disk stats, and DB row ids that diff step had already computed as a side
effect — forcing a redundant stat() per file downstream and writing every
touched file's `content_hash` column back to NULL, disabling the mtime-hash
"confirmed unchanged" optimization for that file until it's naturally
re-hashed on some later run. This asserts a force-reindex actually persists
a non-null content hash for a file whose mtime changed.
"""

from pathlib import Path

import duckdb
import pytest

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


@pytest.fixture
def fixture_dir() -> Path:
    return FIXTURE_DIR


def _content_hash_for_path(db_dir: Path, rel_path: str) -> str | None:
    db_file = db_dir / "chunks.db"
    conn = duckdb.connect(str(db_file))
    try:
        row = conn.execute(
            "SELECT content_hash FROM files WHERE path = ?", [rel_path]
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


class TestForceReindexPreservesContentHash:
    """A force-reindex must persist a content hash, not wipe it to NULL."""

    def test_force_reindex_after_mtime_bump_persists_content_hash(
        self, fixture_dir: Path, tmp_path: Path
    ):
        """Force-reindex twice with a touch-only mtime bump between runs.

        Contract:
        - Rust force-reindex (fresh DB) establishes file rows. content_hash
          is unavoidably NULL here — nothing to compare against yet.
        - Touch main.py's mtime only — no byte changes.
        - Rust force-reindex again on the same DB. main.py's mtime now
          differs from the DB's stored mtime, and the DB has no prior hash
          for it (from the first run), so the diff step computes and stashes
          a fresh hash for it — that hash must survive into the write path,
          not get discarded.
        """
        import shutil

        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)

        db_dir = tmp_path / "db"

        # ── Run 1: first-ever force-reindex ─────────────────────
        first = index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)
        assert first.chunks_written > 0, "baseline force-reindex should produce chunks"

        # ── Touch main.py's mtime only — no byte changes ────────
        import os

        main_py = work_dir / "main.py"
        new_mtime = main_py.stat().st_mtime + 100.0  # well outside mtime_epsilon_seconds
        os.utime(main_py, (new_mtime, new_mtime))

        # ── Run 2: force-reindex again on the same DB ───────────
        index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)

        content_hash = _content_hash_for_path(db_dir, "main.py")
        assert content_hash, (
            "force-reindex must persist main.py's content hash instead of "
            f"discarding the diff step's already-computed hash, got {content_hash!r}"
        )
