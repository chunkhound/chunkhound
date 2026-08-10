"""Force-reindex content-hash regression test.

Gap 1 (fixed): the force-reindex path (`incremental=False`) ran the diff step
only to find deleted files for orphan cleanup, then discarded the content
hashes, disk stats, and DB row ids that diff step had already computed as a
side effect — forcing a redundant stat() per file downstream and writing
every touched file's `content_hash` column back to NULL.

Gap 2 (fixed): even after threading that diff data through, `compute_diff`
itself only ever populated those side maps for files it decided needed
*reprocessing* (mtime differed). A file whose mtime hadn't changed since the
last write — the common case for a repeated force-reindex, or any file
between two runs where nothing touched it — still had no hash to carry
forward, so a force-reindex (which reprocesses every file regardless of the
diff's verdict) kept nulling out an already-established hash on every run.
`compute_diff` now carries the DB's existing hash forward verbatim for an
unchanged file (no extra read — mtime match is proof enough) instead of
leaving it for the write path to null out.

Note what's *not* a bug: a file whose mtime has never once changed since its
very first index can never have a hash established in the first place (there
was never a mtime-differs event to trigger computing one) — that's inherent,
not a regression, and harmless: the mtime-only fast path already handles
"provably untouched" files correctly without needing a hash at all.
"""

import os
import shutil
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
        """Force-reindex three times, with a touch-only mtime bump once.

        Contract:
        - Run 1: fresh-DB force-reindex. content_hash is unavoidably NULL
          for every file — nothing to compare against yet.
        - Touch main.py's mtime only — no byte changes.
        - Run 2: force-reindex again. main.py's mtime now differs from the
          DB's stored mtime and there's no prior hash for it, so the diff
          step computes and stashes a fresh one — it must survive into the
          write path. Files whose mtime never changed (and so never had a
          hash established either) are still NULL here — expected, harmless.
        - Run 3: force-reindex again with *nothing* touched. main.py's mtime
          now matches the DB again, so the diff step doesn't recompute
          anything — it must carry main.py's run-2 hash forward unchanged
          rather than nulling it out just because this file wasn't
          reprocessed for content reasons.
        """
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)

        db_dir = tmp_path / "db"

        # ── Run 1: first-ever force-reindex ─────────────────────
        first = index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)
        assert first.chunks_written > 0, "baseline force-reindex should produce chunks"
        assert _content_hash_for_path(db_dir, "main.py") is None, (
            "a fresh index has nothing to compare against yet — NULL is expected here"
        )

        # ── Touch main.py's mtime only — no byte changes ────────
        main_py = work_dir / "main.py"
        new_mtime = main_py.stat().st_mtime + 100.0  # well outside mtime_epsilon_seconds
        os.utime(main_py, (new_mtime, new_mtime))

        # ── Run 2: force-reindex again on the same DB ───────────
        index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)

        hash_after_run_2 = _content_hash_for_path(db_dir, "main.py")
        assert hash_after_run_2, (
            "force-reindex must persist main.py's content hash instead of "
            f"discarding the diff step's already-computed hash, got {hash_after_run_2!r}"
        )
        assert _content_hash_for_path(db_dir, "empty.py") is None, (
            "empty.py's mtime never changed and it never had a hash established "
            "either — staying NULL here is expected, not a regression"
        )

        # ── Run 3: force-reindex again with nothing touched ─────
        index_with_rust(work_dir, db_dir, skip_embeddings=True, incremental=False)

        hash_after_run_3 = _content_hash_for_path(db_dir, "main.py")
        assert hash_after_run_3 == hash_after_run_2, (
            "a force-reindex must carry an already-established hash forward "
            "for a file whose mtime didn't change, not null it out just "
            f"because the file wasn't reprocessed: {hash_after_run_2!r} -> "
            f"{hash_after_run_3!r}"
        )
