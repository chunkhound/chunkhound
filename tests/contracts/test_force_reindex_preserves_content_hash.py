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

Gap 3 (fixed): `compute_diff_blocking` (src/pipeline/pipeline.rs) used to
short-circuit with an empty `DiffResult` (no hashes at all) whenever
`chunks.db` didn't exist yet, skipping `compute_diff` entirely on the very
first index. That meant a brand-new file's hash was established later than
Python's equivalent diff logic (`IndexingCoordinator`'s "new file not in DB"
branch), which always computes and stores a new file's hash immediately —
this was a latent Rust/Python parity gap, not an inherent limitation.
Removing that short-circuit lets `compute_diff`'s existing "new file" branch
(which already computed a hash for a new file added to an *existing* DB) run
uniformly on the very first index too, so every file gets a hash from its
first appearance onward, matching the Python path.
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
        - Run 1: fresh-DB force-reindex. Every file — including one whose
          mtime will never subsequently change — gets a content_hash
          established immediately, matching the Python path's "new file not
          in DB" branch (IndexingCoordinator), which always computes a new
          file's hash right away rather than deferring it.
        - Touch main.py's mtime only — no byte changes.
        - Run 2: force-reindex again. main.py's mtime now differs from the
          DB's stored mtime, so the diff step recomputes its hash — it must
          survive into the write path. empty.py's mtime never changed, so
          its run-1 hash must carry forward unchanged.
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
        hash_after_run_1 = _content_hash_for_path(db_dir, "main.py")
        assert hash_after_run_1, (
            "a brand-new file must get a content_hash on its very first "
            "index, matching the Python path's new-file branch — got "
            f"{hash_after_run_1!r}"
        )
        empty_hash_after_run_1 = _content_hash_for_path(db_dir, "empty.py")
        assert empty_hash_after_run_1, (
            "empty.py is also a brand-new file on run 1 and must get a hash "
            f"immediately too, got {empty_hash_after_run_1!r}"
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
        assert _content_hash_for_path(db_dir, "empty.py") == empty_hash_after_run_1, (
            "empty.py's mtime never changed — its run-1 hash must carry "
            "forward unchanged, not get nulled out"
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

    def test_force_reindex_with_cleanup_off_still_persists_content_hash(
        self, fixture_dir: Path, tmp_path: Path
    ):
        """do_cleanup=False must not skip the diff that fills content hashes.

        Same touch + two force-reindex sequence as the cleanup-on test.
        Cleanup only means skip orphan deletes; hashes must still be written.
        """
        work_dir = tmp_path / "fixtures"
        shutil.copytree(fixture_dir, work_dir)

        db_dir = tmp_path / "db"
        kwargs = dict(skip_embeddings=True, incremental=False, do_cleanup=False)

        first = index_with_rust(work_dir, db_dir, **kwargs)
        assert first.chunks_written > 0, "baseline force-reindex should produce chunks"

        main_py = work_dir / "main.py"
        new_mtime = main_py.stat().st_mtime + 100.0
        os.utime(main_py, (new_mtime, new_mtime))

        index_with_rust(work_dir, db_dir, **kwargs)
        hash_after_run_2 = _content_hash_for_path(db_dir, "main.py")
        assert hash_after_run_2, (
            "force-reindex with cleanup off must persist main.py's content hash, "
            f"got {hash_after_run_2!r}"
        )

        index_with_rust(work_dir, db_dir, **kwargs)
        hash_after_run_3 = _content_hash_for_path(db_dir, "main.py")
        assert hash_after_run_3 == hash_after_run_2, (
            "cleanup-off force-reindex must carry an established hash forward: "
            f"{hash_after_run_2!r} -> {hash_after_run_3!r}"
        )
