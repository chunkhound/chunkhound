"""Skip-reason persistence contract test — files with nothing to index.

Gap: the Rust pipeline silently dropped any file that produced a parse error
or zero chunks with no detected language (images, binaries, generated data)
— no database row was ever written for it. Because the diff phase only ever
asks "does a row with a matching mtime exist for this path?", a file with no
row is indistinguishable from one that was never seen, so it was rediscovered
as "new" and reprocessed on every subsequent run, forever, even though
nothing about it ever changes. This test asserts such a file gets a
persistent row on the first run and is not rediscovered as new on the next.
"""

import shutil
from pathlib import Path

import duckdb

from tests.contracts.pipeline_harness import index_with_rust

FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "pipeline_skip_reason"
)


def _file_row(db_dir: Path, rel_path: str) -> tuple | None:
    conn = duckdb.connect(str(db_dir / "chunks.db"), read_only=True)
    try:
        return conn.execute(
            "SELECT id, skip_reason FROM files WHERE path = ?", [rel_path]
        ).fetchone()
    finally:
        conn.close()


class TestBinaryFileNotReprocessed:
    """A file with nothing to index gets a persistent row, checked only once."""

    def test_binary_file_gets_row_and_is_not_rediscovered(self, tmp_path: Path):
        work_dir = tmp_path / "fixtures"
        shutil.copytree(FIXTURE_DIR, work_dir)
        db_dir = tmp_path / "db"

        # ── Run 1: fresh index — both files are "new" ──
        result_1 = index_with_rust(
            work_dir, db_dir, skip_embeddings=True, incremental=True
        )
        assert result_1.errors == []
        assert result_1.files_processed == 2, (
            f"both real_code.py and blob.bin should be attempted on the first run, "
            f"got files_processed={result_1.files_processed}"
        )

        row = _file_row(db_dir, "blob.bin")
        assert row is not None, (
            "blob.bin must get a persistent files row even though it has no "
            "chunks to index — without one, the diff phase can't tell it apart "
            "from a file that was never seen, and will reprocess it forever"
        )
        _, skip_reason = row
        assert skip_reason is not None, (
            "a file with nothing to index must have a non-null skip_reason"
        )

        real_row = _file_row(db_dir, "real_code.py")
        assert real_row is not None
        assert real_row[1] is None, (
            "a successfully-parsed file must not carry a skip_reason"
        )

        # ── Run 2: nothing changed — blob.bin must not be rediscovered ──
        result_2 = index_with_rust(
            work_dir, db_dir, skip_embeddings=True, incremental=True
        )
        assert result_2.errors == []
        assert result_2.files_processed == 0, (
            "no file changed since run 1 — blob.bin's mtime-matching row must "
            f"prevent it (and real_code.py) from being reprocessed, got "
            f"files_processed={result_2.files_processed}"
        )

        # Still exactly one row for blob.bin — not duplicated.
        conn = duckdb.connect(str(db_dir / "chunks.db"), read_only=True)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM files WHERE path = 'blob.bin'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == 1
