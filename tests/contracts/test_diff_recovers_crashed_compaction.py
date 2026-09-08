"""Contract test: incremental diff must recover a crashed compaction swap,
not silently reprocess every file from scratch.

Gap this closes: `read_file_states()` (src/db/duckdb_backend/read.rs) runs
`.swap_intent` crash recovery before reading -- but its only production
caller, `IndexingPipeline::compute_diff_blocking()` (src/pipeline/pipeline.rs),
used to short-circuit with "every file is new" whenever `chunks.db` didn't
exist *before* ever calling `read_file_states()`. A crashed compaction's
pre-swap/phase1/phase2 window (compaction.rs's 3-phase swap protocol) renames
the live DB aside to `.old` for its entire duration, so `chunks.db` missing
does not mean "no DB yet" -- it can mean "the real DB is one recovery call
away". That early return made the new recovery code in read_file_states()
unreachable for exactly the crash scenario it was written for, silently
downgrading "recover and diff correctly" into "reprocess everything".

This test simulates a crash right after compaction's phase-1 rename (the
exact on-disk state `run_attach_copy_compaction()` in compaction.rs leaves:
live file renamed to `.old`, intent file says "phase1", no `.compact` copy
yet) and asserts an incremental re-index of otherwise-unchanged files
recognizes them as unchanged -- proven by chunk ids staying identical, since
any file the diff treats as "changed" gets its chunks deleted and
re-inserted with fresh ids (src/db/duckdb_backend/write.rs's
insert_chunks_for_file), even if the content is byte-identical.
"""

import tempfile
from pathlib import Path

from tests.contracts.pipeline_harness import chunk_ids_for_path, index_with_rust

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pipeline"


def _simulate_crashed_phase1_compaction(db_dir: Path) -> None:
    """Reproduce the on-disk state left by a crash right after
    `run_attach_copy_compaction()`'s phase-1 rename: the live DB has been
    renamed aside to `.old` and the intent file says "phase1", but the
    `.compact` copy was never created (crash happened before that point).
    """
    db_file = db_dir / "chunks.db"
    old_file = db_dir / "chunks.db.old"
    wal_file = db_dir / "chunks.db.wal"
    intent_file = db_dir / "chunks.db.swap_intent"

    db_file.rename(old_file)
    if wal_file.exists():
        wal_file.rename(db_dir / "chunks.db.old.wal")
    intent_file.write_text("phase1")


def test_incremental_reindex_recovers_crashed_compaction_without_reprocessing():
    with tempfile.TemporaryDirectory() as tmp:
        db_dir = Path(tmp) / "db"

        index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=True, incremental=True)
        chunk_ids_before = chunk_ids_for_path(db_dir, "main.py")
        assert chunk_ids_before, "fixture file must produce at least one chunk"

        _simulate_crashed_phase1_compaction(db_dir)

        index_with_rust(FIXTURE_DIR, db_dir, skip_embeddings=True, incremental=True)

        assert not (db_dir / "chunks.db.swap_intent").exists(), (
            "a crashed swap_intent must be recovered by the diff phase, "
            "not left behind for a later run to trip over"
        )
        chunk_ids_after = chunk_ids_for_path(db_dir, "main.py")
        assert chunk_ids_after == chunk_ids_before, (
            "an unchanged file must keep its original chunk ids across a "
            "crashed-compaction-then-incremental-reindex cycle; different "
            "ids mean the diff treated it as new/changed instead of "
            "recognizing it as unchanged after recovery"
        )
