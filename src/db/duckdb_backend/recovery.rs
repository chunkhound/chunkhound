use std::path::{Path, PathBuf};

use super::DuckDbHnswBackend;
use crate::error::DbError;

/// Which `.swap_intent` phase `recover_swap_intent` applied.
/// `reopen_after_compaction_failure` uses this to decide whether a leftover
/// `.compact` is an incomplete phase-1 copy (safe to drop) or the phase-2
/// new DB (must not drop).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RecoveredSwap {
    None,
    PreSwap,
    Phase1,
    Phase2,
}

impl DuckDbHnswBackend {
    /// Write a compaction-intent marker file and fsync it to disk.
    /// Used by the 3-phase swap protocol to enable crash recovery (Invariant 17).
    pub(super) fn write_intent(path: &Path, phase: &str) -> Result<(), DbError> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(phase.as_bytes())?;
        f.sync_all()?;
        Ok(())
    }

    fn wal_sidecar(db_file: &Path) -> PathBuf {
        PathBuf::from(format!("{}.wal", db_file.display()))
    }

    /// True when `path` is absent or a zero-length file (Python's
    /// `live_missing_or_empty`). A directory is treated as present so
    /// dest-blocked recovery still fails closed instead of "restoring" over it.
    pub(super) fn is_missing_or_empty(path: &Path) -> bool {
        match std::fs::metadata(path) {
            Ok(meta) if meta.is_file() => meta.len() == 0,
            Ok(_) => false,
            Err(_) => true,
        }
    }

    /// Rename a DuckDB main file and keep its `.wal` sidecar with it.
    ///
    /// Dest WAL is deleted *before* the main rename so a crash after the
    /// main file has moved cannot leave a stale occupant WAL next to the
    /// new main file. If `from` has a WAL it is moved after the main file.
    pub(super) fn rename_db_with_wal(from: &Path, to: &Path) -> Result<(), DbError> {
        let from_wal = Self::wal_sidecar(from);
        let to_wal = Self::wal_sidecar(to);
        if to_wal.exists() {
            std::fs::remove_file(&to_wal)?;
        }
        std::fs::rename(from, to)?;
        if from_wal.exists() {
            std::fs::rename(&from_wal, &to_wal)?;
        }
        Ok(())
    }

    fn remove_db_file(path: &Path) -> Result<(), DbError> {
        std::fs::remove_file(path)?;
        let _ = std::fs::remove_file(Self::wal_sidecar(path));
        Ok(())
    }

    pub(super) fn remove_db_file_best_effort(path: &Path) {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(Self::wal_sidecar(path));
    }

    /// Drop an incomplete `.compact` left by a crashed phase-1 copy.
    /// Shared by `open()` and `reopen_after_compaction_failure()`.
    pub(super) fn discard_incomplete_compact_if_phase1(recovered: RecoveredSwap, db_path: &Path) {
        if !matches!(recovered, RecoveredSwap::PreSwap | RecoveredSwap::Phase1) {
            return;
        }
        if Self::is_missing_or_empty(db_path) {
            return;
        }
        let compact_path = PathBuf::from(format!("{}.compact", db_path.display()));
        if compact_path.exists() {
            Self::remove_db_file_best_effort(&compact_path);
        }
    }

    /// Finish or roll back an interrupted compaction swap (Invariant 17).
    ///
    /// Shared by `open()` (next-process crash recovery) and
    /// `reopen_after_compaction_failure()` (in-process fallback) so the two
    /// cannot drift. Errors are propagated so a failed rename cannot be
    /// followed by deleting the last remaining copy.
    pub(super) fn recover_swap_intent(db_path: &Path) -> Result<RecoveredSwap, DbError> {
        let intent_path = PathBuf::from(format!("{}.swap_intent", db_path.display()));
        if !intent_path.exists() {
            return Ok(RecoveredSwap::None);
        }
        let intent = match std::fs::read_to_string(&intent_path) {
            Ok(s) => s,
            Err(e) => {
                return Err(DbError::Other(format!(
                    "swap_intent present but unreadable: {e}"
                )));
            }
        };
        let old_path = PathBuf::from(format!("{}.old", db_path.display()));
        let compact_path = PathBuf::from(format!("{}.compact", db_path.display()));
        match intent.trim() {
            "pre-swap" => {
                Self::recover_pre_swap_or_phase1(db_path, &old_path, &intent_path, "pre-swap")?;
                Ok(RecoveredSwap::PreSwap)
            }
            "phase1" => {
                Self::recover_pre_swap_or_phase1(db_path, &old_path, &intent_path, "phase1")?;
                Ok(RecoveredSwap::Phase1)
            }
            "phase2" => {
                Self::recover_phase2(db_path, &old_path, &compact_path, &intent_path)?;
                Ok(RecoveredSwap::Phase2)
            }
            other => Err(DbError::Other(format!(
                "unrecognized swap_intent {other:?}"
            ))),
        }
    }

    fn recover_pre_swap_or_phase1(
        db_path: &Path,
        old_path: &Path,
        intent_path: &Path,
        phase: &str,
    ) -> Result<(), DbError> {
        // The db_path -> old_path rename may have completed before the
        // "phase1" intent write landed. Restore .old if the live path is
        // missing so Connection::open does not create an empty database
        // and orphan the real data.
        if old_path.exists() && Self::is_missing_or_empty(db_path) {
            if db_path.exists() {
                Self::remove_db_file(db_path)?;
            }
            Self::rename_db_with_wal(old_path, db_path)?;
        } else if !old_path.exists() && Self::is_missing_or_empty(db_path) {
            return Err(DbError::Other(format!(
                "{phase} crash recovery: live db and .old backup both missing"
            )));
        }
        let _ = std::fs::remove_file(intent_path);
        Ok(())
    }

    /// Finish a phase-2 compact→live rename, or restore `.old` if the
    /// compacted file is gone. Never deletes `.old` or the intent until
    /// `db_path` exists and `.compact` is gone.
    fn recover_phase2(
        db_path: &Path,
        old_path: &Path,
        compact_path: &Path,
        intent_path: &Path,
    ) -> Result<(), DbError> {
        if compact_path.exists() {
            // Finish the swap. Keep .old until rename succeeds so a failed
            // compact→db move cannot wipe every copy. Drop a leftover live
            // file/WAL so they cannot attach onto .compact.
            if db_path.exists() {
                Self::remove_db_file(db_path)?;
            }
            Self::rename_db_with_wal(compact_path, db_path)?;
        } else if Self::is_missing_or_empty(db_path) {
            // Compact already applied or never written; live path is empty.
            // Restore the pre-compaction backup instead of opening a new DB.
            if old_path.exists() {
                if db_path.exists() {
                    Self::remove_db_file(db_path)?;
                }
                Self::rename_db_with_wal(old_path, db_path)?;
            } else {
                return Err(DbError::Other(
                    "phase2 crash recovery: no compact, live db, or .old backup".into(),
                ));
            }
        }

        // Swap finished (or leftover sidecar after a successful rename):
        // live path exists and compact is gone. Safe to drop the backup.
        if !Self::is_missing_or_empty(db_path) && !compact_path.exists() {
            Self::remove_db_file_best_effort(old_path);
            let _ = std::fs::remove_file(intent_path);
        }
        Ok(())
    }
}

/// Crash recovery via `.swap_intent` files (Invariant 17) — ported from the
/// deleted `RustDbWriter` PyO3 wrapper's test suite so these invariants stay
/// covered without going through PyO3.
#[cfg(test)]
mod crash_recovery_tests {
    use duckdb::Connection;

    use super::*;
    use crate::db::DbBackend;

    #[test]
    fn pre_swap_intent_cleared_on_open() {
        // pre-swap with a live DB still present (rename had not happened).
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "pre-swap").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists(), "intent file must be removed");
    }

    #[test]
    fn pre_swap_intent_after_rename_restores_old_file() {
        // Simulates a crash between the db_path->old_path rename (Phase 2)
        // and the "phase1" intent write landing on disk — the breadcrumb
        // still reads "pre-swap" even though the rename already happened.
        // Regression test for the data-loss bug where open() would silently
        // create a fresh empty database at db_path and orphan the real data
        // sitting at old_path.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        // Real data lives at old_path; db_path does not exist, matching the
        // post-rename, pre-"phase1"-write crash state.
        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "pre-swap").expect("write intent");
        assert!(!db_path.exists(), "db_path must not exist pre-recovery");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists());
        assert!(!old_path.exists());

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(
            count, 1,
            "seed.py must have been recovered from the .old backup, not \
             silently discarded by a fresh empty database"
        );
    }

    #[test]
    fn phase1_intent_restores_old_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        // Simulate a DB that was backed up but not yet swapped: a valid DB
        // with one seed file lives at old_path.
        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "phase1").expect("write intent");

        // open() should detect phase1 and rename old -> db_path.
        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists());
        assert!(!old_path.exists());

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(
            count, 1,
            "seed.py must have been recovered from the .old backup"
        );
    }

    #[test]
    fn read_file_states_recovers_phase1_intent_before_reading() {
        // Regression test: read_file_states() opens its own raw connection
        // rather than going through open(), so it must run the same
        // swap_intent recovery open() does before reading -- otherwise a
        // crashed phase-1 (db_path missing, real data still at the .old
        // backup) makes the diff phase see an empty snapshot and conclude
        // every previously indexed file was deleted.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "phase1").expect("write intent");
        assert!(!db_path.exists(), "db_path must not exist pre-recovery");

        // Deliberately call read_file_states() directly, without open() --
        // this is exactly how the pipeline's diff phase calls it.
        let backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        let entries = backend.read_file_states().expect("read_file_states");

        assert!(!intent_path.exists(), "intent file must be recovered away");
        assert_eq!(
            entries.len(),
            1,
            "seed.py must be visible to the diff snapshot, recovered from the \
             .old backup, not silently reported as an empty/all-deleted DB"
        );
        assert_eq!(entries[0].path, "seed.py");
    }

    #[test]
    fn phase2_intent_removes_old_file() {
        // Already-swapped leftover: compact→db rename finished, .old and
        // intent were not cleaned up. Recovery must keep the live DB and
        // drop the sidecar files.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        // Live DB must exist first: this is the post-rename leftover-sidecar
        // case. Writing .old + intent before bootstrap would make recovery
        // treat the marker as the only copy and rename it onto db_path.
        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap.close().expect("close");

        std::fs::write(&old_path, "stale backup marker").expect("write old");
        std::fs::write(&intent_path, "phase2").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists());
        assert!(!old_path.exists());
    }

    #[test]
    fn phase2_intent_finishes_interrupted_compact_rename() {
        // Crash after writing phase2 intent, before compact→db rename:
        // live path is empty, real data is in .compact, pre-compaction
        // backup is in .old. Recovery must finish the swap, not open an
        // empty database and not prefer .old over .compact.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let compact_path = tmp.path().join("t.duckdb.compact");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut old_db = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        old_db.open().expect("open old");
        old_db
            .write_batch(&super::super::test_support::single_file_batch("old.py"))
            .expect("write old");
        old_db.close().expect("close old");

        let mut compact_db = DuckDbHnswBackend::new(super::super::test_support::config(
            compact_path.to_string_lossy().into_owned(),
        ));
        compact_db.open().expect("open compact");
        compact_db
            .write_batch(&super::super::test_support::single_file_batch("compact.py"))
            .expect("write compact");
        compact_db.close().expect("close compact");

        std::fs::write(&intent_path, "phase2").expect("write intent");
        assert!(!db_path.exists(), "db_path must not exist pre-recovery");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists(), "intent must be cleared after swap");
        assert!(!old_path.exists(), "backup must be dropped after swap");
        assert!(
            !compact_path.exists(),
            "compact must have been renamed onto the live path"
        );

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let path: String = conn
            .query_row("SELECT path FROM files", [], |r| r.get(0))
            .expect("path");
        assert_eq!(
            path, "compact.py",
            "live DB must be the compacted copy, not the .old backup \
             and not a freshly created empty database"
        );
    }

    #[test]
    fn phase2_intent_restores_old_when_compact_and_live_missing() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "phase2").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists());
        assert!(!old_path.exists());

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 1, "must restore .old when compact and live are gone");
    }

    #[test]
    fn phase2_failed_swap_keeps_old_and_intent() {
        // Live path is a directory so remove_file(db_path) fails before the
        // compact rename. Recovery must leave .old and the intent in place
        // so a later open can retry instead of wiping the last copies.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let compact_path = tmp.path().join("t.duckdb.compact");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        std::fs::create_dir(&db_path).expect("dir at live path");
        std::fs::write(&old_path, "backup").expect("write old");
        std::fs::write(&compact_path, "compact").expect("write compact");
        std::fs::write(&intent_path, "phase2").expect("write intent");

        DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("blocked dest must fail closed");

        assert!(
            old_path.exists(),
            "must not delete .old after a failed compact→db swap"
        );
        assert!(
            intent_path.exists(),
            "must keep the intent so the next open retries"
        );
        assert!(
            compact_path.exists(),
            "compact must remain the new-db candidate"
        );
    }

    #[test]
    fn pre_swap_intent_cleared_on_open_db_extension() {
        // Regression guard: PathBuf::set_extension() on "chunks.db" would produce
        // "chunks.duckdb.swap_intent" instead of "chunks.db.swap_intent". The
        // correct implementation builds the intent path via string concatenation.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("chunks.db");
        let intent_path = tmp.path().join("chunks.db.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "pre-swap").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(
            !intent_path.exists(),
            "intent file must be removed; wrong path construction would leave it untouched"
        );
    }

    #[test]
    fn phase2_all_copies_missing_fails_closed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");
        std::fs::write(&intent_path, "phase2").expect("write intent");

        let err = DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("must not create an empty live db");
        assert!(
            err.to_string().contains("no compact, live db, or .old"),
            "unexpected error: {err}"
        );
        assert!(
            intent_path.exists(),
            "intent must remain so a later open retries"
        );
        assert!(!db_path.exists(), "must not create a live file");
    }

    #[test]
    fn garbage_intent_live_missing_fails_closed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");
        std::fs::write(&old_path, "backup").expect("write old");
        std::fs::write(&intent_path, "pha").expect("write garbage intent");

        let err = DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("unknown intent + missing live must fail closed");
        assert!(
            err.to_string().contains("unrecognized swap_intent"),
            "unexpected error: {err}"
        );
        assert!(old_path.exists(), "must not touch .old");
        assert!(
            intent_path.exists(),
            "must keep the unreadable-phase intent"
        );
        assert!(!db_path.exists());
    }

    #[test]
    fn unreadable_intent_live_missing_fails_closed() {
        // A directory at the intent path makes read_to_string fail.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");
        std::fs::write(&old_path, "backup").expect("write old");
        std::fs::create_dir(&intent_path).expect("intent as directory");

        DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("unreadable intent + missing live must fail closed");
        assert!(old_path.exists(), "must not touch .old");
        assert!(!db_path.exists());
    }

    #[test]
    fn garbage_intent_live_present_fails_closed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("live.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&old_path, "backup").expect("write old");
        std::fs::write(&intent_path, "pha").expect("write garbage intent");

        DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("unknown intent must fail even when live exists");
        assert!(db_path.exists(), "must not replace the live db");
        assert!(old_path.exists(), "must not touch .old");
        assert!(intent_path.exists(), "must keep the intent");
    }

    #[test]
    fn unreadable_intent_live_present_fails_closed() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap.close().expect("close");

        std::fs::create_dir(&intent_path).expect("intent as directory");

        DuckDbHnswBackend::recover_swap_intent(&db_path)
            .expect_err("unreadable intent must fail even when live exists");
        assert!(db_path.exists());
    }

    #[test]
    fn empty_live_file_is_treated_as_missing_and_restored_from_old() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut bootstrap = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&db_path, b"").expect("empty live");
        std::fs::write(&intent_path, "phase1").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 1, "zero-length live must be replaced from .old");
    }

    #[test]
    fn rename_db_with_wal_drops_dest_wal_before_main_rename() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let from = tmp.path().join("from.duckdb");
        let to = tmp.path().join("to.duckdb");
        let dest_wal = tmp.path().join("to.duckdb.wal");
        std::fs::write(&from, b"new-main").expect("write from");
        std::fs::write(&dest_wal, b"STALE_WAL_MUST_NOT_REPLAY").expect("write dest wal");

        DuckDbHnswBackend::rename_db_with_wal(&from, &to).expect("rename");

        assert_eq!(std::fs::read(&to).expect("read dest"), b"new-main");
        assert!(
            !dest_wal.exists(),
            "dest WAL must be gone before/without a source WAL to replace it"
        );
    }

    #[test]
    fn phase1_open_deletes_incomplete_compact_beside_restored_live() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let compact_path = tmp.path().join("t.duckdb.compact");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut old_db = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        old_db.open().expect("open old");
        old_db
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write old");
        old_db.close().expect("close old");

        std::fs::write(&compact_path, "incomplete compact").expect("write compact");
        std::fs::write(&intent_path, "phase1").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(
            !compact_path.exists(),
            "open() must drop leftover phase1 .compact, not only the in-process fallback"
        );
    }

    #[test]
    fn leftover_live_wal_is_dropped_when_finishing_phase2_swap() {
        // After live→.old, a stale t.duckdb.wal can remain at the live
        // name. Finishing compact→live must not let DuckDB replay that WAL
        // against the compacted main file.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let compact_path = tmp.path().join("t.duckdb.compact");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");
        let stale_wal = tmp.path().join("t.duckdb.wal");

        let mut old_db = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        old_db.open().expect("open old");
        old_db
            .write_batch(&super::super::test_support::single_file_batch("old.py"))
            .expect("write old");
        old_db.close().expect("close old");

        let mut compact_db = DuckDbHnswBackend::new(super::super::test_support::config(
            compact_path.to_string_lossy().into_owned(),
        ));
        compact_db.open().expect("open compact");
        compact_db
            .write_batch(&super::super::test_support::single_file_batch("compact.py"))
            .expect("write compact");
        compact_db.close().expect("close compact");

        std::fs::write(&stale_wal, b"STALE_WAL_MUST_NOT_REPLAY").expect("write stale wal");
        std::fs::write(&intent_path, "phase2").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");
        backend.close().expect("close");

        if stale_wal.exists() {
            let bytes = std::fs::read(&stale_wal).expect("read wal");
            assert!(
                !bytes
                    .windows(b"STALE_WAL_MUST_NOT_REPLAY".len())
                    .any(|w| w == b"STALE_WAL_MUST_NOT_REPLAY"),
                "stale live WAL must not remain after compact→live"
            );
        }

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let path: String = conn
            .query_row("SELECT path FROM files", [], |r| r.get(0))
            .expect("path");
        assert_eq!(path, "compact.py");
    }

    #[test]
    fn phase1_failure_deletes_incomplete_compact_beside_restored_live() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let compact_path = tmp.path().join("t.duckdb.compact");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        let mut old_db = DuckDbHnswBackend::new(super::super::test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        old_db.open().expect("open old");
        old_db
            .write_batch(&super::super::test_support::single_file_batch("seed.py"))
            .expect("write old");
        old_db.close().expect("close old");

        std::fs::write(&compact_path, "incomplete compact").expect("write compact");
        std::fs::write(&intent_path, "phase1").expect("write intent");

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend
            .reopen_after_compaction_failure()
            .expect("reopen after failure");
        backend.close().expect("close");

        assert!(
            !compact_path.exists(),
            "phase1 leftover compact must be dropped"
        );
        assert!(!intent_path.exists());
        let conn = Connection::open(&db_path).expect("reopen for verification");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 1);
    }
}
