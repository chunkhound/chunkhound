use std::path::{Path, PathBuf};

use duckdb::Connection;

use super::DuckDbHnswBackend;
use crate::error::DbError;
use crate::types::DbFileEntry;

/// Columns read by the pipeline's diff phase (`pipeline::differ::compute_diff`).
/// Keep in sync with `DuckDbHnswBackend::FILES_COLUMNS_DDL` in `schema.rs` — if
/// `modified_time` or `content_hash` are renamed there, update this too.
///
/// `modified_time` is written via `to_timestamp(?)` (an epoch -> TIMESTAMPTZ
/// conversion), which DuckDB then implicitly casts down into this naive
/// TIMESTAMP column using the session's local timezone — so the stored wall-
/// clock digits already have a local-time shift baked in. Casting back to
/// TIMESTAMPTZ before extracting the epoch reverses that same shift (assuming
/// the session timezone hasn't changed between write and read), matching what
/// Python's `datetime.timestamp()` does when it reads the same naive value
/// back via the driver. Extracting the epoch directly from the naive column
/// would skip that reversal and return a value off by the full UTC offset.
const FILE_STATE_SELECT: &str =
    "SELECT id, path, EXTRACT(EPOCH FROM modified_time::TIMESTAMPTZ), content_hash FROM files";

pub(super) fn read_file_states(backend: &DuckDbHnswBackend) -> Result<Vec<DbFileEntry>, DbError> {
    let db_path = Path::new(&backend.config.db_path);
    if !db_path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open(db_path)?;
    let mut stmt = conn.prepare(FILE_STATE_SELECT)?;
    let rows = stmt
        .query_map([], |row| {
            Ok(DbFileEntry {
                id: row.get(0)?,
                path: row.get(1)?,
                mtime: row.get(2)?,
                content_hash: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

/// Mirrors `indexing_coordinator.py`'s `_check_disk_usage_limit` for a
/// file-based (DuckDB) database: stats the exact `db_path` given (never a
/// WAL/`.compact`/`.old`/`.swap_intent` sidecar), compares with `>=`, and
/// fails OPEN (returns `None`) if the stat call itself errors or no limit is
/// configured — matching Python's "never block indexing on a measurement
/// error" behavior.
///
/// Returns `Some((size_mb, limit_mb))` when the limit is exceeded, `None`
/// otherwise.
pub(crate) fn check_disk_usage_limit(db_path: &Path, limit_mb: Option<f64>) -> Option<(f64, f64)> {
    let limit_mb = limit_mb?;
    let db_size = match std::fs::metadata(db_path) {
        Ok(meta) => meta.len(),
        Err(e) => {
            log::warn!(
                "Failed to check disk usage for {}: {}",
                db_path.display(),
                e
            );
            return None;
        }
    };
    // open() defers checkpoints until the WAL hits checkpoint_threshold (up to
    // 8GB by default), so writes can sit in the `.wal` sidecar well past the
    // main file's on-disk size — include it, or a deferred checkpoint lets
    // true usage silently blow past the configured limit before this trips.
    // A missing/unreadable WAL (e.g. already checkpointed) contributes 0
    // rather than failing the whole check open.
    let wal_path = PathBuf::from(format!("{}.wal", db_path.display()));
    let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
    let size_mb = (db_size + wal_size) as f64 / (1024.0 * 1024.0);
    (size_mb >= limit_mb).then_some((size_mb, limit_mb))
}

#[cfg(test)]
mod file_state_roundtrip_tests {
    use super::*;
    use crate::db::{DbBackend, DbConfig};

    #[test]
    fn mtime_roundtrip_is_timezone_symmetric() {
        // Regression test: the diff phase's read of `modified_time`
        // (FILE_STATE_SELECT) must reverse whatever local-timezone cast
        // `to_timestamp(?)` applied at write time, or every stored mtime
        // comes back shifted by the local UTC offset — pushing nearly every
        // file outside mtime_epsilon and forcing a full content-hash
        // re-verification (or reprocessing) of files that never changed.
        //
        // Rather than mutating the process's TZ (this crate forbids unsafe
        // code, and `std::env::set_var` requires it), set DuckDB's session
        // TimeZone explicitly and identically on both the write and read
        // connections — exactly what two connections opened by the same
        // process on the same non-UTC machine would see by default, and
        // deterministic regardless of the host running this test.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db");

        let original_mtime = 1_735_689_600.123_456_f64; // arbitrary UTC epoch
        {
            let conn = Connection::open(&db_path).expect("open for write");
            conn.execute_batch("SET TimeZone = 'America/New_York';")
                .expect("set tz");
            conn.execute_batch(
                "CREATE TABLE files (id BIGINT, path TEXT, modified_time TIMESTAMP, \
                 content_hash TEXT)",
            )
            .expect("create table");
            conn.execute(
                "INSERT INTO files VALUES (1, 'a.py', to_timestamp(?), 'abc')",
                [original_mtime],
            )
            .expect("insert");
        }

        let entries = {
            let conn = Connection::open(&db_path).expect("open for read");
            conn.execute_batch("SET TimeZone = 'America/New_York';")
                .expect("set tz");
            let mut stmt = conn.prepare(FILE_STATE_SELECT).expect("prepare");
            stmt.query_map([], |row| {
                Ok(DbFileEntry {
                    id: row.get(0)?,
                    path: row.get(1)?,
                    mtime: row.get(2)?,
                    content_hash: row.get(3)?,
                })
            })
            .expect("query")
            .collect::<Result<Vec<_>, _>>()
            .expect("rows")
        };

        assert_eq!(entries.len(), 1);
        let read_mtime = entries[0].mtime.expect("mtime must not be NULL");
        assert!(
            (read_mtime - original_mtime).abs() < 0.001,
            "read-back mtime {read_mtime} must match the written mtime {original_mtime} \
             (within float precision) even under a non-UTC session timezone"
        );
    }

    #[test]
    fn read_file_states_keeps_null_mtime_rows() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db");
        {
            let conn = Connection::open(&db_path).expect("open");
            conn.execute_batch(
                "CREATE TABLE files (id BIGINT, path TEXT, modified_time TIMESTAMP, \
                 content_hash TEXT)",
            )
            .expect("create table");
            conn.execute(
                "INSERT INTO files (id, path, modified_time, content_hash) \
                 VALUES (1, 'gone.py', NULL, NULL)",
                [],
            )
            .expect("insert null mtime");
        }

        let backend = DuckDbHnswBackend::new(DbConfig {
            db_path: db_path.to_string_lossy().into_owned(),
            compaction_threshold: Some(0.3),
            compaction_min_size_bytes: 52_428_800,
            insert_batch_size: 100,
        });
        let entries = backend.read_file_states().expect("read");
        assert_eq!(
            entries.len(),
            1,
            "NULL modified_time must not drop the row from the snapshot"
        );
        assert_eq!(entries[0].path, "gone.py");
        assert_eq!(entries[0].mtime, None);
        assert_eq!(entries[0].id, 1);
    }
}

#[cfg(test)]
mod disk_usage_limit_tests {
    use super::*;

    fn write_file_of_size(path: &Path, bytes: usize) {
        std::fs::write(path, vec![0u8; bytes]).expect("write fixture file");
    }

    #[test]
    fn no_limit_configured_never_exceeded() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        write_file_of_size(&db_path, 10 * 1024 * 1024);
        assert_eq!(check_disk_usage_limit(&db_path, None), None);
    }

    #[test]
    fn size_below_limit_not_exceeded() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        write_file_of_size(&db_path, 1024 * 1024); // 1 MB
        assert_eq!(check_disk_usage_limit(&db_path, Some(10.0)), None);
    }

    #[test]
    fn size_at_exact_limit_is_exceeded() {
        // Encodes Python's strict `>=` — a DB exactly at the limit already trips.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        write_file_of_size(&db_path, 5 * 1024 * 1024); // exactly 5 MB
        let result = check_disk_usage_limit(&db_path, Some(5.0));
        assert_eq!(result, Some((5.0, 5.0)));
    }

    #[test]
    fn size_above_limit_exceeded() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        write_file_of_size(&db_path, 10 * 1024 * 1024); // 10 MB
        let result = check_disk_usage_limit(&db_path, Some(5.0));
        assert_eq!(result, Some((10.0, 5.0)));
    }

    #[test]
    fn stat_failure_fails_open() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing_path = tmp.path().join("does_not_exist.duckdb");
        assert_eq!(check_disk_usage_limit(&missing_path, Some(0.0)), None);
    }

    #[test]
    fn sibling_wal_file_included_in_measurement() {
        // Main file well under the limit alone, but the deferred-checkpoint
        // `.wal` sidecar pushes combined usage over — must be counted, or a
        // large deferred checkpoint could let true usage silently exceed the
        // configured limit undetected.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let wal_path = tmp.path().join("t.duckdb.wal");
        write_file_of_size(&db_path, 1024); // 1 KB
        write_file_of_size(&wal_path, 20 * 1024 * 1024); // 20 MB
        let result = check_disk_usage_limit(&db_path, Some(5.0));
        let (size_mb, limit_mb) = result.expect("combined size should exceed the 5MB limit");
        assert!((size_mb - (20.0 + 1.0 / 1024.0)).abs() < 0.01);
        assert_eq!(limit_mb, 5.0);
    }

    #[test]
    fn missing_wal_file_contributes_zero() {
        // No `.wal` sidecar at all (e.g. already checkpointed) — must not be
        // treated as a stat failure, and must not fail the check open.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        write_file_of_size(&db_path, 10 * 1024 * 1024); // 10 MB
        let result = check_disk_usage_limit(&db_path, Some(5.0));
        assert_eq!(result, Some((10.0, 5.0)));
    }
}
