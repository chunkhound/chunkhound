use duckdb::Connection;

use super::DuckDbHnswBackend;
use crate::error::DbError;

#[derive(Debug, Clone)]
struct HnswIndexInfo {
    index_name: String,
    table_name: String,
    metric: String,
}

impl DuckDbHnswBackend {
    fn discover_hnsw_indexes(conn: &Connection) -> Result<Vec<HnswIndexInfo>, DbError> {
        let mut stmt = conn.prepare(
            "SELECT index_name, table_name, sql FROM duckdb_indexes()
             WHERE table_name SIMILAR TO 'embeddings_[0-9]+'
             AND schema_name = 'main'",
        )?;
        let rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();

        let indexes = rows
            .into_iter()
            .filter(|(name, _, sql)| {
                sql.as_deref()
                    .map(|s| s.to_uppercase().contains("USING HNSW"))
                    .unwrap_or(false)
                    || name.starts_with("hnsw_")
                    || name.starts_with("idx_hnsw_")
            })
            .map(|(name, table, _sql)| {
                let metric = Self::extract_hnsw_metric(conn, &name);
                HnswIndexInfo {
                    index_name: name,
                    table_name: table,
                    metric,
                }
            })
            .collect();
        Ok(indexes)
    }

    /// Return the live HNSW similarity metric from `pragma_hnsw_index_info()`.
    ///
    /// DuckDB strips the `WITH (metric = '...')` clause from `duckdb_indexes().sql`,
    /// so the CREATE INDEX DDL alone cannot tell us the metric a dropped index used.
    /// Mirrors `_extract_hnsw_metric` in `duckdb_provider.py` — must be called while
    /// the index still exists (i.e. before it is dropped for a rebuild).
    fn extract_hnsw_metric(conn: &Connection, index_name: &str) -> String {
        conn.query_row(
            "SELECT metric FROM pragma_hnsw_index_info() WHERE index_name = ? LIMIT 1",
            [index_name],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| "cosine".to_string())
    }
}

pub(super) fn drop_all_hnsw_indexes(backend: &mut DuckDbHnswBackend) -> Result<(), DbError> {
    let indexes = {
        let conn = backend.conn_or_err()?;
        DuckDbHnswBackend::discover_hnsw_indexes(conn)?
    };
    // Snapshot metrics and enter bulk mode before any DROP so a mid-loop
    // failure still causes close() to restore indexes (CREATE IF NOT EXISTS
    // is a no-op for indexes that never dropped). Assignments happen with
    // no live conn borrow.
    backend.saved_hnsw_metrics = indexes
        .iter()
        .filter_map(|idx| {
            idx.table_name
                .strip_prefix("embeddings_")
                .and_then(|s| s.parse::<u32>().ok())
                .map(|dims| (dims, idx.metric.clone()))
        })
        .collect();
    backend.hnsw_bulk_mode = true;

    let conn = backend.conn_or_err()?;
    #[cfg(test)]
    for (dropped, idx) in indexes.iter().enumerate() {
        if backend.fail_drop_after == Some(dropped) {
            return Err(DbError::Other(
                "simulated mid-loop HNSW drop failure".into(),
            ));
        }
        let safe_name = idx.index_name.replace('"', "\"\"");
        conn.execute(&format!("DROP INDEX IF EXISTS \"{safe_name}\""), [])?;
    }
    #[cfg(not(test))]
    for idx in &indexes {
        let safe_name = idx.index_name.replace('"', "\"\"");
        conn.execute(&format!("DROP INDEX IF EXISTS \"{safe_name}\""), [])?;
    }
    Ok(())
}

pub(super) fn ensure_all_hnsw_indexes(backend: &mut DuckDbHnswBackend) -> Result<(), DbError> {
    // Reset first so that any error path (including early returns) leaves bulk
    // mode off — otherwise close() would retry in a partially-indexed state.
    backend.hnsw_bulk_mode = false;
    if !backend.has_vss {
        return Ok(());
    }
    // Query the DB directly for embedding tables — mirrors Python's
    // _executor_ensure_all_hnsw_indexes which does not rely on in-memory tracked state.
    // This is more robust than known_dims when the connection is reopened after
    // compaction or when edge cases cause the in-memory set to diverge from DB state.
    // Scope the first conn borrow so it's dropped before we access self.saved_hnsw_metrics.
    let existing = {
        let conn = backend.conn_or_err()?;
        DuckDbHnswBackend::discover_embedding_tables(conn)?
    };
    let dims_metrics: Vec<(u32, String)> = existing
        .into_iter()
        .map(|(_, dims)| {
            let metric = backend.saved_hnsw_metrics.get(&dims).cloned();
            if metric.is_none() {
                // No captured metric for this dims — either this process never
                // saw a live index for it (e.g. drop_all_hnsw_indexes wasn't
                // called this session, such as after a mid-run crash), or the
                // index genuinely used cosine. Falling back to cosine is silent
                // data loss if a non-default metric was ever in use, so surface
                // it instead of guessing quietly.
                log::warn!(
                    "No captured HNSW metric for {dims}-dim embeddings — \
                     defaulting to cosine (this loses a non-default metric if \
                     one was previously configured for this table)"
                );
            }
            (dims, metric.unwrap_or_else(|| "cosine".to_string()))
        })
        .collect();
    let conn = backend.conn_or_err()?;
    // DuckDB VSS HNSW builds can be CPU-intensive.  Increase the thread
    // count and disable any internal timeout so large tables don't fail.
    let _ = conn.execute_batch("SET threads = 8");
    // Build inside a closure so a failed CREATE INDEX/CHECKPOINT can't
    // skip the thread-count restore below via an early `?` return —
    // otherwise this connection would stay pinned at 8 threads and
    // compete with a concurrently-running embed thread pool.
    let build_result: Result<(), DbError> = (|| {
        for (dims, metric) in &dims_metrics {
            let hnsw_name = format!("idx_hnsw_{dims}");
            conn.execute_batch(&format!(
                "CREATE INDEX IF NOT EXISTS \"{hnsw_name}\" ON \"embeddings_{dims}\" USING HNSW (embedding) WITH (metric = '{metric}')"
            ))?;
        }
        if !dims_metrics.is_empty() {
            conn.execute_batch("CHECKPOINT")?;
        }
        Ok(())
    })();
    // Restore a conservative thread count — this connection may still be
    // used for a concurrent write loop (the streaming pipeline's store
    // thread writes/checkpoints while the embed thread's rayon pool is
    // active), which must not compete with DuckDB's own internal
    // parallelism for this machine's cores. Unconditional: must run
    // whether or not the index build above succeeded.
    let _ = conn.execute_batch("SET threads = 1");
    build_result
}

#[cfg(test)]
mod hnsw_tests {
    use super::*;
    use crate::db::DbBackend;

    #[test]
    fn ensure_all_hnsw_indexes_preserves_non_cosine_metric() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db").to_string_lossy().into_owned();
        let config = crate::db::DbConfig {
            db_path: db_path.clone(),
            compaction_threshold: Some(0.3),
            compaction_min_size_bytes: 52_428_800,
            insert_batch_size: 100,
        };
        let mut backend = DuckDbHnswBackend::new(config);
        backend.open().expect("open");

        if !backend.has_vss {
            eprintln!("VSS extension unavailable, skipping ensure_all_hnsw_indexes metric test");
            return;
        }

        // Create an embedding table and a non-cosine HNSW index.
        {
            let conn = backend.conn_or_err().expect("conn");
            DuckDbHnswBackend::ensure_embedding_table_dims(conn, 3).expect("create embeddings_3");
            conn.execute_batch(
                "CREATE INDEX idx_hnsw_3 ON embeddings_3 USING HNSW (embedding) WITH (metric = 'l2sq')",
            )
            .expect("create l2sq HNSW index");
        }
        backend.known_dims.insert(3);

        // Simulate what the pipeline does: drop_all_hnsw_indexes (saves metrics) then
        // ensure_all_hnsw_indexes (rebuilds using saved metrics).
        backend.drop_all_hnsw_indexes().expect("drop");
        assert_eq!(
            backend.saved_hnsw_metrics.get(&3).map(|s| s.as_str()),
            Some("l2sq"),
            "drop_all_hnsw_indexes must save the original metric"
        );

        backend.ensure_all_hnsw_indexes().expect("ensure");

        let conn = backend.conn_or_err().expect("conn");
        let after = DuckDbHnswBackend::discover_hnsw_indexes(conn).expect("discover after ensure");
        assert_eq!(after.len(), 1);
        assert_eq!(
            after[0].metric, "l2sq",
            "ensure_all_hnsw_indexes must preserve the original non-cosine metric"
        );
    }

    #[test]
    fn ensure_all_hnsw_indexes_restores_index_dropped_outside_lifecycle() {
        // Crash between write_batch's HNSW drop (Step 2) and recreate (Step 5) leaves
        // embeddings_N tables with data but no HNSW index. open() must detect this via
        // an unconditional ensure_all_hnsw_indexes() call and restore it.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let db_path_str = db_path.to_string_lossy().into_owned();

        let mut backend1 =
            DuckDbHnswBackend::new(super::super::test_support::config(db_path_str.clone()));
        backend1.open().expect("open");
        if !backend1.has_vss {
            eprintln!("VSS extension unavailable, skipping");
            return;
        }
        backend1
            .write_batch(&super::super::test_support::embedding_batch("a", 128, 50))
            .expect("write");
        backend1.close().expect("close");

        // Simulate a crash: drop the HNSW index behind the backend's back.
        {
            let conn = Connection::open(&db_path).expect("reopen raw");
            let _ = conn.execute_batch("LOAD vss");
            conn.execute_batch("DROP INDEX IF EXISTS idx_hnsw_128")
                .expect("drop index");
            conn.execute_batch("CHECKPOINT").expect("checkpoint");
            let remaining = DuckDbHnswBackend::discover_hnsw_indexes(&conn).expect("discover");
            assert!(
                remaining.is_empty(),
                "expected HNSW index to be absent after manual drop"
            );
        }

        let mut backend2 = DuckDbHnswBackend::new(super::super::test_support::config(db_path_str));
        backend2.open().expect("open");
        backend2.close().expect("close");

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let _ = conn.execute_batch("LOAD vss");
        let restored = DuckDbHnswBackend::discover_hnsw_indexes(&conn).expect("discover");
        assert!(
            !restored.is_empty(),
            "HNSW index not restored after crash recovery"
        );
    }

    #[test]
    fn close_restores_hnsw_after_partial_drop_failure() {
        // If the second DROP fails after the first succeeded, close() must
        // still rebuild missing indexes because bulk mode was entered before
        // the loop.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let db_path_str = db_path.to_string_lossy().into_owned();

        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(db_path_str));
        backend.open().expect("open");
        if !backend.has_vss {
            eprintln!("VSS extension unavailable, skipping");
            return;
        }
        backend
            .write_batch(&super::super::test_support::embedding_batch("a", 8, 2))
            .expect("write 8-dim");
        backend
            .write_batch(&super::super::test_support::embedding_batch("b", 16, 2))
            .expect("write 16-dim");
        backend.ensure_all_hnsw_indexes().expect("ensure");

        let before = {
            let conn = backend.conn_or_err().expect("conn");
            DuckDbHnswBackend::discover_hnsw_indexes(conn).expect("discover")
        };
        assert_eq!(before.len(), 2, "need two HNSW indexes to fail mid-loop");

        backend.fail_drop_after = Some(1);
        let drop_err = backend
            .drop_all_hnsw_indexes()
            .expect_err("second DROP should fail");
        assert!(
            drop_err.to_string().contains("simulated mid-loop"),
            "unexpected drop error: {drop_err}"
        );
        assert!(
            backend.hnsw_bulk_mode,
            "bulk mode must be set before the failing DROP"
        );

        backend.close().expect("close restores remaining indexes");

        let conn = Connection::open(&db_path).expect("reopen for verification");
        let _ = conn.execute_batch("LOAD vss");
        let restored = DuckDbHnswBackend::discover_hnsw_indexes(&conn).expect("discover");
        assert_eq!(
            restored.len(),
            2,
            "both HNSW indexes must exist after close() following a partial drop"
        );
    }
}
