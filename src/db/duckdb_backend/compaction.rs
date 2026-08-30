use std::path::PathBuf;

use duckdb::Connection;

use super::DuckDbHnswBackend;
use crate::db::DbBackend;
use crate::error::DbError;

/// Two-signal compaction metrics (Phase 0).
#[derive(Debug, Clone)]
struct CompactionStats {
    /// Fraction of DB blocks that are free (0.0–1.0).
    free_ratio: f64,
    /// Fraction of stored rows that are dead (0.0–1.0).
    row_waste_ratio: f64,
    /// Estimated reclaimable bytes = db_size × max(free_ratio, row_waste_ratio).
    reclaimable: u64,
}

impl DuckDbHnswBackend {
    /// Check available disk space on the filesystem containing `dir`.
    /// Returns None when the platform does not support the query.
    /// ATTACH + INSERT SELECT compaction — copies canonical tables into a
    /// fresh DB file via DuckDB's in-process attach mechanism, avoiding the
    /// filesystem I/O overhead of EXPORT/IMPORT via Parquet.
    ///
    /// Phase 1: CHECKPOINT + close live connection.
    /// Phase 2: ATTACH old DB as 'src', CREATE tables + sequences, INSERT
    ///          SELECT data, DETACH, CHECKPOINT.
    /// Phase 3: Atomic rename of compacted DB to active path, cleanup,
    ///          reopen (which also ensures HNSW indexes).
    fn run_attach_copy_compaction(&mut self) -> Result<(), DbError> {
        let db_path = PathBuf::from(&self.config.db_path);
        let compact_path = PathBuf::from(format!("{}.compact", self.config.db_path));
        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
        let intent_path = PathBuf::from(format!("{}.swap_intent", self.config.db_path));

        // --- Phase 1: CHECKPOINT + close live connection --------------------
        let conn = self.conn_or_err()?;

        // Drop staging temp tables so they don't interfere with the copy.
        let _ = conn.execute_batch("DROP TABLE IF EXISTS rust_temp_chunks");
        if let Ok(mut stmt) = conn.prepare(
            "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'rust_temp_%'",
        ) {
            if let Ok(tables) = stmt.query_map([], |row| row.get::<_, String>(0)) {
                for name in tables.flatten() {
                    let _ = conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\"", name));
                }
            }
        }

        conn.execute_batch("CHECKPOINT")?;
        self.conn = None;

        // A previous crashed compact can leave .old / .compact that would
        // make Windows rename(live→.old) fail. Live is the good copy here.
        if !Self::is_missing_or_empty(&db_path) {
            Self::remove_db_file_best_effort(&old_path);
            Self::remove_db_file_best_effort(&compact_path);
        }

        // --- Phase 2: Copy data via ATTACH + INSERT SELECT -----------------
        Self::write_intent(&intent_path, "pre-swap")?;
        Self::rename_db_with_wal(&db_path, &old_path)?;
        Self::write_intent(&intent_path, "phase1")?;

        // Create fresh DB and attach the old DB as 'src'.
        let import_conn = Connection::open(&compact_path)?;
        let attach_sql = format!(
            "ATTACH '{}' AS src",
            old_path.to_string_lossy().replace('\'', "''")
        );
        log::info!("compaction: {}", attach_sql);
        import_conn.execute_batch(&attach_sql)?;

        // --- Compute MAX(id) from source for sequence seeding ---
        let max_file_id: i64 =
            import_conn.query_row("SELECT COALESCE(MAX(id), 0) FROM src.files", [], |row| {
                row.get(0)
            })?;
        let max_chunk_id: i64 =
            import_conn.query_row("SELECT COALESCE(MAX(id), 0) FROM src.chunks", [], |row| {
                row.get(0)
            })?;

        // Discover embedding tables in the source catalog.
        let emb_tables: Vec<String> = import_conn
            .prepare(
                "SELECT table_name FROM information_schema.tables \
                 WHERE table_catalog = 'src' \
                 AND table_name SIMILAR TO 'embeddings_[0-9]+'",
            )?
            .query_map([], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();

        let mut max_embedding_id: i64 = 0;
        for tname in &emb_tables {
            let table_max: i64 = import_conn.query_row(
                &format!("SELECT COALESCE(MAX(id), 0) FROM src.\"{}\"", tname),
                [],
                |row| row.get(0),
            )?;
            max_embedding_id = max_embedding_id.max(table_max);
        }

        // --- Create sequences + tables in the fresh DB ---
        import_conn.execute_batch(&format!(
            "CREATE SEQUENCE files_id_seq START {}",
            max_file_id + 1
        ))?;
        import_conn.execute_batch(&format!(
            "CREATE SEQUENCE chunks_id_seq START {}",
            max_chunk_id + 1
        ))?;
        import_conn.execute_batch(&format!(
            "CREATE SEQUENCE embeddings_id_seq START {}",
            max_embedding_id + 1
        ))?;

        // Shares column DDL with setup_schema() / ensure_embedding_table_dims()
        // via the *_COLUMNS_DDL constants and embedding_columns_ddl(), so the
        // two can't drift apart.
        import_conn.execute_batch(&format!("CREATE TABLE files ({})", Self::FILES_COLUMNS_DDL))?;
        import_conn.execute_batch(&format!(
            "CREATE TABLE chunks ({})",
            Self::CHUNKS_COLUMNS_DDL
        ))?;
        import_conn.execute_batch(&format!(
            "CREATE TABLE schema_version ({})",
            Self::SCHEMA_VERSION_COLUMNS_DDL
        ))?;
        import_conn.execute_batch("INSERT INTO schema_version SELECT * FROM src.schema_version")?;

        // --- Copy data: files, chunks ---
        let col = "id, path, name, extension, size, modified_time, \
                    content_hash, language, skip_reason, created_at, updated_at";
        import_conn.execute_batch(&format!(
            "INSERT INTO files ({col}) SELECT {col} FROM src.files"
        ))?;

        let col = "id, file_id, chunk_type, symbol, code, start_line, \
                    end_line, start_byte, end_byte, language, metadata, \
                    created_at, updated_at";
        import_conn.execute_batch(&format!(
            "INSERT INTO chunks ({col}) SELECT {col} FROM src.chunks"
        ))?;

        // --- Copy embedding tables ---
        for tname in &emb_tables {
            let dims: u32 = tname
                .strip_prefix("embeddings_")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            import_conn.execute_batch(&format!(
                "CREATE TABLE \"{tname}\" ({})",
                Self::embedding_columns_ddl(dims)
            ))?;
            // Explicit column list (like files/chunks above), not `SELECT *`:
            // a positional copy would silently misalign or fail if the source
            // table's column shape ever differs from the freshly created one.
            let emb_col = "id, chunk_id, provider, model, embedding, dims, created_at";
            import_conn.execute_batch(&format!(
                "INSERT INTO \"{tname}\" ({emb_col}) SELECT {emb_col} FROM src.\"{tname}\""
            ))?;
            // Restore the canonical index set (chunk_id / provider_model /
            // unique upsert-contract) that the bare CREATE TABLE above
            // doesn't include. Without this, Python's post-reconnect
            // _executor_ensure_embedding_upsert_contract finds the unique
            // index missing and repairs it itself — which drops and rebuilds
            // the HNSW index a second time (this same reopen() already builds
            // it via ensure_all_hnsw_indexes()), an expensive redundant full
            // index rebuild on every compaction.
            Self::ensure_embedding_table_dims(&import_conn, dims)?;
        }

        // DETACH and CHECKPOINT.
        import_conn.execute_batch("DETACH src")?;
        import_conn.execute_batch("CHECKPOINT")?;
        drop(import_conn);

        // --- Phase 3: Atomic rename to active path --------------------------
        Self::write_intent(&intent_path, "phase2")?;
        Self::rename_db_with_wal(&compact_path, &db_path)?;

        // Clean up.
        let _ = std::fs::remove_file(&intent_path);
        Self::remove_db_file_best_effort(&old_path);

        // Reopen — also handles HNSW index creation via ensure_all_hnsw_indexes().
        self.reopen()?;
        log::info!("compaction: complete");
        Ok(())
    }

    /// Reopen the connection after a successful compaction.
    fn reopen(&mut self) -> Result<(), DbError> {
        self.known_dims.clear();
        let conn = Connection::open(&self.config.db_path)?;
        // VSS must be loaded unconditionally on reopen — compaction may have
        // imported HNSW index definitions from EXPORT/IMPORT, and DuckDB won't
        // serialize them during CHECKPOINT without VSS loaded.  The `has_vss`
        // flag is stale after the connection was closed and reopened.
        self.has_vss = Self::try_load_vss(&conn);
        Self::setup_schema(&conn)?;
        let existing = Self::discover_embedding_tables(&conn)?;
        self.known_dims
            .extend(existing.into_iter().map(|(_, dims)| dims));
        self.conn = Some(conn);
        // Recreate HNSW indexes on existing embedding tables.
        self.ensure_all_hnsw_indexes()?;
        Ok(())
    }

    /// Recover after a failed compaction attempt: reopen connection and
    /// restore state so the caller can continue or fall back to CHECKPOINT.
    pub(super) fn reopen_after_compaction_failure(&mut self) -> Result<(), DbError> {
        let db_path = PathBuf::from(&self.config.db_path);
        let recovered = Self::recover_swap_intent(&db_path)?;
        Self::discard_incomplete_compact_if_phase1(recovered, &db_path);
        let export_dir = PathBuf::from(format!("{}.export_tmp", self.config.db_path));
        let _ = std::fs::remove_dir_all(&export_dir);

        self.reopen()
    }

    /// Two-signal fragmentation detection (Phase 0).
    ///
    /// `free_ratio`: fraction of DB blocks that are free (freed by CHECKPOINT,
    /// not reused). Queried from DuckDB's `pragma_database_size`.
    ///
    /// `row_waste_ratio`: fraction of rows in storage that are dead (deleted but
    /// still occupying space in row groups). Estimated by comparing total stored
    /// row counts from `pragma_storage_info` against live `COUNT(*)` from our
    /// tables.
    ///
    /// Both signals fall back to zero when the DB is empty or pragmas are
    /// unavailable (safe default: no compaction needed).
    fn compaction_stats(&self) -> Result<CompactionStats, DbError> {
        let conn = self.conn_or_err()?;

        // --- free_ratio: from pragma_database_size -----------------------------
        let free_ratio: f64 = conn
            .query_row(
                "SELECT CASE WHEN total_blocks > 0 THEN free_blocks::DOUBLE \
                           / total_blocks ELSE 0.0 END FROM pragma_database_size()",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0.0);

        // --- row_waste_ratio: stored vs live rows across our tables ------------
        let live_chunks: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .unwrap_or(0);

        let live_embeddings: i64 = {
            let mut total = 0i64;
            for &dims in &self.known_dims {
                let table_name = format!("embeddings_{dims}");
                if let Ok(cnt) =
                    conn.query_row(&format!("SELECT COUNT(*) FROM \"{table_name}\""), [], |r| {
                        r.get::<_, i64>(0)
                    })
                {
                    total += cnt;
                }
            }
            total
        };

        // MAX(count) per row_group avoids counting the same row N times (once
        // per column segment).
        let stored_chunks: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(cnt), 0) FROM (\
                   SELECT row_group_id, MAX(count) AS cnt \
                   FROM pragma_storage_info('chunks') \
                   GROUP BY row_group_id)",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);

        let stored_embeddings: i64 = {
            let mut total = 0i64;
            for &dims in &self.known_dims {
                let table_name = format!("embeddings_{dims}");
                let safe = table_name.replace('"', "\"\"");
                if let Ok(cnt) = conn.query_row(
                    &format!(
                        "SELECT COALESCE(SUM(cnt), 0) FROM (\
                           SELECT row_group_id, MAX(count) AS cnt \
                           FROM pragma_storage_info('\"{safe}\"') \
                           GROUP BY row_group_id)"
                    ),
                    [],
                    |r| r.get::<_, i64>(0),
                ) {
                    total += cnt;
                }
            }
            total
        };

        let total_stored = stored_chunks + stored_embeddings;
        let total_live = live_chunks + live_embeddings;
        let row_waste_ratio = if total_stored > total_live && total_stored > 0 {
            (total_stored - total_live) as f64 / total_stored as f64
        } else {
            0.0
        };

        // --- reclaimable bytes -----------------------------------------------
        let db_size = std::fs::metadata(&self.config.db_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let reclaimable = (db_size as f64 * free_ratio.max(row_waste_ratio)) as u64;

        Ok(CompactionStats {
            free_ratio,
            row_waste_ratio,
            reclaimable,
        })
    }
}

pub(super) fn needs_compaction(backend: &DuckDbHnswBackend) -> Result<bool, DbError> {
    // None means auto-compaction is explicitly disabled.
    let Some(threshold) = backend.config.compaction_threshold else {
        return Ok(false);
    };
    // Two-signal metric-based detection (Phase 0). If stats are unavailable
    // (e.g. DB not open), there's nothing to compact yet.
    let Ok(stats) = backend.compaction_stats() else {
        return Ok(false);
    };
    let effective = stats.free_ratio.max(stats.row_waste_ratio);
    Ok(effective >= threshold && stats.reclaimable >= backend.config.compaction_min_size_bytes)
}

pub(super) fn run_compaction(backend: &mut DuckDbHnswBackend) -> Result<(), DbError> {
    // 3-phase atomic EXPORT/IMPORT compaction (Phase 0).
    // Falls back to CHECKPOINT-only if EXPORT/IMPORT is unavailable
    // (e.g. DuckDB build without Parquet support).
    if let Err(e) = backend.run_attach_copy_compaction() {
        log::warn!(
            "compaction: EXPORT/IMPORT failed ({}), falling back to CHECKPOINT",
            e
        );
        backend.reopen_after_compaction_failure()?;
        let conn = backend.conn_or_err()?;
        conn.execute_batch("CHECKPOINT")?;
    }
    Ok(())
}

#[cfg(test)]
mod compaction_threshold_tests {
    use super::*;

    #[test]
    fn none_threshold_disables_compaction_without_opening_db() {
        // Mirrors Python's fragmentation_threshold_pct=None ("never
        // auto-compact") opt-out. Must short-circuit before touching the
        // DB connection at all, so this works even on a never-`.open()`ed
        // backend (needs_compaction() is polled opportunistically and must
        // not itself force a connection).
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let backend = DuckDbHnswBackend::new(
            super::super::test_support::config_with_compaction_threshold(db_path, None),
        );
        assert!(!backend.needs_compaction().expect("needs_compaction"));
    }

    #[test]
    fn some_threshold_falls_through_to_metric_check() {
        // With a real DB open and no fragmentation yet, a configured
        // threshold must not itself force compaction — this pins the
        // "Some(threshold) still requires exceeding it" half of the branch.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend = DuckDbHnswBackend::new(
            super::super::test_support::config_with_compaction_threshold(db_path, Some(0.30)),
        );
        backend.open().expect("open");
        assert!(!backend.needs_compaction().expect("needs_compaction"));
    }
}

#[cfg(test)]
mod compaction_index_restore_tests {
    use super::*;
    use crate::db::DbBackend;

    #[test]
    fn compaction_restores_embedding_table_indexes() {
        // Regression test: run_attach_copy_compaction() rebuilds each
        // embedding table via a bare CREATE TABLE — it must also restore the
        // chunk_id, provider_model, and unique upsert-contract indexes
        // (normally created by ensure_embedding_table_dims() on first table
        // creation). Without this, Python's post-reconnect
        // _executor_ensure_embedding_upsert_contract finds the unique index
        // missing and repairs it itself, which drops and rebuilds the HNSW
        // index a second time — an expensive redundant full index rebuild on
        // every compaction.
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
            .write_batch(&super::super::test_support::embedding_batch("a", 8, 10))
            .expect("write");

        backend
            .run_attach_copy_compaction()
            .expect("compaction should succeed");

        let conn = backend.conn_or_err().expect("conn");

        // Regression guard for the explicit-column INSERT (replacing a
        // positional `SELECT *`): every row's `dims` must still read back as
        // 8, proving the copy landed each column in the right place rather
        // than shifting values across a mismatched positional layout.
        let (row_count, dims_ok): (i64, i64) = conn
            .query_row(
                "SELECT COUNT(*), COUNT(*) FILTER (WHERE dims = 8) FROM embeddings_8",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("query embeddings_8");
        assert!(
            row_count > 0,
            "embeddings_8 must not be empty after compaction"
        );
        assert_eq!(
            row_count, dims_ok,
            "every copied embedding row must have dims = 8"
        );

        for index_name in [
            "idx_8_chunk_id",
            "idx_8_provider_model",
            "idx_8_chunk_provider_model_unique",
        ] {
            let exists: bool = conn
                .query_row(
                    "SELECT COUNT(*) > 0 FROM duckdb_indexes() \
                     WHERE table_name = 'embeddings_8' AND index_name = ?",
                    [index_name],
                    |r| r.get(0),
                )
                .expect("query duckdb_indexes");
            assert!(
                exists,
                "{index_name} must exist on embeddings_8 after compaction"
            );
        }
    }
}
