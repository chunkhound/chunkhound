use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use duckdb::Connection;

use crate::db::{DbBackend, DbConfig};
use crate::error::DbError;
use crate::types::{BatchResult, ChunkRecord, DbWriterBatch, FileRecord};

pub struct DuckDbHnswBackend {
    config: DbConfig,
    conn: Option<Connection>,
    has_vss: bool,
    hnsw_bulk_mode: bool,
    // Dims for which embeddings_N tables are known to exist in this session.
    known_dims: HashSet<u32>,
    // Metrics (e.g. "cosine", "l2sq") for each dims value, captured by
    // drop_all_hnsw_indexes() before bulk-mode drop so that ensure_all_hnsw_indexes()
    // can recreate indexes with the original metric instead of hardcoding cosine.
    saved_hnsw_metrics: HashMap<u32, String>,
}

#[derive(Debug, Clone)]
struct HnswIndexInfo {
    index_name: String,
    table_name: String,
    metric: String,
}

struct BatchInner {
    file_ids: Vec<i64>,
    chunks_written: u64,
    embedding_pairs: Vec<(i64, usize, usize)>,
}

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
    /// Batch size for path-based DELETE operations (delete_paths, pre_delete_for_upsert).
    /// 500 paths per batch balances SQL round-trip overhead vs memory usage for IN-list
    /// parameter binding.
    const DELETE_BATCH: usize = 500;

    pub fn new(config: DbConfig) -> Self {
        DuckDbHnswBackend {
            config,
            conn: None,
            has_vss: false,
            hnsw_bulk_mode: false,
            known_dims: HashSet::new(),
            saved_hnsw_metrics: HashMap::new(),
        }
    }

    fn conn_or_err(&self) -> Result<&Connection, DbError> {
        self.conn
            .as_ref()
            .ok_or_else(|| DbError::Other("not open".into()))
    }

    // SCHEMA PARITY: This DDL must stay in sync with the Python canonical source at
    // chunkhound/providers/database/duckdb/schema_constants.py (_FILES_TABLE_COLUMNS,
    // _CHUNKS_TABLE_COLUMNS, _SCHEMA_VERSION_TABLE_COLUMNS).  The cross-check test
    // tests/contracts/test_schema_parity.py::TestSchemaParity catches column-level drift at CI time.
    // When adding or renaming columns, update schema_constants.py FIRST, then mirror here.
    fn setup_schema(conn: &Connection) -> Result<(), DbError> {
        conn.execute_batch(
            "
            CREATE SEQUENCE IF NOT EXISTS files_id_seq START 1;
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),
                path TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                extension TEXT,
                size INTEGER,
                modified_time TIMESTAMP,
                content_hash TEXT,
                language TEXT,
                skip_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE SEQUENCE IF NOT EXISTS chunks_id_seq START 1;
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),
                file_id INTEGER REFERENCES files(id),
                chunk_type TEXT NOT NULL,
                symbol TEXT,
                code TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                language TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE SEQUENCE IF NOT EXISTS embeddings_id_seq START 1;
            CREATE INDEX IF NOT EXISTS idx_files_path ON files(path);
            CREATE INDEX IF NOT EXISTS idx_files_language ON files(language);
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol);
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            );
            INSERT INTO schema_version (version, description)
                SELECT 1, 'Initial schema'
                WHERE NOT EXISTS (SELECT 1 FROM schema_version WHERE version = 1);
        ",
        )?;
        Ok(())
    }

    fn try_load_vss(conn: &Connection) -> bool {
        // INSTALL may fail in air-gapped environments or when the extension is already present
        // at a different version — that is non-fatal.  Mirror Python's connection_manager.py
        // which uses separate execute() calls so that LOAD is always attempted regardless of
        // INSTALL's outcome.
        let _ = conn.execute_batch("INSTALL vss");
        if let Err(e) = conn.execute_batch("LOAD vss") {
            log::warn!("VSS extension unavailable (vector search disabled): {e}");
            return false;
        }
        if let Err(e) = conn.execute_batch("SET hnsw_enable_experimental_persistence = true") {
            log::warn!(
                "HNSW persistence unavailable \
                 (hnsw_enable_experimental_persistence not supported by this DuckDB build): {e}"
            );
            return false;
        }
        true
    }

    /// Ensure VSS is loaded, lazily — only on first write with embeddings.
    fn ensure_vss(&mut self) -> Result<bool, DbError> {
        if self.has_vss {
            return Ok(true);
        }
        let conn = self.conn_or_err()?;
        let ok = Self::try_load_vss(conn);
        self.has_vss = ok;
        Ok(ok)
    }

    fn ensure_embedding_table_dims(conn: &Connection, dims: u32) -> Result<(), DbError> {
        let table = format!("embeddings_{dims}");
        // Legacy compat: the 1536-dimension chunk_id index was originally named
        // idx_embeddings_1536_chunk_id; newer dimensions use idx_{dims}_chunk_id.
        // Mirrors schema_constants._embedding_chunk_id_index_name in Python.
        let chunk_id_idx = if dims == 1536 {
            format!("idx_embeddings_{dims}_chunk_id")
        } else {
            format!("idx_{dims}_chunk_id")
        };
        conn.execute_batch(&format!(
            "
            CREATE TABLE IF NOT EXISTS \"{table}\" (
                id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),
                chunk_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                embedding FLOAT[{dims}],
                dims INTEGER NOT NULL DEFAULT {dims},
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_{dims}_chunk_provider_model_unique
            ON \"{table}\" (chunk_id, provider, model);
            CREATE INDEX IF NOT EXISTS {chunk_id_idx}
            ON \"{table}\" (chunk_id);
            CREATE INDEX IF NOT EXISTS idx_{dims}_provider_model
            ON \"{table}\" (provider, model);
        "
        ))?;
        Ok(())
    }

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

    fn collect_dims_and_count(batch: &DbWriterBatch) -> (HashSet<u32>, usize) {
        let mut dims = HashSet::new();
        let mut count = 0usize;
        for file in &batch.files {
            for chunk in &file.chunks {
                if let Some(e) = &chunk.embedding {
                    if !e.is_empty() {
                        dims.insert(e.len() as u32);
                        count += 1;
                    }
                }
            }
        }
        (dims, count)
    }

    fn upsert_file(conn: &Connection, file: &FileRecord) -> Result<i64, DbError> {
        let path = Path::new(&file.path);
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(file.path.as_str())
            .to_string();
        let ext: Option<String> = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_string());

        // Fast path: the diff phase already knows this file's DB id (incremental re-index).
        // Skip the SELECT and go straight to UPDATE.
        if let Some(id) = file.existing_file_id {
            conn.execute(
                "UPDATE files SET size = ?, modified_time = CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, content_hash = ?, language = ?, updated_at = now() WHERE id = ?",
                duckdb::params![file.size_bytes, file.mtime, file.mtime, file.content_hash, file.language, id],
            )?;
            return Ok(id);
        }

        // Slow path (new files or non-incremental runs): DuckDB rejects ON CONFLICT DO UPDATE
        // inside an explicit transaction when a FK child table (chunks) has rows referencing
        // the conflicting parent row, even if those children were deleted earlier in the same
        // transaction. Work around by doing an explicit SELECT then UPDATE-or-INSERT.
        let existing_id: Option<i64> = conn
            .query_row("SELECT id FROM files WHERE path = ?", [&file.path], |r| {
                r.get(0)
            })
            .ok();

        if let Some(id) = existing_id {
            conn.execute(
                "UPDATE files SET size = ?, modified_time = CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, content_hash = ?, language = ?, updated_at = now() WHERE id = ?",
                duckdb::params![file.size_bytes, file.mtime, file.mtime, file.content_hash, file.language, id],
            )?;
            Ok(id)
        } else {
            let id: i64 = conn.query_row(
                "INSERT INTO files (path, name, extension, size, modified_time, content_hash, language)
                 VALUES (?, ?, ?, ?, CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, ?, ?)
                 RETURNING id",
                duckdb::params![
                    file.path,
                    name,
                    ext,
                    file.size_bytes,
                    file.mtime,
                    file.mtime,
                    file.content_hash,
                    file.language,
                ],
                |row| row.get(0),
            )?;
            Ok(id)
        }
    }

    fn insert_chunks_for_file(
        conn: &Connection,
        file_id: i64,
        chunks: &[ChunkRecord],
        insert_batch_size: usize,
    ) -> Result<Vec<i64>, DbError> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        // Insert directly into chunks with RETURNING id, batched to cut round-trips.
        // Avoids CREATE/DROP TEMPORARY TABLE DDL so this function is safe to call
        // inside an open transaction (DDL would cause implicit commits in some
        // DuckDB versions).
        let mut ids: Vec<i64> = Vec::with_capacity(chunks.len());

        for chunk_slice in chunks.chunks(insert_batch_size.max(1)) {
            let row_ph = std::iter::repeat_n("(?,?,?,?,?,?,?,?,?,?)", chunk_slice.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "INSERT INTO chunks \
                 (file_id, chunk_type, symbol, code, start_line, end_line, \
                  start_byte, end_byte, language, metadata) VALUES {row_ph} RETURNING id"
            );
            let mut params: Vec<duckdb::types::Value> = Vec::with_capacity(chunk_slice.len() * 10);
            for chunk in chunk_slice {
                params.push(duckdb::types::Value::BigInt(file_id));
                params.push(duckdb::types::Value::Text(chunk.chunk_type.clone()));
                params.push(
                    chunk
                        .symbol
                        .as_deref()
                        .map_or(duckdb::types::Value::Null, |s| {
                            duckdb::types::Value::Text(s.to_string())
                        }),
                );
                params.push(duckdb::types::Value::Text(chunk.code.clone()));
                params.push(
                    chunk
                        .start_line
                        .map_or(duckdb::types::Value::Null, duckdb::types::Value::BigInt),
                );
                params.push(
                    chunk
                        .end_line
                        .map_or(duckdb::types::Value::Null, duckdb::types::Value::BigInt),
                );
                params.push(
                    chunk
                        .start_byte
                        .map_or(duckdb::types::Value::Null, duckdb::types::Value::BigInt),
                );
                params.push(
                    chunk
                        .end_byte
                        .map_or(duckdb::types::Value::Null, duckdb::types::Value::BigInt),
                );
                params.push(
                    chunk
                        .language
                        .as_deref()
                        .map_or(duckdb::types::Value::Null, |s| {
                            duckdb::types::Value::Text(s.to_string())
                        }),
                );
                params.push(
                    chunk
                        .metadata
                        .as_deref()
                        .map_or(duckdb::types::Value::Null, |s| {
                            duckdb::types::Value::Text(s.to_string())
                        }),
                );
            }
            let mut stmt = conn.prepare(&sql)?;
            let batch_ids: Vec<i64> = stmt
                .query_map(duckdb::params_from_iter(params), |row| row.get(0))?
                .collect::<Result<Vec<i64>, _>>()
                .map_err(DbError::DuckDb)?;
            ids.extend(batch_ids);
        }

        Ok(ids)
    }

    fn insert_embeddings_txn(
        conn: &Connection,
        batch: &DbWriterBatch,
        embedding_pairs: &[(i64, usize, usize)], // (chunk_id, file_idx, chunk_idx)
        insert_batch_size: usize,
    ) -> Result<u64, DbError> {
        if embedding_pairs.is_empty() {
            return Ok(0);
        }

        // Group by dims
        let mut by_dims: HashMap<u32, Vec<(i64, &ChunkRecord)>> = HashMap::new();
        for &(chunk_id, file_idx, chunk_idx) in embedding_pairs {
            let chunk = &batch.files[file_idx].chunks[chunk_idx];
            if let Some(emb) = &chunk.embedding {
                if !emb.is_empty() {
                    by_dims
                        .entry(emb.len() as u32)
                        .or_default()
                        .push((chunk_id, chunk));
                }
            }
        }

        // Insert directly into embeddings_N, batched to cut round-trips.
        // Avoids CREATE/DROP TEMPORARY TABLE DDL so this function is safe to call
        // inside an open transaction (DDL would cause implicit commits in some
        // DuckDB versions).
        let insert_batch_size = insert_batch_size.max(1);
        let mut total = 0u64;
        for (dims, items) in &by_dims {
            let table = format!("embeddings_{dims}");

            for chunk_slice in items.chunks(insert_batch_size) {
                let row_ph = std::iter::repeat_n("(?,?,?,?::FLOAT[{dims}],?)", chunk_slice.len())
                    .collect::<Vec<_>>()
                    .join(",")
                    .replace("{dims}", &dims.to_string());
                let sql = format!(
                    "INSERT INTO \"{table}\" (chunk_id, provider, model, embedding, dims) \
                     VALUES {row_ph} \
                     ON CONFLICT (chunk_id, provider, model) DO UPDATE \
                     SET embedding = EXCLUDED.embedding, dims = EXCLUDED.dims"
                );
                let mut params: Vec<duckdb::types::Value> =
                    Vec::with_capacity(chunk_slice.len() * 5);
                for (chunk_id, chunk) in chunk_slice.iter() {
                    let emb = chunk.embedding.as_ref().expect(
                        "embedding is Some: only chunks with Some(emb) are in embedding_pairs",
                    );
                    let emb_json = serde_json::to_string(emb).map_err(DbError::Json)?;
                    params.push(duckdb::types::Value::BigInt(*chunk_id));
                    params.push(duckdb::types::Value::Text(
                        chunk.provider.as_deref().unwrap_or("unknown").to_string(),
                    ));
                    params.push(duckdb::types::Value::Text(
                        chunk.model.as_deref().unwrap_or("unknown").to_string(),
                    ));
                    params.push(duckdb::types::Value::Text(emb_json));
                    params.push(duckdb::types::Value::BigInt(*dims as i64));
                }
                let rows = conn.execute(&sql, duckdb::params_from_iter(params))?;
                total += rows as u64;
            }
        }
        Ok(total)
    }

    // delete_paths (explicit path removals from the caller) run outside the transaction
    // to avoid the DuckDB limitation where ON CONFLICT DO UPDATE on a FK parent row
    // is rejected inside an explicit transaction even after child rows are deleted.
    // The pre-deletes inside write_batch_inner work because upsert_file uses an explicit
    // SELECT + UPDATE/INSERT rather than ON CONFLICT DO UPDATE syntax.
    fn delete_paths(
        conn: &Connection,
        paths: &[String],
        known_dims: &HashSet<u32>,
    ) -> Result<(), DbError> {
        if paths.is_empty() {
            return Ok(());
        }
        let emb_tables: Vec<(String, u32)> = known_dims
            .iter()
            .map(|&dims| (format!("embeddings_{dims}"), dims))
            .collect();

        // Phase 1: atomically delete embeddings + chunks together.
        // embeddings_N tables have no FK to chunks — delete embeddings first
        // to avoid ghost rows accumulating on re-index (CF-1).
        // Wrapping in a transaction ensures emb and chunk deletes are atomic
        // with each other (no ghost emb rows if the process crashes mid-batch).
        conn.execute_batch("BEGIN")?;
        let result = (|| -> Result<(), DbError> {
            for batch in paths.chunks(Self::DELETE_BATCH) {
                let ph = std::iter::repeat_n("?", batch.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let params: Vec<duckdb::types::Value> = batch
                    .iter()
                    .map(|p| duckdb::types::Value::Text(p.clone()))
                    .collect();
                let chunk_subquery = format!(
                    "SELECT id FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path IN ({ph}))"
                );
                for (table_name, _dims) in &emb_tables {
                    conn.execute(
                        &format!(
                            "DELETE FROM \"{table_name}\" WHERE chunk_id IN ({chunk_subquery})"
                        ),
                        duckdb::params_from_iter(params.clone()),
                    )?;
                }
                conn.execute(
                    &format!("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path IN ({ph}))"),
                    duckdb::params_from_iter(params),
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => conn.execute_batch("COMMIT").map_err(DbError::DuckDb)?,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        }

        // Phase 2: delete parent rows (files) in auto-commit mode.
        // Must be separate from Phase 1: DuckDB's FK check engine reads the committed
        // DB state, not the current transaction's in-progress deletes.  If Phase 1's
        // chunk deletes were in the same transaction as the files delete, the engine
        // would still see the (not-yet-committed) chunks referencing the file and
        // reject the DELETE with a FK constraint error.
        for batch in paths.chunks(Self::DELETE_BATCH) {
            let ph = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(",");
            let params: Vec<duckdb::types::Value> = batch
                .iter()
                .map(|p| duckdb::types::Value::Text(p.clone()))
                .collect();
            conn.execute(
                &format!("DELETE FROM files WHERE path IN ({ph})"),
                duckdb::params_from_iter(params),
            )?;
        }
        Ok(())
    }

    fn discover_embedding_tables(conn: &Connection) -> Result<Vec<(String, u32)>, DbError> {
        let mut stmt = conn.prepare(
            "SELECT table_name FROM information_schema.tables
             WHERE table_schema = 'main'
             AND table_name SIMILAR TO 'embeddings_[0-9]+'",
        )?;
        let names: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        let result = names
            .into_iter()
            .filter_map(|name| {
                let dims: u32 = name.strip_prefix("embeddings_")?.parse().ok()?;
                Some((name, dims))
            })
            .collect();
        Ok(result)
    }

    // Pre-deletes chunks (and orphaned embeddings) for files about to be upserted.
    // Must run OUTSIDE the write transaction: DuckDB's FK check engine sees the committed
    // state of the DB, not the current transaction's state. Any UPDATE on files inside a
    // transaction where child chunks were deleted earlier in the same transaction is rejected
    // with a FK constraint error — even though no FK is actually violated at commit time.
    // The same limitation affects delete_paths; both are handled identically (pre-txn commit).
    //
    // Atomicity gap — two cases:
    //
    // (a) Upsert files: only chunks/embeddings are deleted here; the file row survives.
    //     If the process crashes after this COMMIT but before the write transaction below,
    //     the file still exists in the files table with stale metadata.  Recovery is
    //     self-healing: on the next index run the file is found in the DB and re-indexed
    //     from disk — no data is permanently lost.
    //
    // (b) delete_paths (handled in Step 0a): files ARE removed from the DB.  If the process
    //     crashes after delete_paths commits but before the write transaction below commits,
    //     those files are absent from the DB and will not be re-populated unless the caller
    //     explicitly re-requests them.  This is an inherent limitation of the two-phase
    //     commit approach — the caller must be prepared to re-submit deletes after a crash.
    fn pre_delete_for_upsert(
        conn: &Connection,
        batch: &DbWriterBatch,
        known_dims: &HashSet<u32>,
    ) -> Result<(), DbError> {
        let by_id: Vec<i64> = batch
            .files
            .iter()
            .filter_map(|f| f.existing_file_id)
            .collect();
        let by_path: Vec<String> = batch
            .files
            .iter()
            .filter(|f| f.existing_file_id.is_none())
            .map(|f| f.path.clone())
            .collect();

        if by_id.is_empty() && by_path.is_empty() {
            return Ok(());
        }

        let emb_tables: Vec<(String, u32)> = known_dims
            .iter()
            .map(|&dims| (format!("embeddings_{dims}"), dims))
            .collect();
        conn.execute_batch("BEGIN")?;
        let result = (|| -> Result<(), DbError> {
            if !by_id.is_empty() {
                let ph = std::iter::repeat_n("?", by_id.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let params: Vec<duckdb::types::Value> = by_id
                    .iter()
                    .map(|&id| duckdb::types::Value::BigInt(id))
                    .collect();
                for (table_name, _dims) in &emb_tables {
                    conn.execute(
                        &format!(
                            "DELETE FROM \"{table_name}\" WHERE chunk_id IN \
                             (SELECT id FROM chunks WHERE file_id IN ({ph}))"
                        ),
                        duckdb::params_from_iter(params.clone()),
                    )?;
                }
                conn.execute(
                    &format!("DELETE FROM chunks WHERE file_id IN ({ph})"),
                    duckdb::params_from_iter(params),
                )?;
            }
            if !by_path.is_empty() {
                let ph = std::iter::repeat_n("?", by_path.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let params: Vec<duckdb::types::Value> = by_path
                    .iter()
                    .map(|p| duckdb::types::Value::Text(p.clone()))
                    .collect();
                let chunk_subquery = format!(
                    "SELECT id FROM chunks WHERE file_id IN \
                     (SELECT id FROM files WHERE path IN ({ph}))"
                );
                for (table_name, _dims) in &emb_tables {
                    conn.execute(
                        &format!(
                            "DELETE FROM \"{table_name}\" WHERE chunk_id IN ({chunk_subquery})"
                        ),
                        duckdb::params_from_iter(params.clone()),
                    )?;
                }
                conn.execute(
                    &format!(
                        "DELETE FROM chunks WHERE file_id IN \
                         (SELECT id FROM files WHERE path IN ({ph}))"
                    ),
                    duckdb::params_from_iter(params),
                )?;
            }
            Ok(())
        })();
        match result {
            Ok(()) => conn.execute_batch("COMMIT").map_err(DbError::DuckDb),
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Write a compaction-intent marker file and fsync it to disk.
    /// Used by the 3-phase swap protocol to enable crash recovery (Invariant 17).
    fn write_intent(path: &Path, phase: &str) -> Result<(), DbError> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        f.write_all(phase.as_bytes())?;
        f.sync_all()?;
        Ok(())
    }

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

        // --- Phase 2: Copy data via ATTACH + INSERT SELECT -----------------
        Self::write_intent(&intent_path, "pre-swap")?;
        std::fs::rename(&db_path, &old_path)?;
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

        // Mirror the DDL from setup_schema().
        import_conn.execute_batch(
            "CREATE TABLE files (\
                id INTEGER PRIMARY KEY DEFAULT nextval('files_id_seq'),\
                path TEXT UNIQUE NOT NULL,\
                name TEXT NOT NULL,\
                extension TEXT,\
                size INTEGER,\
                modified_time TIMESTAMP,\
                content_hash TEXT,\
                language TEXT,\
                skip_reason TEXT,\
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\
            )",
        )?;
        import_conn.execute_batch(
            "CREATE TABLE chunks (\
                id INTEGER PRIMARY KEY DEFAULT nextval('chunks_id_seq'),\
                file_id INTEGER REFERENCES files(id),\
                chunk_type TEXT NOT NULL,\
                symbol TEXT,\
                code TEXT NOT NULL,\
                start_line INTEGER,\
                end_line INTEGER,\
                start_byte INTEGER,\
                end_byte INTEGER,\
                language TEXT,\
                metadata TEXT,\
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\
            )",
        )?;
        import_conn.execute_batch(
            "CREATE TABLE schema_version (\
                version INTEGER PRIMARY KEY,\
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,\
                description TEXT\
            )",
        )?;
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
                "CREATE TABLE \"{tname}\" (\
                    id INTEGER PRIMARY KEY DEFAULT nextval('embeddings_id_seq'),\
                    chunk_id INTEGER NOT NULL,\
                    provider TEXT NOT NULL,\
                    model TEXT NOT NULL,\
                    embedding FLOAT[{dims}],\
                    dims INTEGER NOT NULL DEFAULT {dims},\
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\
                )"
            ))?;
            import_conn.execute_batch(&format!(
                "INSERT INTO \"{tname}\" SELECT * FROM src.\"{tname}\""
            ))?;
        }

        // DETACH and CHECKPOINT.
        import_conn.execute_batch("DETACH src")?;
        import_conn.execute_batch("CHECKPOINT")?;
        drop(import_conn);

        // --- Phase 3: Atomic rename to active path --------------------------
        Self::write_intent(&intent_path, "phase2")?;
        std::fs::rename(&compact_path, &db_path)?;

        // Clean up.
        let _ = std::fs::remove_file(&intent_path);
        let _ = std::fs::remove_file(&old_path);

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
    fn reopen_after_compaction_failure(&mut self) -> Result<(), DbError> {
        // If the old DB was renamed away, try to restore it from intent.
        let db_path = PathBuf::from(&self.config.db_path);
        let intent_path = PathBuf::from(format!("{}.swap_intent", self.config.db_path));
        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
        if intent_path.exists() {
            if let Ok(intent) = std::fs::read_to_string(&intent_path) {
                match intent.trim() {
                    "pre-swap" => {
                        // Original DB was renamed to .old; restore it.
                        if old_path.exists() && !db_path.exists() {
                            let _ = std::fs::rename(&old_path, &db_path);
                        }
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    "phase1" | "phase2" => {
                        // Old DB already renamed; compact may or may not
                        // exist. Try to restore the original.
                        if old_path.exists() {
                            if db_path.exists() {
                                let _ = std::fs::remove_file(&db_path);
                            }
                            let _ = std::fs::rename(&old_path, &db_path);
                        }
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    _ => {}
                }
            }
        }
        // Clean up any compaction artifacts.
        let compact_path = PathBuf::from(format!("{}.compact", self.config.db_path));
        let export_dir = PathBuf::from(format!("{}.export_tmp", self.config.db_path));
        let _ = std::fs::remove_file(&compact_path);
        let _ = std::fs::remove_dir_all(&export_dir);

        // Reopen.
        self.reopen()
    }

    // Runs inside an already-open BEGIN/COMMIT envelope managed by the caller.
    // Handles file upserts and chunk inserts; returns intermediate state needed
    // for the embedding insert step that follows in the same transaction.
    // Pre-deletes for upserted files are handled by pre_delete_for_upsert (called
    // before BEGIN to avoid DuckDB's intra-transaction FK check limitation).
    fn write_batch_inner(
        conn: &Connection,
        batch: &DbWriterBatch,
        insert_batch_size: usize,
    ) -> Result<BatchInner, DbError> {
        // Upsert files → collect file_ids
        let mut file_ids = Vec::with_capacity(batch.files.len());
        for file in &batch.files {
            let fid = Self::upsert_file(conn, file)?;
            file_ids.push(fid);
        }

        // Insert chunks per file; collect (chunk_id, file_idx, chunk_idx) for embeddings
        let mut total_chunks = 0u64;
        let mut embedding_pairs: Vec<(i64, usize, usize)> = Vec::new();

        for (file_idx, (file, &file_id)) in batch.files.iter().zip(file_ids.iter()).enumerate() {
            let chunk_ids =
                Self::insert_chunks_for_file(conn, file_id, &file.chunks, insert_batch_size)?;
            total_chunks += chunk_ids.len() as u64;

            for (chunk_idx, chunk_id) in chunk_ids.into_iter().enumerate() {
                if file.chunks[chunk_idx]
                    .embedding
                    .as_ref()
                    .map(|e| !e.is_empty())
                    .unwrap_or(false)
                {
                    embedding_pairs.push((chunk_id, file_idx, chunk_idx));
                }
            }
        }

        Ok(BatchInner {
            file_ids,
            chunks_written: total_chunks,
            embedding_pairs,
        })
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

impl crate::db::DbBackend for DuckDbHnswBackend {
    fn open(&mut self) -> Result<(), DbError> {
        // Idempotent: already open on Windows (exclusive file lock) would error on re-open.
        if self.conn.is_some() {
            return Ok(());
        }
        // Crash recovery: check for swap_intent file (Invariant 17)
        let db_path = PathBuf::from(&self.config.db_path);
        let intent_path = PathBuf::from(format!("{}.swap_intent", self.config.db_path));
        if intent_path.exists() {
            if let Ok(intent) = std::fs::read_to_string(&intent_path) {
                match intent.trim() {
                    "pre-swap" => {
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    "phase1" => {
                        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
                        if old_path.exists() {
                            let _ = std::fs::rename(&old_path, &db_path);
                        }
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    "phase2" => {
                        // The final compact_path -> db_path rename may not
                        // have completed before the crash. If compact_path
                        // still exists, the rename never happened (or only
                        // partially did) — finish it before touching the
                        // pre-compaction backup, otherwise db_path is left
                        // missing and the next open() below would silently
                        // create an empty database.
                        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
                        let compact_path =
                            PathBuf::from(format!("{}.compact", self.config.db_path));
                        if compact_path.exists() {
                            if db_path.exists() {
                                let _ = std::fs::remove_file(&db_path);
                            }
                            let _ = std::fs::rename(&compact_path, &db_path);
                        }
                        let _ = std::fs::remove_file(&old_path);
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    _ => {}
                }
            }
        }

        self.known_dims.clear();
        let conn = Connection::open(&self.config.db_path)?;
        // Defer WAL auto-checkpoints. DuckDB checkpoints synchronously on
        // COMMIT once the WAL exceeds checkpoint_threshold, and each checkpoint
        // does work proportional to the whole DB file (measured ~30ms/MiB) —
        // NOT to the small WAL delta being flushed. At the default (~16MB) this
        // fires every ~2 batches and grows unbounded as the DB grows, which is
        // what dominates and monotonically degrades the store stage. Raising
        // the threshold collapses hundreds of ever-growing checkpoints into a
        // handful; close() issues the final CHECKPOINT to flush deferred WAL.
        // Env-tunable so the ceiling can be adjusted per-run without rebuilding.
        let checkpoint_threshold = std::env::var("CHUNKHOUND_DUCKDB_CHECKPOINT_THRESHOLD")
            .unwrap_or_else(|_| "8GB".to_string());
        if let Err(e) = conn.execute_batch(&format!(
            "SET checkpoint_threshold='{checkpoint_threshold}'"
        )) {
            log::warn!("failed to set checkpoint_threshold='{checkpoint_threshold}': {e}");
        }
        // VSS must be loaded on open — the DB on disk may already have
        // VSS catalog entries from a previous session, and DuckDB won't
        // deserialize them without VSS loaded.
        self.has_vss = Self::try_load_vss(&conn);
        Self::setup_schema(&conn)?;
        // Prime known_dims from tables that already exist so the first batch
        // with an existing dimension does not trigger a spurious cache invalidation.
        let existing = Self::discover_embedding_tables(&conn)?;
        self.known_dims
            .extend(existing.into_iter().map(|(_, dims)| dims));
        self.conn = Some(conn);
        // Crash recovery: if the process was killed between drop_all_hnsw_indexes()
        // and ensure_all_hnsw_indexes(), HNSW indexes are absent but the
        // embeddings_N tables still hold data.  Recreate any missing indexes now so
        // the next session doesn't silently fall back to brute-force vector scan.
        self.ensure_all_hnsw_indexes()?;
        Ok(())
    }

    fn close(&mut self) -> Result<(), DbError> {
        // Collect the first error encountered but always drop the connection so the
        // DB file is released even when cleanup steps fail.
        let mut result: Result<(), DbError> = Ok(());

        if self.hnsw_bulk_mode {
            if let Err(e) = self.ensure_all_hnsw_indexes() {
                result = Err(e);
            }
        }
        if let Some(conn) = self.conn.as_ref() {
            if let Err(e) = conn.execute_batch("CHECKPOINT") {
                if result.is_ok() {
                    result = Err(DbError::DuckDb(e));
                }
            }
        }
        self.conn = None;
        result
    }

    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        self.prepare_write(batch)?;
        self.write_batch_incremental(batch)
    }

    fn prepare_write(&mut self, batch: &DbWriterBatch) -> Result<(), DbError> {
        // Step 0a: Handle delete_paths OUTSIDE transaction.
        if !batch.delete_paths.is_empty() {
            let conn = self.conn_or_err()?;
            Self::delete_paths(conn, &batch.delete_paths, &self.known_dims)?;
        }

        // Step 0b: Pre-delete chunks/embeddings for files being upserted, OUTSIDE transaction.
        {
            let conn = self.conn_or_err()?;
            Self::pre_delete_for_upsert(conn, batch, &self.known_dims)?;
        }

        // Step 0c: Ensure embedding tables outside txn (Invariant 13).
        let (unique_dims, total_emb) = Self::collect_dims_and_count(batch);
        {
            let conn = self.conn_or_err()?;
            for &dims in &unique_dims {
                Self::ensure_embedding_table_dims(conn, dims)?;
            }
        }
        self.known_dims.extend(unique_dims.iter().copied());

        // Lazy VSS load — only when embeddings are actually present.
        if total_emb > 0 {
            self.ensure_vss()?;
        }

        Ok(())
    }

    fn write_batch_incremental(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        // BEGIN + write inner.
        let batch_inner = {
            let conn = self.conn_or_err()?;
            conn.execute_batch("BEGIN")?;

            match Self::write_batch_inner(conn, batch, self.config.insert_batch_size) {
                Ok(inner) => inner,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    return Err(e);
                }
            }
        };
        let (file_ids, chunks_written, embedding_pairs) = (
            batch_inner.file_ids,
            batch_inner.chunks_written,
            batch_inner.embedding_pairs,
        );

        // Insert embeddings (still inside txn).
        let embeddings_written = {
            let insert_batch_size = self.config.insert_batch_size;
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and BEGIN passed");
            match Self::insert_embeddings_txn(conn, batch, &embedding_pairs, insert_batch_size) {
                Ok(n) => n,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    return Err(e);
                }
            }
        };

        // COMMIT. DuckDB runs its automatic WAL checkpoint synchronously on
        // COMMIT once the WAL exceeds checkpoint_threshold, so this timing
        // isolates checkpoint cost from the inserts above — the key signal for
        // diagnosing whether write-stage slowdown is checkpoint-driven.
        {
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and BEGIN passed");
            let t_commit = Instant::now();
            if let Err(e) = conn.execute_batch("COMMIT") {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(DbError::DuckDb(e));
            }
            log::debug!(
                "[store]   commit+checkpoint {:.1}ms",
                t_commit.elapsed().as_secs_f64() * 1e3
            );
        }

        Ok(BatchResult {
            file_ids,
            chunks_written,
            embeddings_written,
        })
    }

    fn needs_compaction(&self) -> Result<bool, DbError> {
        // Two-signal metric-based detection (Phase 0). If stats are unavailable
        // (e.g. DB not open), there's nothing to compact yet.
        let Ok(stats) = self.compaction_stats() else {
            return Ok(false);
        };
        let effective = stats.free_ratio.max(stats.row_waste_ratio);
        Ok(effective >= self.config.compaction_threshold
            && stats.reclaimable >= self.config.compaction_min_size_bytes)
    }

    fn run_compaction(&mut self) -> Result<(), DbError> {
        // 3-phase atomic EXPORT/IMPORT compaction (Phase 0).
        // Falls back to CHECKPOINT-only if EXPORT/IMPORT is unavailable
        // (e.g. DuckDB build without Parquet support).
        if let Err(e) = self.run_attach_copy_compaction() {
            log::warn!(
                "compaction: EXPORT/IMPORT failed ({}), falling back to CHECKPOINT",
                e
            );
            self.reopen_after_compaction_failure()?;
            let conn = self.conn_or_err()?;
            conn.execute_batch("CHECKPOINT")?;
        }
        Ok(())
    }

    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        let conn = self.conn_or_err()?;
        let indexes = Self::discover_hnsw_indexes(conn)?;
        // Build metrics map from discovered indexes before dropping them.  Done as
        // a local so we can drop all conn borrows before assigning to self.
        let new_metrics: HashMap<u32, String> = indexes
            .iter()
            .filter_map(|idx| {
                idx.table_name
                    .strip_prefix("embeddings_")
                    .and_then(|s| s.parse::<u32>().ok())
                    .map(|dims| (dims, idx.metric.clone()))
            })
            .collect();
        for idx in &indexes {
            let safe_name = idx.index_name.replace('"', "\"\"");
            conn.execute(&format!("DROP INDEX IF EXISTS \"{safe_name}\""), [])?;
        }
        // conn borrow ends above; safe to mutate self fields now.
        self.saved_hnsw_metrics = new_metrics;
        self.hnsw_bulk_mode = true;
        Ok(())
    }

    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        // Reset first so that any error path (including early returns) leaves bulk
        // mode off — otherwise close() would retry in a partially-indexed state.
        self.hnsw_bulk_mode = false;
        if !self.has_vss {
            return Ok(());
        }
        // Query the DB directly for embedding tables — mirrors Python's
        // _executor_ensure_all_hnsw_indexes which does not rely on in-memory tracked state.
        // This is more robust than known_dims when the connection is reopened after
        // compaction or when edge cases cause the in-memory set to diverge from DB state.
        // Scope the first conn borrow so it's dropped before we access self.saved_hnsw_metrics.
        let existing = {
            let conn = self.conn_or_err()?;
            Self::discover_embedding_tables(conn)?
        };
        let dims_metrics: Vec<(u32, String)> = existing
            .into_iter()
            .map(|(_, dims)| {
                let metric = self.saved_hnsw_metrics.get(&dims).cloned();
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
        let conn = self.conn_or_err()?;
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
    let size_mb = match std::fs::metadata(db_path) {
        Ok(meta) => meta.len() as f64 / (1024.0 * 1024.0),
        Err(e) => {
            log::warn!(
                "Failed to check disk usage for {}: {}",
                db_path.display(),
                e
            );
            return None;
        }
    };
    (size_mb >= limit_mb).then_some((size_mb, limit_mb))
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
    fn sibling_wal_file_excluded_from_measurement() {
        // Main file well under the limit; a huge sibling `.wal` file must NOT
        // be counted — the helper stats only the exact path given.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let wal_path = tmp.path().join("t.duckdb.wal");
        write_file_of_size(&db_path, 1024); // 1 KB
        write_file_of_size(&wal_path, 20 * 1024 * 1024); // 20 MB — must be ignored
        assert_eq!(check_disk_usage_limit(&db_path, Some(5.0)), None);
    }
}

#[cfg(test)]
mod hnsw_metric_tests {
    use super::*;

    #[test]
    fn test_upsert_file_with_known_id_skips_insert() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db").to_string_lossy().into_owned();
        let config = DbConfig {
            db_path,
            compaction_threshold: 0.3,
            compaction_min_size_bytes: 52_428_800,
            insert_batch_size: 100,
        };
        let mut backend = DuckDbHnswBackend::new(config);
        backend.open().expect("open");

        let batch1 = crate::types::DbWriterBatch {
            files: vec![crate::types::FileRecord {
                existing_file_id: None,
                path: "a.py".into(),
                mtime: Some(1.0),
                size_bytes: Some(100),
                content_hash: Some("abc".into()),
                language: Some("python".into()),
                chunks: vec![],
            }],
            delete_paths: vec![],
        };
        let result1 = backend.write_batch(&batch1).expect("first write");
        let original_id = result1.file_ids[0];

        // Second write: same path, different mtime, but now we know the file's DB id.
        let batch2 = crate::types::DbWriterBatch {
            files: vec![crate::types::FileRecord {
                existing_file_id: Some(original_id),
                path: "a.py".into(),
                mtime: Some(2.0),
                size_bytes: Some(200),
                content_hash: Some("def".into()),
                language: Some("python".into()),
                chunks: vec![],
            }],
            delete_paths: vec![],
        };
        let result2 = backend.write_batch(&batch2).expect("second write");

        assert_eq!(
            result2.file_ids[0], original_id,
            "upsert with known id must return the same id (UPDATE path, not INSERT)"
        );

        // Verify no phantom row was inserted.
        let conn = backend.conn_or_err().expect("conn");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(
            count, 1,
            "files table must have exactly one row after two writes to the same path"
        );
    }

    #[test]
    fn ensure_all_hnsw_indexes_preserves_non_cosine_metric() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db").to_string_lossy().into_owned();
        let config = DbConfig {
            db_path: db_path.clone(),
            compaction_threshold: 0.3,
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

        let mut backend1 = DuckDbHnswBackend::new(test_support::config(db_path_str.clone()));
        backend1.open().expect("open");
        if !backend1.has_vss {
            eprintln!("VSS extension unavailable, skipping");
            return;
        }
        backend1
            .write_batch(&test_support::embedding_batch("a", 128, 50))
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

        let mut backend2 = DuckDbHnswBackend::new(test_support::config(db_path_str));
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
}

/// Shared test fixtures for the modules below — kept separate from
/// `hnsw_metric_tests` because it's used by three otherwise-unrelated modules.
#[cfg(test)]
mod test_support {
    use super::*;

    pub(super) fn config(db_path: String) -> DbConfig {
        DbConfig {
            db_path,
            compaction_threshold: 0.30,
            compaction_min_size_bytes: 52_428_800,
            insert_batch_size: 100,
        }
    }

    pub(super) fn config_with_insert_batch_size(
        db_path: String,
        insert_batch_size: usize,
    ) -> DbConfig {
        DbConfig {
            insert_batch_size,
            ..config(db_path)
        }
    }

    fn chunk_record(code: &str, embedding_dims: Option<u32>) -> ChunkRecord {
        ChunkRecord {
            chunk_type: "function".into(),
            symbol: Some("foo".into()),
            code: code.into(),
            start_line: Some(1),
            end_line: Some(2),
            start_byte: None,
            end_byte: None,
            language: Some("python".into()),
            metadata: None,
            embedding: embedding_dims.map(|d| vec![0.1f32; d as usize]),
            provider: embedding_dims.map(|_| "test".to_string()),
            model: embedding_dims.map(|_| "test-model".to_string()),
        }
    }

    pub(super) fn file_record(path: &str, embedding_dims: Option<u32>) -> FileRecord {
        FileRecord {
            existing_file_id: None,
            path: path.into(),
            mtime: Some(1.0),
            size_bytes: Some(100),
            content_hash: Some("abc123".into()),
            language: Some("python".into()),
            chunks: vec![chunk_record("def foo(): pass", embedding_dims)],
        }
    }

    pub(super) fn single_file_batch(path: &str) -> DbWriterBatch {
        DbWriterBatch {
            files: vec![file_record(path, None)],
            delete_paths: vec![],
        }
    }

    /// `count` files, each with one chunk holding a `dims`-wide embedding.
    /// Paths are prefixed so multiple calls within one test don't collide.
    pub(super) fn embedding_batch(prefix: &str, dims: u32, count: usize) -> DbWriterBatch {
        DbWriterBatch {
            files: (0..count)
                .map(|i| file_record(&format!("{prefix}{i}.py"), Some(dims)))
                .collect(),
            delete_paths: vec![],
        }
    }

    /// A single file with `chunk_count` chunks, each holding a `dims`-wide
    /// embedding when `embedding_dims` is `Some`.
    pub(super) fn file_with_n_chunks(
        path: &str,
        chunk_count: usize,
        embedding_dims: Option<u32>,
    ) -> FileRecord {
        FileRecord {
            existing_file_id: None,
            path: path.into(),
            mtime: Some(1.0),
            size_bytes: Some(100),
            content_hash: Some("abc123".into()),
            language: Some("python".into()),
            chunks: (0..chunk_count)
                .map(|i| chunk_record(&format!("def foo_{i}(): pass"), embedding_dims))
                .collect(),
        }
    }
}

/// Crash recovery via `.swap_intent` files (Invariant 17) — ported from the
/// deleted `RustDbWriter` PyO3 wrapper's test suite so these invariants stay
/// covered without going through PyO3.
#[cfg(test)]
mod crash_recovery_tests {
    use super::*;

    #[test]
    fn pre_swap_intent_cleared_on_open() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");
        std::fs::write(&intent_path, "pre-swap").expect("write intent");

        let mut backend =
            DuckDbHnswBackend::new(test_support::config(db_path.to_string_lossy().into_owned()));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists(), "intent file must be removed");
    }

    #[test]
    fn phase1_intent_restores_old_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        // Simulate a DB that was backed up but not yet swapped: a valid DB
        // with one seed file lives at old_path.
        let mut bootstrap = DuckDbHnswBackend::new(test_support::config(
            old_path.to_string_lossy().into_owned(),
        ));
        bootstrap.open().expect("open");
        bootstrap
            .write_batch(&test_support::single_file_batch("seed.py"))
            .expect("write");
        bootstrap.close().expect("close");

        std::fs::write(&intent_path, "phase1").expect("write intent");

        // open() should detect phase1 and rename old -> db_path.
        let mut backend =
            DuckDbHnswBackend::new(test_support::config(db_path.to_string_lossy().into_owned()));
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
    fn phase2_intent_removes_old_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let old_path = tmp.path().join("t.duckdb.old");
        let intent_path = tmp.path().join("t.duckdb.swap_intent");

        std::fs::write(&old_path, "stale backup marker").expect("write old");
        std::fs::write(&intent_path, "phase2").expect("write intent");

        // A normal DB must already exist at db_path for open() to succeed.
        let mut bootstrap =
            DuckDbHnswBackend::new(test_support::config(db_path.to_string_lossy().into_owned()));
        bootstrap.open().expect("open");
        bootstrap.close().expect("close");

        let mut backend =
            DuckDbHnswBackend::new(test_support::config(db_path.to_string_lossy().into_owned()));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(!intent_path.exists());
        assert!(!old_path.exists());
    }

    #[test]
    fn pre_swap_intent_cleared_on_open_db_extension() {
        // Regression guard: PathBuf::set_extension() on "chunks.db" would produce
        // "chunks.duckdb.swap_intent" instead of "chunks.db.swap_intent". The
        // correct implementation builds the intent path via string concatenation.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("chunks.db");
        let intent_path = tmp.path().join("chunks.db.swap_intent");
        std::fs::write(&intent_path, "pre-swap").expect("write intent");

        let mut backend =
            DuckDbHnswBackend::new(test_support::config(db_path.to_string_lossy().into_owned()));
        backend.open().expect("open");
        backend.close().expect("close");

        assert!(
            !intent_path.exists(),
            "intent file must be removed; wrong path construction would leave it untouched"
        );
    }
}

/// `insert_batch_size` (from Python's `indexing.db_batch_size`) parameterizes
/// the row-count-per-INSERT-statement chunking in `insert_chunks_for_file`/
/// `insert_embeddings_txn`. These tests prove correctness at a non-default
/// batch size that doesn't evenly divide the row count — the legitimate
/// external contract here is "no rows dropped/duplicated at a batch
/// boundary," not the internal INSERT-statement count itself.
#[cfg(test)]
mod insert_batch_size_tests {
    use super::*;

    #[test]
    fn chunks_persist_correctly_at_non_default_batch_size() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend =
            DuckDbHnswBackend::new(test_support::config_with_insert_batch_size(db_path, 3));
        backend.open().expect("open");

        // 7 chunks, no embeddings — doesn't evenly divide the batch size of 3
        // (batches of 3, 3, 1), exercising insert_chunks_for_file's chunking.
        let batch = DbWriterBatch {
            files: vec![test_support::file_with_n_chunks("a.py", 7, None)],
            delete_paths: vec![],
        };
        let result = backend.write_batch(&batch).expect("write");
        backend.close().expect("close");

        assert_eq!(result.chunks_written, 7);
        let conn = Connection::open(tmp.path().join("t.duckdb")).expect("reopen");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .expect("count");
        assert_eq!(
            count, 7,
            "all 7 chunks must persist despite the 3-row batch boundary"
        );
    }

    #[test]
    fn embeddings_persist_correctly_at_non_default_batch_size() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend =
            DuckDbHnswBackend::new(test_support::config_with_insert_batch_size(db_path, 3));
        backend.open().expect("open");

        // 7 chunks each with a 4-dim embedding, in one file — doesn't evenly
        // divide the batch size of 3, exercising insert_embeddings_txn's
        // chunking (grouped by dims).
        let batch = DbWriterBatch {
            files: vec![test_support::file_with_n_chunks("a.py", 7, Some(4))],
            delete_paths: vec![],
        };
        let result = backend.write_batch(&batch).expect("write");
        backend.close().expect("close");

        assert_eq!(result.embeddings_written, 7);
        let conn = Connection::open(tmp.path().join("t.duckdb")).expect("reopen");
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM embeddings_4", [], |r| r.get(0))
            .expect("count");
        assert_eq!(
            count, 7,
            "all 7 embeddings must persist despite the 3-row batch boundary"
        );
    }
}
