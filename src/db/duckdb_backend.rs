use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use duckdb::Connection;

use crate::db::{DbBackend, DbConfig};
use crate::error::DbError;
use crate::types::{BatchResult, ChunkRecord, DbWriterBatch, FileRecord};

pub struct DuckDbHnswBackend {
    config: DbConfig,
    conn: Option<Connection>,
    write_count: u32,
    has_vss: bool,
    hnsw_cache: Option<Vec<HnswIndexInfo>>,
    hnsw_bulk_mode: bool,
    // Dims for which embeddings_N tables are known to exist in this session.
    // Used to detect new dimensions mid-session so the HNSW cache can be
    // invalidated and a fresh HNSW index created after the first commit.
    known_dims: HashSet<u32>,
}

#[derive(Debug, Clone)]
struct HnswIndexInfo {
    index_name: String,
    table_name: String,
    create_sql: Option<String>,
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
            write_count: 0,
            has_vss: false,
            hnsw_cache: None,
            hnsw_bulk_mode: false,
            known_dims: HashSet::new(),
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
    // tests/test_rust_db_writer.py::TestSchemaParity catches column-level drift at CI time.
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
        match conn.execute_batch(
            "INSTALL vss; LOAD vss; SET hnsw_enable_experimental_persistence = true;",
        ) {
            Ok(()) => true,
            Err(e) => {
                log::warn!("VSS extension unavailable (vector search disabled): {e}");
                false
            }
        }
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
            .map(|(name, table, sql)| HnswIndexInfo {
                index_name: name,
                table_name: table,
                create_sql: sql,
            })
            .collect();
        Ok(indexes)
    }

    fn drop_hnsw_indexes(conn: &Connection, indexes: &[HnswIndexInfo]) -> Result<(), DbError> {
        for idx in indexes {
            let safe_name = idx.index_name.replace('"', "\"\"");
            conn.execute(&format!("DROP INDEX IF EXISTS \"{safe_name}\""), [])?;
        }
        Ok(())
    }

    fn recreate_hnsw_indexes(conn: &Connection, indexes: &[HnswIndexInfo]) -> Result<(), DbError> {
        for idx in indexes {
            if let Some(create_sql) = &idx.create_sql {
                // Ensure idempotent
                let sql = if create_sql.contains("IF NOT EXISTS") {
                    create_sql.clone()
                } else {
                    create_sql.replacen("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)
                };
                conn.execute_batch(&sql)?;
            } else {
                let safe_idx = idx.index_name.replace('"', "\"\"");
                let safe_tbl = idx.table_name.replace('"', "\"\"");
                conn.execute_batch(&format!(
                    "CREATE INDEX IF NOT EXISTS \"{safe_idx}\" ON \"{safe_tbl}\" USING HNSW (embedding) WITH (metric = 'cosine')"
                ))?;
            }
        }
        Ok(())
    }

    fn count_total_embeddings(batch: &DbWriterBatch) -> usize {
        batch
            .files
            .iter()
            .flat_map(|f| &f.chunks)
            .filter(|c| c.embedding.as_ref().map(|e| !e.is_empty()).unwrap_or(false))
            .count()
    }

    fn collect_unique_dims(batch: &DbWriterBatch) -> HashSet<u32> {
        batch
            .files
            .iter()
            .flat_map(|f| &f.chunks)
            .filter_map(|c| c.embedding.as_ref())
            .filter(|e| !e.is_empty())
            .map(|e| e.len() as u32)
            .collect()
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

        // DuckDB rejects ON CONFLICT DO UPDATE inside an explicit transaction when
        // a FK child table (chunks) has rows referencing the conflicting parent row,
        // even if those children were deleted earlier in the same transaction.
        // Work around by doing an explicit SELECT then UPDATE-or-INSERT.
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
    ) -> Result<Vec<i64>, DbError> {
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        conn.execute_batch(
            "CREATE TEMPORARY TABLE IF NOT EXISTS rust_temp_chunks (
                file_id INTEGER,
                chunk_type TEXT,
                symbol TEXT,
                code TEXT,
                start_line INTEGER,
                end_line INTEGER,
                start_byte INTEGER,
                end_byte INTEGER,
                language TEXT,
                metadata TEXT
            );
            DELETE FROM rust_temp_chunks;",
        )?;

        // Batch 100 rows per INSERT to cut SQL round-trips ~100× vs one-row-at-a-time.
        // Mirrors the same pattern used in insert_embeddings_txn.
        const CHUNK_INSERT_BATCH: usize = 100;
        for chunk_slice in chunks.chunks(CHUNK_INSERT_BATCH) {
            let row_ph = std::iter::repeat_n("(?,?,?,?,?,?,?,?,?,?)", chunk_slice.len())
                .collect::<Vec<_>>()
                .join(",");
            let sql = format!(
                "INSERT INTO rust_temp_chunks \
                 (file_id, chunk_type, symbol, code, start_line, end_line, \
                  start_byte, end_byte, language, metadata) VALUES {row_ph}"
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
            conn.execute(&sql, duckdb::params_from_iter(params))?;
        }

        let mut stmt = conn.prepare(
            "INSERT INTO chunks
             (file_id, chunk_type, symbol, code, start_line, end_line,
              start_byte, end_byte, language, metadata)
             SELECT file_id, chunk_type, symbol, code, start_line, end_line,
                    start_byte, end_byte, language, metadata
             FROM rust_temp_chunks
             RETURNING id",
        )?;

        let ids: Vec<i64> = stmt
            .query_map([], |row| row.get(0))?
            .collect::<Result<Vec<i64>, _>>()
            .map_err(DbError::DuckDb)?;
        Ok(ids)
    }

    fn insert_embeddings_txn(
        conn: &Connection,
        batch: &DbWriterBatch,
        embedding_pairs: &[(i64, usize, usize)], // (chunk_id, file_idx, chunk_idx)
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

        let mut total = 0u64;
        for (dims, items) in &by_dims {
            let table = format!("embeddings_{dims}");
            let temp = format!("rust_temp_emb_{dims}");

            conn.execute_batch(&format!(
                "CREATE TEMPORARY TABLE IF NOT EXISTS {temp} (
                    chunk_id INTEGER,
                    provider TEXT,
                    model TEXT,
                    embedding TEXT,
                    dims INTEGER
                );
                DELETE FROM {temp};"
            ))?;

            // Batch 100 rows per INSERT to cut SQL round-trips ~100×.
            const EMBED_INSERT_BATCH: usize = 100;
            for chunk_slice in items.chunks(EMBED_INSERT_BATCH) {
                let row_ph = std::iter::repeat_n("(?,?,?,?,?)", chunk_slice.len())
                    .collect::<Vec<_>>()
                    .join(",");
                let sql = format!(
                    "INSERT INTO {temp} (chunk_id, provider, model, embedding, dims) VALUES {row_ph}"
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
                conn.execute(&sql, duckdb::params_from_iter(params))?;
            }

            let rows = conn.execute(
                &format!(
                    "INSERT INTO \"{table}\" (chunk_id, provider, model, embedding, dims)
                     SELECT chunk_id, provider, model, embedding::FLOAT[{dims}], dims
                     FROM {temp}
                     ON CONFLICT (chunk_id, provider, model) DO UPDATE
                     SET embedding = EXCLUDED.embedding, dims = EXCLUDED.dims"
                ),
                [],
            )?;
            total += rows as u64;
        }
        Ok(total)
    }

    // delete_paths (explicit path removals from the caller) run outside the transaction
    // to avoid the DuckDB limitation where ON CONFLICT DO UPDATE on a FK parent row
    // is rejected inside an explicit transaction even after child rows are deleted.
    // The pre-deletes inside write_batch_inner work because upsert_file uses an explicit
    // SELECT + UPDATE/INSERT rather than ON CONFLICT DO UPDATE syntax.
    fn delete_paths(conn: &Connection, paths: &[String]) -> Result<(), DbError> {
        if paths.is_empty() {
            return Ok(());
        }
        // TODO(Phase 1): cache emb_tables on DuckDbHnswBackend (alongside hnsw_cache) so
        // this catalog scan is not repeated for every batch.  See pre_delete_for_upsert for the
        // matching TODO and the invalidation note (cache must grow when a new embeddings_N table
        // appears).  When caching is implemented, both TODOs should be resolved together.
        let emb_tables = Self::discover_embedding_tables(conn)?;

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
    fn pre_delete_for_upsert(conn: &Connection, batch: &DbWriterBatch) -> Result<(), DbError> {
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

        // TODO(Phase 1): cache emb_tables on DuckDbHnswBackend (alongside hnsw_cache) instead
        // of re-querying the catalog on every batch. Hoisting is non-trivial because the set can
        // grow mid-session when a new embedding dimension appears for the first time — the cache
        // must be invalidated whenever a new embeddings_N table is created.
        let emb_tables = Self::discover_embedding_tables(conn)?;
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
    fn available_disk_space(_dir: &Path) -> Option<u64> {
        // Best-effort: platform-specific implementations can be added.
        None
    }

    /// 3-phase EXPORT/IMPORT atomic swap (Section 22.5).
    ///
    /// Phase 1: EXPORT current DB to Parquet while connection is open,
    ///          then CHECKPOINT + close.
    /// Phase 2: Write intent files, rename old DB, IMPORT into a fresh DB,
    ///          rebuild HNSW indexes.
    /// Phase 3: Atomic rename of compacted DB to active path, cleanup.
    fn run_export_import_compaction(&mut self) -> Result<(), DbError> {
        let db_path = PathBuf::from(&self.config.db_path);
        let export_dir = PathBuf::from(format!("{}.export_tmp", self.config.db_path));
        let compact_path = PathBuf::from(format!("{}.compact", self.config.db_path));
        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
        let intent_path = PathBuf::from(format!("{}.swap_intent", self.config.db_path));

        // --- Phase 1: Export while connection is open ----------------------
        let conn = self.conn_or_err()?;

        // Snapshot HNSW DDL before closing (needed for rebuild in Phase 2).
        let hnsw_indexes = Self::discover_hnsw_indexes(conn)?;

        // Preflight disk space: need ~3× DB size for export + compact files.
        let db_size = std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0);
        if let Some(avail) = Self::available_disk_space(&db_path) {
            if avail < db_size * 3 {
                return Err(DbError::Other(format!(
                    "insufficient disk space for compaction: \
                     need {} bytes, have {} bytes",
                    db_size * 3,
                    avail
                )));
            }
        }

        // Clean up stale artifacts from a prior failed compaction.
        let _ = std::fs::remove_dir_all(&export_dir);
        let _ = std::fs::remove_file(&compact_path);

        // EXPORT current database schema + data to Parquet files.
        let export_sql = format!(
            "EXPORT DATABASE '{}' (FORMAT PARQUET)",
            export_dir.display()
        );
        log::info!("compaction: {}", export_sql);
        conn.execute_batch(&export_sql)?;

        // CHECKPOINT and close the live connection.
        conn.execute_batch("CHECKPOINT")?;
        self.conn = None;

        // --- Phase 2: Rename old DB, create compacted DB ------------------
        Self::write_intent(&intent_path, "pre-swap")?;
        std::fs::rename(&db_path, &old_path)?;

        Self::write_intent(&intent_path, "phase1")?;

        // Create a fresh DB file and IMPORT the Parquet export.
        let import_conn = Connection::open(&compact_path)?;
        let import_sql = format!("IMPORT DATABASE '{}'", export_dir.display());
        log::info!("compaction: {}", import_sql);
        let import_result = import_conn.execute_batch(&import_sql);

        // Rebuild HNSW indexes on the compacted DB.
        if !hnsw_indexes.is_empty() {
            if let Err(e) = Self::recreate_hnsw_indexes(&import_conn, &hnsw_indexes) {
                log::warn!(
                    "compaction: HNSW rebuild failed ({}), \
                     continuing without indexes",
                    e
                );
            }
        }
        import_conn.execute_batch("CHECKPOINT")?;
        drop(import_conn); // Close before rename.

        // Check import result AFTER closing the connection.
        import_result?;

        // Clean up export temp directory.
        let _ = std::fs::remove_dir_all(&export_dir);

        // --- Phase 3: Atomic rename to active path ------------------------
        Self::write_intent(&intent_path, "phase2")?;
        std::fs::rename(&compact_path, &db_path)?;

        // Clean up intent file and old DB.
        let _ = std::fs::remove_file(&intent_path);
        let _ = std::fs::remove_file(&old_path);

        // --- Reopen connection to compacted DB ----------------------------
        self.reopen()?;
        self.write_count = 0;
        self.hnsw_cache = None;
        log::info!("compaction: complete");
        Ok(())
    }

    /// Reopen the connection after a successful compaction.
    fn reopen(&mut self) -> Result<(), DbError> {
        self.known_dims.clear();
        let conn = Connection::open(&self.config.db_path)?;
        self.has_vss = Self::try_load_vss(&conn);
        Self::setup_schema(&conn)?;
        let existing = Self::discover_embedding_tables(&conn)?;
        self.known_dims
            .extend(existing.into_iter().map(|(_, dims)| dims));
        self.conn = Some(conn);
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
    fn write_batch_inner(conn: &Connection, batch: &DbWriterBatch) -> Result<BatchInner, DbError> {
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
            let chunk_ids = Self::insert_chunks_for_file(conn, file_id, &file.chunks)?;
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
            let tables = Self::discover_embedding_tables(conn).unwrap_or_default();
            let mut total = 0i64;
            for (table_name, _) in &tables {
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
            let tables = Self::discover_embedding_tables(conn).unwrap_or_default();
            let mut total = 0i64;
            for (table_name, _) in &tables {
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
                        let old_path = PathBuf::from(format!("{}.old", self.config.db_path));
                        let _ = std::fs::remove_file(&old_path);
                        let _ = std::fs::remove_file(&intent_path);
                    }
                    _ => {}
                }
            }
        }

        self.known_dims.clear();
        let conn = Connection::open(&self.config.db_path)?;
        self.has_vss = Self::try_load_vss(&conn);
        Self::setup_schema(&conn)?;
        // Prime known_dims from tables that already exist so the first batch
        // with an existing dimension does not trigger a spurious cache invalidation.
        let existing = Self::discover_embedding_tables(&conn)?;
        self.known_dims
            .extend(existing.into_iter().map(|(_, dims)| dims));
        self.conn = Some(conn);
        // Crash recovery: if the process was killed between Step 2 (DROP HNSW) and
        // Step 5 (RECREATE HNSW) in write_batch, HNSW indexes are absent but the
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
        // Step 0a: Handle delete_paths OUTSIDE transaction (Invariant: DuckDB rejects
        // FK parent-row deletes inside an explicit transaction after child deletes).
        //
        // Atomicity gap: if the process crashes after delete_paths commits but before
        // the write transaction below commits, the deleted files are permanently absent
        // from the DB.  Unlike the upsert path (file rows survive → self-healing),
        // delete_paths removes the file rows themselves.  The caller must be prepared
        // to re-submit delete requests after a crash — the indexer has no way to know
        // which files were intended for deletion.  Source files on disk are never
        // touched by this code path.
        if !batch.delete_paths.is_empty() {
            let conn = self.conn_or_err()?;
            Self::delete_paths(conn, &batch.delete_paths)?;
        }

        // Step 0b: Pre-delete chunks/embeddings for files being upserted, OUTSIDE transaction.
        // DuckDB's FK check engine sees the committed DB state, not the current transaction's
        // in-flight deletes.  Any UPDATE on files inside a txn where child chunks were deleted
        // earlier in that same txn is rejected — even though no FK is violated at commit time.
        // Running these deletes before BEGIN avoids the spurious constraint error.
        {
            let conn = self.conn_or_err()?;
            Self::pre_delete_for_upsert(conn, batch)?;
        }

        // Step 0c: Ensure embedding tables outside txn (Invariant 13).
        // Track which dims are genuinely new so we can (a) invalidate the stale
        // HNSW cache and (b) create a fresh HNSW index for the new table after
        // the commit (Step 5+).  The per-batch bookend only manages EXISTING
        // indexes, so a brand-new embeddings_N table would never get one otherwise.
        let unique_dims = Self::collect_unique_dims(batch);
        let new_dims: Vec<u32> = unique_dims.difference(&self.known_dims).copied().collect();
        {
            let conn = self.conn_or_err()?;
            for &dims in &unique_dims {
                Self::ensure_embedding_table_dims(conn, dims)?;
            }
        }
        self.known_dims.extend(unique_dims.iter().copied());
        if !new_dims.is_empty() {
            // The cached HNSW snapshot predates the new table(s) — force rediscovery.
            self.hnsw_cache = None;
        }

        // Step 1: Count embeddings to decide HNSW lifecycle
        let total_emb = Self::count_total_embeddings(batch);

        // Step 2: Discover + DROP HNSW indexes BEFORE BEGIN (Invariant 14).
        // Cache the index DDL after first successful discovery to skip the
        // catalog query on every subsequent batch.
        let hnsw_indexes = {
            let conn = self
                .conn
                .as_ref()
                .ok_or_else(|| DbError::Other("not open".into()))?;
            if !self.hnsw_bulk_mode && self.has_vss && total_emb >= 50 {
                let indexes = match &self.hnsw_cache {
                    Some(cached) => cached.clone(),
                    None => {
                        let discovered = Self::discover_hnsw_indexes(conn)?;
                        // Cache even an empty result so subsequent batches don't
                        // re-query the catalog when no HNSW indexes exist yet.
                        self.hnsw_cache = Some(discovered.clone());
                        discovered
                    }
                };
                if !indexes.is_empty() {
                    Self::drop_hnsw_indexes(conn, &indexes)?;
                }
                indexes
            } else {
                vec![]
            }
        };

        // Step 3: Transaction
        let batch_inner = {
            let conn = self.conn_or_err()?;
            conn.execute_batch("BEGIN")?;

            match Self::write_batch_inner(conn, batch) {
                Ok(inner) => inner,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    // Try to restore HNSW even on error (Invariant 14)
                    if !hnsw_indexes.is_empty() {
                        let _ = Self::recreate_hnsw_indexes(conn, &hnsw_indexes);
                    }
                    return Err(e);
                }
            }
        };
        let (file_ids, chunks_written, embedding_pairs) = (
            batch_inner.file_ids,
            batch_inner.chunks_written,
            batch_inner.embedding_pairs,
        );

        // Step 3e: Insert embeddings (still inside txn)
        let embeddings_written = {
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and BEGIN passed");
            match Self::insert_embeddings_txn(conn, batch, &embedding_pairs) {
                Ok(n) => n,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    if !hnsw_indexes.is_empty() {
                        let _ = Self::recreate_hnsw_indexes(conn, &hnsw_indexes);
                    }
                    return Err(e);
                }
            }
        };

        // Step 4: COMMIT
        {
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and BEGIN passed");
            if let Err(e) = conn.execute_batch("COMMIT") {
                let _ = conn.execute_batch("ROLLBACK");
                if !hnsw_indexes.is_empty() {
                    let _ = Self::recreate_hnsw_indexes(conn, &hnsw_indexes);
                }
                return Err(DbError::DuckDb(e));
            }
            // Force CHECKPOINT after COMMIT to flush all data to the main DB file.
            // This is critical when the initial DB was created by a different DuckDB
            // library (e.g. Python), which may leave a WAL whose replay state must be
            // merged with our writes before subsequent connections can see the data.
            conn.execute_batch("CHECKPOINT")?;
        }

        // Step 5: RECREATE HNSW AFTER COMMIT (Invariant 14)
        if !hnsw_indexes.is_empty() {
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and COMMIT passed");
            Self::recreate_hnsw_indexes(conn, &hnsw_indexes)?;
        }

        // Step 5+: Create HNSW indexes for newly-introduced embedding dimensions.
        // The bookend above only recreates indexes that existed before Step 2's drop;
        // a brand-new embeddings_N table has no HNSW yet.  Create it now so that
        // subsequent batches can manage it through the normal drop/recreate cycle.
        // hnsw_cache was already invalidated in Step 0c, so the next batch rediscovers.
        if !new_dims.is_empty() && self.has_vss {
            let conn = self
                .conn
                .as_ref()
                .expect("conn is Some: open() succeeded and COMMIT passed");
            for &dims in &new_dims {
                let hnsw_name = format!("idx_hnsw_{dims}");
                let table = format!("embeddings_{dims}");
                conn.execute_batch(&format!(
                    "CREATE INDEX IF NOT EXISTS \"{hnsw_name}\" ON \"{table}\" USING HNSW (embedding) WITH (metric = 'cosine')"
                ))?;
            }
        }

        self.write_count += 1;
        Ok(BatchResult {
            file_ids,
            chunks_written,
            embeddings_written,
        })
    }

    fn needs_compaction(&self) -> Result<bool, DbError> {
        // Two-signal metric-based detection (Phase 0).
        // Falls back to simple write-count threshold when stats are unavailable.
        if let Ok(stats) = self.compaction_stats() {
            let effective = stats.free_ratio.max(stats.row_waste_ratio);
            if effective >= self.config.compaction_threshold
                && stats.reclaimable >= self.config.compaction_min_size_bytes
            {
                return Ok(true);
            }
        }
        Ok(self.write_count >= self.config.compaction_batch_threshold)
    }

    fn run_compaction(&mut self) -> Result<(), DbError> {
        // 3-phase atomic EXPORT/IMPORT compaction (Phase 0).
        // Falls back to CHECKPOINT-only if EXPORT/IMPORT is unavailable
        // (e.g. DuckDB build without Parquet support).
        if let Err(e) = self.run_export_import_compaction() {
            log::warn!(
                "compaction: EXPORT/IMPORT failed ({}), falling back to CHECKPOINT",
                e
            );
            self.reopen_after_compaction_failure()?;
            let conn = self.conn_or_err()?;
            conn.execute_batch("CHECKPOINT")?;
            self.write_count = 0;
        }
        Ok(())
    }

    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        let conn = self.conn_or_err()?;
        let indexes = Self::discover_hnsw_indexes(conn)?;
        for idx in &indexes {
            let safe_name = idx.index_name.replace('"', "\"\"");
            conn.execute(&format!("DROP INDEX IF EXISTS \"{safe_name}\""), [])?;
        }
        self.hnsw_bulk_mode = true;
        self.hnsw_cache = None;
        Ok(())
    }

    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        // Reset first so that any error path (including early returns) leaves bulk
        // mode off — otherwise close() would retry in a partially-indexed state.
        self.hnsw_bulk_mode = false;
        if !self.has_vss {
            return Ok(());
        }
        let conn = self.conn_or_err()?;
        let tables = Self::discover_embedding_tables(conn)?;
        for (table_name, dims) in &tables {
            let hnsw_name = format!("idx_hnsw_{dims}");
            let safe_tbl = table_name.replace('"', "\"\"");
            conn.execute_batch(&format!(
                "CREATE INDEX IF NOT EXISTS \"{hnsw_name}\" ON \"{safe_tbl}\" USING HNSW (embedding) WITH (metric = 'cosine')"
            ))?;
        }
        if !tables.is_empty() {
            conn.execute_batch("CHECKPOINT")?;
        }
        Ok(())
    }
}
