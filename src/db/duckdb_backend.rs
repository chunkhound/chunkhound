use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use duckdb::Connection;

use crate::db::DbConfig;
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

impl DuckDbHnswBackend {
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
        conn.execute_batch(
            "INSTALL vss; LOAD vss; SET hnsw_enable_experimental_persistence = true;",
        )
        .is_ok()
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
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_string();

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
    // The pre-deletes inside write_batch_txn work because upsert_file uses an explicit
    // SELECT + UPDATE/INSERT rather than ON CONFLICT DO UPDATE syntax.
    fn delete_paths(conn: &Connection, paths: &[String]) -> Result<(), DbError> {
        if paths.is_empty() {
            return Ok(());
        }
        // Discover embedding tables once for this call.
        let emb_tables = Self::discover_embedding_tables(conn)?;
        const DELETE_BATCH: usize = 500;
        for batch in paths.chunks(DELETE_BATCH) {
            let ph = std::iter::repeat_n("?", batch.len())
                .collect::<Vec<_>>()
                .join(",");
            let params: Vec<duckdb::types::Value> = batch
                .iter()
                .map(|p| duckdb::types::Value::Text(p.clone()))
                .collect();
            // embeddings_N tables have no FK to chunks — delete embeddings first
            // to avoid ghost rows accumulating on re-index (CF-1).
            let chunk_subquery = format!(
                "SELECT id FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path IN ({ph}))"
            );
            for (table_name, _dims) in &emb_tables {
                conn.execute(
                    &format!("DELETE FROM \"{table_name}\" WHERE chunk_id IN ({chunk_subquery})"),
                    duckdb::params_from_iter(params.clone()),
                )?;
            }
            conn.execute(
                &format!("DELETE FROM chunks WHERE file_id IN (SELECT id FROM files WHERE path IN ({ph}))"),
                duckdb::params_from_iter(params.clone()),
            )?;
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

    // (file_ids, chunks_written, embedding_pairs: (chunk_id, file_idx, chunk_idx))
    // The tuple return avoids an intermediate struct for an internal helper that is only
    // ever called once; extracting a named struct would add indirection with no clarity gain.
    #[allow(clippy::type_complexity)]
    fn write_batch_txn(
        conn: &Connection,
        batch: &DbWriterBatch,
    ) -> Result<(Vec<i64>, u64, Vec<(i64, usize, usize)>), DbError> {
        // PF-3: Batch pre-deletes — one IN-list per group instead of one statement per file.
        // embeddings_N tables have no FK cascade from chunks, so embeddings must be deleted
        // before chunks to avoid ghost rows on re-index (CF-2).
        // Split files into those with a known file_id (by_id) and path-only (by_path);
        // each group is cleared with 2 statements rather than 2 × N.
        // TODO(Phase 1): pass known emb_tables in from the caller instead of re-querying
        // the catalog on every batch inside the transaction.
        let emb_tables = Self::discover_embedding_tables(conn)?;

        let by_id: Vec<i64> = batch
            .files
            .iter()
            .filter_map(|f| f.existing_file_id)
            .collect();
        let by_path: Vec<&str> = batch
            .files
            .iter()
            .filter(|f| f.existing_file_id.is_none())
            .map(|f| f.path.as_str())
            .collect();

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
                .map(|p| duckdb::types::Value::Text(p.to_string()))
                .collect();
            let chunk_subquery = format!(
                "SELECT id FROM chunks WHERE file_id IN \
                 (SELECT id FROM files WHERE path IN ({ph}))"
            );
            for (table_name, _dims) in &emb_tables {
                conn.execute(
                    &format!("DELETE FROM \"{table_name}\" WHERE chunk_id IN ({chunk_subquery})"),
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

        Ok((file_ids, total_chunks, embedding_pairs))
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
        Ok(())
    }

    fn close(&mut self) -> Result<(), DbError> {
        if self.hnsw_bulk_mode {
            if let Err(e) = self.ensure_all_hnsw_indexes() {
                eprintln!(
                    "chunkhound_native: warning: ensure_all_hnsw_indexes on close failed: {e}"
                );
            }
        }
        if let Some(conn) = self.conn.as_ref() {
            if let Err(e) = conn.execute_batch("CHECKPOINT") {
                eprintln!("chunkhound_native: warning: CHECKPOINT on close failed: {e}");
            }
        }
        self.conn = None;
        Ok(())
    }

    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        // Step 0a: Handle delete_paths OUTSIDE transaction (Invariant: DuckDB rejects
        // FK parent-row deletes inside an explicit transaction after child deletes)
        if !batch.delete_paths.is_empty() {
            let conn = self
                .conn
                .as_ref()
                .ok_or_else(|| DbError::Other("not open".into()))?;
            Self::delete_paths(conn, &batch.delete_paths)?;
        }

        // Step 0b: Ensure embedding tables outside txn (Invariant 13).
        // Track which dims are genuinely new so we can (a) invalidate the stale
        // HNSW cache and (b) create a fresh HNSW index for the new table after
        // the commit (Step 5+).  The per-batch bookend only manages EXISTING
        // indexes, so a brand-new embeddings_N table would never get one otherwise.
        let unique_dims = Self::collect_unique_dims(batch);
        let new_dims: Vec<u32> = unique_dims.difference(&self.known_dims).copied().collect();
        {
            let conn = self
                .conn
                .as_ref()
                .ok_or_else(|| DbError::Other("not open".into()))?;
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
        let (file_ids, chunks_written, embedding_pairs) = {
            let conn = self
                .conn
                .as_ref()
                .ok_or_else(|| DbError::Other("not open".into()))?;
            conn.execute_batch("BEGIN")?;

            match Self::write_batch_txn(conn, batch) {
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
        // hnsw_cache was already invalidated in Step 0b, so the next batch rediscovers.
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

    fn needs_compaction(&mut self) -> Result<bool, DbError> {
        Ok(self.write_count >= self.config.compaction_batch_threshold)
    }

    fn run_compaction(&mut self) -> Result<(), DbError> {
        // Phase 0: CHECKPOINT only.
        // TODO(Phase 1): replace with full 3-phase atomic EXPORT/IMPORT swap.
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| DbError::Other("not open".into()))?;
        conn.execute_batch("CHECKPOINT")?;
        self.write_count = 0;
        Ok(())
    }

    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| DbError::Other("not open".into()))?;
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
        let conn = self
            .conn
            .as_ref()
            .ok_or_else(|| DbError::Other("not open".into()))?;
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
