use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

use duckdb::Connection;

use super::DuckDbHnswBackend;
use crate::error::DbError;
use crate::types::{BatchResult, ChunkRecord, DbWriterBatch, FileRecord};

struct BatchInner {
    file_ids: Vec<i64>,
    chunks_written: u64,
    embedding_pairs: Vec<(i64, usize, usize)>,
}

impl DuckDbHnswBackend {
    /// Batch size for path-based DELETE operations (delete_paths, pre_delete_for_upsert).
    /// 500 paths per batch balances SQL round-trip overhead vs memory usage for IN-list
    /// parameter binding.
    const DELETE_BATCH: usize = 500;

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
        // Skip the SELECT and go straight to UPDATE — but require the id to still match
        // this exact path and to have actually matched a row before trusting it. A stale
        // id (e.g. the diff snapshot outliving a concurrent delete/rename) must not be
        // reported as success: insert_chunks_for_file would either violate the
        // files->chunks FK on a nonexistent id, or — worse — silently attach this file's
        // chunks to an unrelated file's row. On a mismatch, fall through to the
        // path-keyed slow path below instead of trusting the stale id.
        if let Some(id) = file.existing_file_id {
            let rows_updated = conn.execute(
                "UPDATE files SET size = ?, modified_time = CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, content_hash = ?, language = ?, skip_reason = ?, updated_at = now() WHERE id = ? AND path = ?",
                duckdb::params![file.size_bytes, file.mtime, file.mtime, file.content_hash, file.language, file.skip_reason, id, file.path],
            )?;
            if rows_updated == 1 {
                return Ok(id);
            }
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
                "UPDATE files SET size = ?, modified_time = CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, content_hash = ?, language = ?, skip_reason = ?, updated_at = now() WHERE id = ?",
                duckdb::params![file.size_bytes, file.mtime, file.mtime, file.content_hash, file.language, file.skip_reason, id],
            )?;
            Ok(id)
        } else {
            let id: i64 = conn.query_row(
                "INSERT INTO files (path, name, extension, size, modified_time, content_hash, language, skip_reason)
                 VALUES (?, ?, ?, ?, CASE WHEN ? IS NOT NULL THEN to_timestamp(?) ELSE NULL END, ?, ?, ?)
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
                    file.skip_reason,
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

    // Pre-deletes chunks (and orphaned embeddings) for files about to be upserted.
    // Must run OUTSIDE the write transaction: DuckDB's FK check engine sees the committed
    // state of the DB, not the current transaction's state. Any UPDATE on files inside a
    // transaction where child chunks were deleted earlier in the same transaction is rejected
    // with a FK constraint error — even though no FK is actually violated at commit time.
    // The same limitation affects delete_paths; both are handled identically (pre-txn commit).
    //
    // Atomicity gap — two cases:
    //
    // (a) Upsert files: chunks/embeddings are deleted here and the file row is marked
    //     dirty (`modified_time` and `content_hash` set to NULL) in the same COMMIT.
    //     If the process crashes before the write transaction below, the next
    //     incremental run sees NULL mtime and reprocesses the file (differ.rs treats
    //     NULL modified_time as "changed" and does not hash-confirm skip). Leaving
    //     mtime/hash intact would look "unchanged" after a force-reindex crash and
    //     skip rewrite, leaving the file with zero chunks.
    //     The dirty UPDATE runs *before* the chunk DELETEs in this transaction so we
    //     do not trip DuckDB's FK check (UPDATE on files after deleting child chunks
    //     in the same txn is rejected). Successful upsert_file overwrites the NULLs.
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
        // Verify each candidate id still points at the same path before trusting it — a
        // stale id (e.g. the diff snapshot outliving a concurrent delete/rename) must fall
        // back to the path-keyed path below instead of dirtying/deleting an unrelated
        // file's row. Mirrors upsert_file's `id = ? AND path = ?` fast-path guard.
        let candidate_ids: Vec<i64> = batch
            .files
            .iter()
            .filter_map(|f| f.existing_file_id)
            .collect();
        let mut id_to_path: HashMap<i64, String> = HashMap::new();
        if !candidate_ids.is_empty() {
            let ph = std::iter::repeat_n("?", candidate_ids.len())
                .collect::<Vec<_>>()
                .join(",");
            let params: Vec<duckdb::types::Value> = candidate_ids
                .iter()
                .map(|&id| duckdb::types::Value::BigInt(id))
                .collect();
            let mut stmt =
                conn.prepare(&format!("SELECT id, path FROM files WHERE id IN ({ph})"))?;
            let rows = stmt
                .query_map(duckdb::params_from_iter(params), |r| {
                    Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(DbError::DuckDb)?;
            id_to_path.extend(rows);
        }

        let by_id: Vec<i64> = batch
            .files
            .iter()
            .filter_map(|f| {
                f.existing_file_id
                    .filter(|id| id_to_path.get(id) == Some(&f.path))
            })
            .collect();
        let by_path: Vec<String> = batch
            .files
            .iter()
            .filter(|f| match f.existing_file_id {
                None => true,
                Some(id) => id_to_path.get(&id) != Some(&f.path),
            })
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
                // Dirty marker first — see comment (a) above.
                conn.execute(
                    &format!(
                        "UPDATE files SET modified_time = NULL, content_hash = NULL \
                         WHERE id IN ({ph})"
                    ),
                    duckdb::params_from_iter(params.clone()),
                )?;
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
                conn.execute(
                    &format!(
                        "UPDATE files SET modified_time = NULL, content_hash = NULL \
                         WHERE path IN ({ph})"
                    ),
                    duckdb::params_from_iter(params.clone()),
                )?;
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
}

pub(super) fn prepare_write(
    backend: &mut DuckDbHnswBackend,
    batch: &DbWriterBatch,
) -> Result<(), DbError> {
    // Step 0a: Handle delete_paths OUTSIDE transaction.
    if !batch.delete_paths.is_empty() {
        let conn = backend.conn_or_err()?;
        DuckDbHnswBackend::delete_paths(conn, &batch.delete_paths, &backend.known_dims)?;
    }

    // Step 0b: Pre-delete chunks/embeddings for files being upserted, OUTSIDE transaction.
    {
        let conn = backend.conn_or_err()?;
        DuckDbHnswBackend::pre_delete_for_upsert(conn, batch, &backend.known_dims)?;
    }

    // Step 0c: Ensure embedding tables outside txn (Invariant 13).
    let (unique_dims, total_emb) = DuckDbHnswBackend::collect_dims_and_count(batch);
    {
        let conn = backend.conn_or_err()?;
        for &dims in &unique_dims {
            DuckDbHnswBackend::ensure_embedding_table_dims(conn, dims)?;
        }
    }
    backend.known_dims.extend(unique_dims.iter().copied());

    // Lazy VSS load — only when embeddings are actually present.
    if total_emb > 0 {
        backend.ensure_vss()?;
    }

    Ok(())
}

pub(super) fn write_batch_incremental(
    backend: &mut DuckDbHnswBackend,
    batch: &DbWriterBatch,
) -> Result<BatchResult, DbError> {
    // BEGIN + write inner.
    let batch_inner = {
        let conn = backend.conn_or_err()?;
        conn.execute_batch("BEGIN")?;

        match DuckDbHnswBackend::write_batch_inner(conn, batch, backend.config.insert_batch_size) {
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
        let insert_batch_size = backend.config.insert_batch_size;
        let conn = backend
            .conn
            .as_ref()
            .expect("conn is Some: open() succeeded and BEGIN passed");
        match DuckDbHnswBackend::insert_embeddings_txn(
            conn,
            batch,
            &embedding_pairs,
            insert_batch_size,
        ) {
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
        let conn = backend
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

/// Write all `batches` inside a single BEGIN/COMMIT, reducing checkpoint
/// frequency versus one commit per batch. `prepare_write` must already
/// have been called for each batch (per the trait's documented contract),
/// so all embedding tables this loop needs already exist and `known_dims`
/// is already up to date — this function only touches `conn`, never
/// `backend.known_dims` or other `&mut` backend state.
pub(super) fn write_batches_in_one_txn(
    backend: &mut DuckDbHnswBackend,
    batches: &[DbWriterBatch],
) -> Result<Vec<BatchResult>, DbError> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let insert_batch_size = backend.config.insert_batch_size;
    let conn = backend.conn_or_err()?;
    conn.execute_batch("BEGIN")?;

    let mut results = Vec::with_capacity(batches.len());
    for batch in batches {
        let batch_inner = match DuckDbHnswBackend::write_batch_inner(conn, batch, insert_batch_size)
        {
            Ok(inner) => inner,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        };
        let embeddings_written = match DuckDbHnswBackend::insert_embeddings_txn(
            conn,
            batch,
            &batch_inner.embedding_pairs,
            insert_batch_size,
        ) {
            Ok(n) => n,
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        };
        results.push(BatchResult {
            file_ids: batch_inner.file_ids,
            chunks_written: batch_inner.chunks_written,
            embeddings_written,
        });
    }

    if let Err(e) = conn.execute_batch("COMMIT") {
        let _ = conn.execute_batch("ROLLBACK");
        return Err(DbError::DuckDb(e));
    }

    Ok(results)
}

#[cfg(test)]
mod upsert_tests {
    use super::*;
    use crate::db::{DbBackend, DbConfig};

    #[test]
    fn test_upsert_file_with_known_id_skips_insert() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db").to_string_lossy().into_owned();
        let config = DbConfig {
            db_path,
            compaction_threshold: Some(0.3),
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
                skip_reason: None,
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
                skip_reason: None,
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
    fn test_upsert_file_with_stale_id_falls_back_to_path_lookup() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("test.db").to_string_lossy().into_owned();
        let config = DbConfig {
            db_path,
            compaction_threshold: Some(0.3),
            compaction_min_size_bytes: 52_428_800,
            insert_batch_size: 100,
        };
        let mut backend = DuckDbHnswBackend::new(config);
        backend.open().expect("open");

        // Write two distinct files so we have a real "someone else's id" to collide with.
        let batch1 = crate::types::DbWriterBatch {
            files: vec![
                crate::types::FileRecord {
                    existing_file_id: None,
                    path: "a.py".into(),
                    mtime: Some(1.0),
                    size_bytes: Some(100),
                    content_hash: Some("abc".into()),
                    language: Some("python".into()),
                    skip_reason: None,
                    chunks: vec![],
                },
                crate::types::FileRecord {
                    existing_file_id: None,
                    path: "b.py".into(),
                    mtime: Some(1.0),
                    size_bytes: Some(50),
                    content_hash: Some("xyz".into()),
                    language: Some("python".into()),
                    skip_reason: None,
                    chunks: vec![],
                },
            ],
            delete_paths: vec![],
        };
        let result1 = backend.write_batch(&batch1).expect("first write");
        let a_id = result1.file_ids[0];
        let b_id = result1.file_ids[1];

        // Simulate a stale diff snapshot: "b.py" is written carrying a's id (e.g. a
        // rename/delete race between the diff snapshot and this write). The fast path
        // must not blindly trust this and must not corrupt a's row.
        let batch2 = crate::types::DbWriterBatch {
            files: vec![crate::types::FileRecord {
                existing_file_id: Some(a_id),
                path: "b.py".into(),
                mtime: Some(2.0),
                size_bytes: Some(200),
                content_hash: Some("def".into()),
                language: Some("python".into()),
                skip_reason: None,
                chunks: vec![],
            }],
            delete_paths: vec![],
        };
        let result2 = backend.write_batch(&batch2).expect("second write");

        assert_eq!(
            result2.file_ids[0], b_id,
            "a mismatched (id, path) pair must fall back to the path-keyed row for b.py, \
             not silently report success against a's row"
        );

        let conn = backend.conn_or_err().expect("conn");
        let a_hash: String = conn
            .query_row("SELECT content_hash FROM files WHERE id = ?", [a_id], |r| {
                r.get(0)
            })
            .expect("a row must still exist untouched");
        assert_eq!(
            a_hash, "abc",
            "a's row must not have been overwritten by b's stale-id update"
        );

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count");
        assert_eq!(count, 2, "no phantom row should be created for b.py");
    }

    #[test]
    fn pre_delete_for_upsert_nulls_mtime_and_hash() {
        // Crash window after prepare_write / before insert: chunks are gone and
        // the file row must look dirty so the next incremental differ reprocesses.
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb");
        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(
            db_path.to_string_lossy().into_owned(),
        ));
        backend.open().expect("open");

        let result = backend
            .write_batch(&super::super::test_support::single_file_batch("a.py"))
            .expect("seed write");
        let file_id = result.file_ids[0];
        assert!(
            result.chunks_written > 0,
            "seed must insert at least one chunk"
        );

        let dirty = crate::types::DbWriterBatch {
            files: vec![crate::types::FileRecord {
                existing_file_id: Some(file_id),
                path: "a.py".into(),
                mtime: Some(1.0),
                size_bytes: Some(100),
                content_hash: Some("abc123".into()),
                language: Some("python".into()),
                skip_reason: None,
                chunks: vec![],
            }],
            delete_paths: vec![],
        };
        {
            let conn = backend.conn_or_err().expect("conn");
            DuckDbHnswBackend::pre_delete_for_upsert(conn, &dirty, &backend.known_dims)
                .expect("pre_delete");
        }

        let conn = backend.conn_or_err().expect("conn");
        let chunk_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .expect("chunk count");
        assert_eq!(chunk_count, 0, "pre_delete must remove chunks");

        let (mtime_is_null, hash_is_null): (bool, bool) = conn
            .query_row(
                "SELECT modified_time IS NULL, content_hash IS NULL FROM files WHERE id = ?",
                [file_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("file dirty flags");
        assert!(
            mtime_is_null,
            "modified_time must be NULL so differ reprocesses"
        );
        assert!(
            hash_is_null,
            "content_hash must be NULL so hash-confirm cannot skip"
        );
    }
}

#[cfg(test)]
mod insert_batch_size_tests {
    use super::*;
    use crate::db::DbBackend;

    #[test]
    fn chunks_persist_correctly_at_non_default_batch_size() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend = DuckDbHnswBackend::new(
            super::super::test_support::config_with_insert_batch_size(db_path, 3),
        );
        backend.open().expect("open");

        // 7 chunks, no embeddings — doesn't evenly divide the batch size of 3
        // (batches of 3, 3, 1), exercising insert_chunks_for_file's chunking.
        let batch = DbWriterBatch {
            files: vec![super::super::test_support::file_with_n_chunks(
                "a.py", 7, None,
            )],
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
        let mut backend = DuckDbHnswBackend::new(
            super::super::test_support::config_with_insert_batch_size(db_path, 3),
        );
        backend.open().expect("open");

        // 7 chunks each with a 4-dim embedding, in one file — doesn't evenly
        // divide the batch size of 3, exercising insert_embeddings_txn's
        // chunking (grouped by dims).
        let batch = DbWriterBatch {
            files: vec![super::super::test_support::file_with_n_chunks(
                "a.py",
                7,
                Some(4),
            )],
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

#[cfg(test)]
mod write_batches_in_one_txn_tests {
    use super::*;
    use crate::db::DbBackend;

    #[test]
    fn commits_all_batches_and_preserves_order() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(db_path));
        backend.open().expect("open");

        let batch_a = super::super::test_support::embedding_batch("a", 4, 2);
        let batch_b = super::super::test_support::embedding_batch("b", 4, 3);
        backend.prepare_write(&batch_a).expect("prepare a");
        backend.prepare_write(&batch_b).expect("prepare b");

        let results = backend
            .write_batches_in_one_txn(&[batch_a, batch_b])
            .expect("write batches in one txn");
        backend.close().expect("close");

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].file_ids.len(),
            2,
            "results[0] must correspond to batch_a (2 files), preserving input order"
        );
        assert_eq!(
            results[1].file_ids.len(),
            3,
            "results[1] must correspond to batch_b (3 files), preserving input order"
        );
        assert_eq!(results[0].chunks_written, 2);
        assert_eq!(results[1].chunks_written, 3);

        let conn = Connection::open(tmp.path().join("t.duckdb")).expect("reopen");
        let files_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count files");
        let chunks_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .expect("count chunks");
        let emb_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM embeddings_4", [], |r| r.get(0))
            .expect("count embeddings_4");
        assert_eq!(files_count, 5, "both batches' files must persist");
        assert_eq!(chunks_count, 5, "both batches' chunks must persist");
        assert_eq!(emb_count, 5, "both batches' embeddings must persist");
    }

    #[test]
    fn rolls_back_entire_window_on_mid_window_failure() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(db_path));
        backend.open().expect("open");

        let batch0 = super::super::test_support::single_file_batch("a.py");
        // Deliberately built with an 8-dim embedding but prepare_write is
        // never called for it below, so embeddings_8 is never created —
        // insert_embeddings_txn's INSERT into it will fail with a real
        // DuckDB catalog error, deterministically forcing a mid-window
        // failure without any test-only hooks.
        let batch1 = super::super::test_support::embedding_batch("b", 8, 1);
        let batch2 = super::super::test_support::single_file_batch("c.py");

        backend.prepare_write(&batch0).expect("prepare batch0");
        backend.prepare_write(&batch2).expect("prepare batch2");

        let result = backend.write_batches_in_one_txn(&[batch0, batch1, batch2]);
        assert!(
            result.is_err(),
            "missing embeddings_8 table must surface as an error"
        );

        let conn = backend.conn_or_err().expect("conn");
        let files_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count files");
        let chunks_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .expect("count chunks");
        assert_eq!(
            files_count, 0,
            "batch0's already-written file row must be rolled back with the rest of the window"
        );
        assert_eq!(
            chunks_count, 0,
            "batch0's already-written chunk must be rolled back with the rest of the window"
        );

        // Sanity check: the manual ROLLBACK must leave the connection usable,
        // not stuck inside a broken transaction.
        let post_rollback = super::super::test_support::single_file_batch("d.py");
        backend
            .write_batch(&post_rollback)
            .expect("connection must remain usable after rollback");
    }

    #[test]
    fn empty_slice_returns_empty_vec() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let db_path = tmp.path().join("t.duckdb").to_string_lossy().into_owned();
        let mut backend = DuckDbHnswBackend::new(super::super::test_support::config(db_path));
        backend.open().expect("open");

        let results = backend
            .write_batches_in_one_txn(&[])
            .expect("empty slice must not error");
        assert!(results.is_empty());

        let conn = backend.conn_or_err().expect("conn");
        let files_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .expect("count files");
        assert_eq!(files_count, 0, "empty slice must not touch the DB");
    }
}
