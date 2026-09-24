use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use duckdb::Connection;

use crate::db::DbConfig;
use crate::error::DbError;
use crate::types::{BatchResult, DbFileEntry, DbWriterBatch};

mod compaction;
mod hnsw;
mod read;
mod recovery;
mod schema;
mod write;

pub(crate) use read::check_disk_usage_limit;

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
    // Test seam: fail drop_all_hnsw_indexes after this many successful DROPs
    // so we can assert close() still restores HNSW after a partial drop.
    #[cfg(test)]
    fail_drop_after: Option<usize>,
}

impl DuckDbHnswBackend {
    pub fn new(config: DbConfig) -> Self {
        DuckDbHnswBackend {
            config,
            conn: None,
            has_vss: false,
            hnsw_bulk_mode: false,
            known_dims: HashSet::new(),
            saved_hnsw_metrics: HashMap::new(),
            #[cfg(test)]
            fail_drop_after: None,
        }
    }

    fn conn_or_err(&self) -> Result<&Connection, DbError> {
        self.conn
            .as_ref()
            .ok_or_else(|| DbError::Other("not open".into()))
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
        let recovered = Self::recover_swap_intent(&db_path)?;
        Self::discard_incomplete_compact_if_phase1(recovered, &db_path);

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
        write::prepare_write(self, batch)
    }

    fn write_batch_incremental(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        write::write_batch_incremental(self, batch)
    }

    fn write_batches_in_one_txn(
        &mut self,
        batches: &[DbWriterBatch],
    ) -> Result<Vec<BatchResult>, DbError> {
        write::write_batches_in_one_txn(self, batches)
    }

    fn needs_compaction(&self) -> Result<bool, DbError> {
        compaction::needs_compaction(self)
    }

    fn run_compaction(&mut self) -> Result<(), DbError> {
        compaction::run_compaction(self)
    }

    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        hnsw::drop_all_hnsw_indexes(self)
    }

    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        hnsw::ensure_all_hnsw_indexes(self)
    }

    fn read_file_states(&self) -> Result<Vec<DbFileEntry>, DbError> {
        read::read_file_states(self)
    }
}

/// Shared test fixtures used by the test modules in this crate's sibling
/// submodules (`recovery`, `read`, `hnsw`, `write`, `compaction`), reached via
/// `super::super::test_support` from a nested test module.
#[cfg(test)]
mod test_support {
    use super::*;
    use crate::types::{ChunkRecord, FileRecord};

    pub(super) fn config(db_path: String) -> DbConfig {
        DbConfig {
            db_path,
            compaction_threshold: Some(0.30),
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

    pub(super) fn config_with_compaction_threshold(
        db_path: String,
        compaction_threshold: Option<f64>,
    ) -> DbConfig {
        DbConfig {
            compaction_threshold,
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
            skip_reason: None,
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
            skip_reason: None,
            chunks: (0..chunk_count)
                .map(|i| chunk_record(&format!("def foo_{i}(): pass"), embedding_dims))
                .collect(),
        }
    }
}
