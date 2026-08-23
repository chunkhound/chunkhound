use crate::error::DbError;
use crate::types::{BatchResult, DbFileEntry, DbWriterBatch};

pub mod duckdb_backend;
pub(crate) use duckdb_backend::check_disk_usage_limit;
pub use duckdb_backend::DuckDbHnswBackend;

pub trait DbBackend: Send {
    fn open(&mut self) -> Result<(), DbError>;
    fn close(&mut self) -> Result<(), DbError>;
    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError>;

    /// Pipeline parallelism: phase 0 — pre-deletes, embed-table setup.
    /// Runs OUTSIDE any write transaction. HNSW drop/rebuild around the whole
    /// run is handled separately via `drop_all_hnsw_indexes`/`ensure_all_hnsw_indexes`.
    fn prepare_write(&mut self, batch: &DbWriterBatch) -> Result<(), DbError> {
        let _ = batch;
        Ok(())
    }

    /// Pipeline parallelism: phase 1 — write ONE batch inside its own transaction.
    /// BEGIN → upsert files + insert chunks + insert embeddings → COMMIT.
    fn write_batch_incremental(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        self.write_batch(batch)
    }

    /// Write multiple pre-prepared batches in a single transaction.
    ///
    /// `prepare_write` must have been called for each batch already (in
    /// auto-commit mode, as usual).  The default implementation calls
    /// `write_batch_incremental` for each batch, preserving per-batch commit
    /// behaviour.  Override to reduce checkpoint frequency by committing N
    /// batches at once.
    fn write_batches_in_one_txn(
        &mut self,
        batches: &[DbWriterBatch],
    ) -> Result<Vec<BatchResult>, DbError> {
        batches
            .iter()
            .map(|b| self.write_batch_incremental(b))
            .collect()
    }

    fn needs_compaction(&self) -> Result<bool, DbError>;
    fn run_compaction(&mut self) -> Result<(), DbError>;
    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        Ok(())
    }
    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        Ok(())
    }

    /// Snapshot every row of the `files` table for the pipeline's diff phase.
    /// Default: no-op (empty index / non-DuckDB backends treat every file as
    /// new). Deliberately `&self`, not `&mut self`, and does NOT go through
    /// `open()`'s heavier lifecycle (crash recovery, VSS load,
    /// `ensure_all_hnsw_indexes()`) -- callers may invoke this on a backend
    /// that was only just `create_backend()`'d and never `.open()`ed.
    fn read_file_states(&self) -> Result<Vec<DbFileEntry>, DbError> {
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone)]
pub struct DbConfig {
    pub db_path: String,
    /// Threshold for effective_waste = max(free_ratio, row_waste_ratio).
    /// Default: 0.30 (30% of DB space is reclaimable). `None` disables
    /// auto-compaction entirely (mirrors Python's
    /// `fragmentation_threshold_pct = None` opt-out).
    pub compaction_threshold: Option<f64>,
    /// Minimum reclaimable bytes required before compaction triggers.
    /// Default: 52428800 (50 MB).
    pub compaction_min_size_bytes: u64,
    /// Rows per INSERT statement for chunk/embedding writes (mirrors Python's
    /// `indexing.db_batch_size`). Must be >= 1 — callers should clamp before
    /// constructing this struct, since `slice::chunks(0)` panics.
    pub insert_batch_size: usize,
}

pub fn create_backend(cfg: DbConfig) -> Box<dyn DbBackend> {
    Box::new(DuckDbHnswBackend::new(cfg))
}
