use crate::error::DbError;
use crate::types::{BatchResult, DbWriterBatch};

pub mod duckdb_backend;
pub use duckdb_backend::DuckDbHnswBackend;

pub trait DbBackend: Send {
    fn open(&mut self) -> Result<(), DbError>;
    fn close(&mut self) -> Result<(), DbError>;
    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError>;

    /// Pipeline parallelism: phase 0 — pre-deletes, embed-table setup, HNSW drop.
    /// Runs OUTSIDE any write transaction.  Takes an aggregated batch containing
    /// all files and delete_paths for the entire indexing run so HNSW lifecycle
    /// decisions see the full picture.
    fn prepare_write(&mut self, batch: &DbWriterBatch) -> Result<(), DbError> {
        let _ = batch;
        Ok(())
    }

    /// Pipeline parallelism: phase 1 — write ONE batch inside its own transaction.
    /// BEGIN → upsert files + insert chunks + insert embeddings → COMMIT.
    fn write_batch_incremental(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
        self.write_batch(batch)
    }

    /// Pipeline parallelism: phase 2 — recreate HNSW indexes + final CHECKPOINT.
    fn finish_write(&mut self) -> Result<(), DbError> {
        Ok(())
    }

    fn needs_compaction(&self) -> Result<bool, DbError>;
    fn run_compaction(&mut self) -> Result<(), DbError>;
    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        Ok(())
    }
    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DbConfig {
    pub db_path: String,
    /// Legacy simple-threshold for fallback; prefer two-signal detection.
    pub compaction_batch_threshold: u32,
    /// Threshold for effective_waste = max(free_ratio, row_waste_ratio).
    /// Default: 0.30 (30% of DB space is reclaimable).
    pub compaction_threshold: f64,
    /// Minimum reclaimable bytes required before compaction triggers.
    /// Default: 52428800 (50 MB).
    pub compaction_min_size_bytes: u64,
}

pub fn create_backend(cfg: DbConfig) -> Box<dyn DbBackend> {
    Box::new(DuckDbHnswBackend::new(cfg))
}
