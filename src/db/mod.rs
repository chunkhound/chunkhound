use crate::error::DbError;
use crate::types::{BatchResult, DbWriterBatch};

pub mod duckdb_backend;
pub use duckdb_backend::DuckDbHnswBackend;

/// A single chunk row returned by read_chunks.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    pub code: String,
}

pub trait DbBackend: Send {
    fn open(&mut self) -> Result<(), DbError>;
    fn close(&mut self) -> Result<(), DbError>;
    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError>;
    fn read_chunks(&mut self) -> Result<Vec<ChunkRow>, DbError>;
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
