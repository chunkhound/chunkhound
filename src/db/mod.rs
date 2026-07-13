use crate::error::DbError;
use crate::types::{BatchResult, DbWriterBatch};

pub mod duckdb_backend;
pub use duckdb_backend::DuckDbHnswBackend;

pub trait DbBackend: Send {
    fn open(&mut self) -> Result<(), DbError>;
    fn close(&mut self) -> Result<(), DbError>;
    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError>;
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
    pub compaction_batch_threshold: u32,
}


pub fn create_backend(cfg: DbConfig) -> Box<dyn DbBackend> {
    Box::new(DuckDbHnswBackend::new(cfg))
}
