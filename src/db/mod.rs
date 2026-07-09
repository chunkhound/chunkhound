use crate::error::DbError;
use crate::types::{BatchResult, DbWriterBatch};

pub mod duckdb_backend;
pub use duckdb_backend::DuckDbHnswBackend;

pub trait DbBackend: Send {
    fn open(&mut self) -> Result<(), DbError>;
    fn close(&mut self) -> Result<(), DbError>;
    fn write_batch(&mut self, batch: &DbWriterBatch) -> Result<BatchResult, DbError>;
    fn needs_compaction(&mut self) -> Result<bool, DbError>;
    fn run_compaction(&mut self) -> Result<(), DbError>;
    fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> { Ok(()) }
    fn ensure_all_hnsw_indexes(&mut self) -> Result<(), DbError> { Ok(()) }
}

#[derive(Debug, Clone)]
pub struct DbConfig {
    pub db_path: String,
    pub compaction_batch_threshold: u32,
}

impl DbConfig {
    pub fn from_json_value(v: &serde_json::Value) -> Result<Self, DbError> {
        let db_path = v
            .get("db_path")
            .and_then(|x| x.as_str())
            .ok_or_else(|| DbError::Other("db_path required in db_config".into()))?
            .to_string();
        let compaction_batch_threshold = v
            .get("compaction_batch_threshold")
            .and_then(|x| x.as_u64())
            .unwrap_or(50) as u32;
        Ok(DbConfig {
            db_path,
            compaction_batch_threshold,
        })
    }
}

pub fn create_backend(cfg: DbConfig) -> Box<dyn DbBackend> {
    Box::new(DuckDbHnswBackend::new(cfg))
}
