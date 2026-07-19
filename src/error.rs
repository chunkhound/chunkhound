use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("duckdb: {0}")]
    DuckDb(#[from] duckdb::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

impl From<DbError> for PyErr {
    fn from(e: DbError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
