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

#[derive(Debug, thiserror::Error)]
pub enum ScanError {
    #[error("root '{root}' does not exist or is not a readable directory: {source}")]
    RootUnreadable {
        root: String,
        source: std::io::Error,
    },
    #[error(
        "scan of '{root}' hit {count} walk error(s) and found zero files (e.g. {example}) \
         -- refusing to report this as an empty project, since that would be \
         indistinguishable from every file having been deleted"
    )]
    Incomplete {
        root: String,
        count: usize,
        example: String,
    },
}

impl From<ScanError> for PyErr {
    fn from(e: ScanError) -> PyErr {
        PyRuntimeError::new_err(e.to_string())
    }
}
