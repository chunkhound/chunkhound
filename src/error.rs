use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

/// Errors raised while obtaining an embedding.  These are deliberately
/// separate from database/scan errors because provider failures are attached
/// to individual chunks by the streaming pipeline.
#[derive(Debug, Clone, thiserror::Error)]
pub enum PipelineError {
    #[error("embedding authentication failed")]
    Auth,
    #[error("embedding request rejected: {0}")]
    BadRequest(String),
    #[error("embedding request cancelled")]
    Cancelled,
    #[error("embedding provider error: {0}")]
    ProviderError(String),
    #[error("embedding rate limited")]
    RateLimited { retry_after_secs: Option<u64> },
    #[allow(dead_code)] // Reserved for a future provider-backed vector store.
    #[error("embedding database error: {0}")]
    DbError(String),
    #[error("embedding IO error: {0}")]
    IoError(String),
    #[error("embedding input exceeds the provider context limit")]
    ContextLengthExceeded,
    #[error("embedding provider returned an invalid response: {0}")]
    ResponseFormat(String),
}

impl PipelineError {
    /// Whether the error is non-retryable. This does not mean that the whole
    /// indexing run aborts: the current pipeline intentionally records the
    /// error and stores the affected chunks without embeddings.
    #[allow(dead_code)] // Kept as a classification API; pipeline abort is deliberate policy.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Self::Auth | Self::BadRequest(_) | Self::Cancelled | Self::ResponseFormat(_)
        )
    }

    pub fn retry_after_secs(&self) -> Option<u64> {
        match self {
            Self::RateLimited { retry_after_secs } => *retry_after_secs,
            _ => None,
        }
    }
}

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
