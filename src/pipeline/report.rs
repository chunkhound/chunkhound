//! Pipeline report — returned to Python after indexing completes.

use pyo3::prelude::*;

/// Exposed to Python as the result of `IndexingPipeline.run()`.
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct PipelineReport {
    #[pyo3(get)]
    pub files_processed: u64,

    #[pyo3(get)]
    pub files_skipped: u64,

    #[pyo3(get)]
    pub chunks_written: u64,

    #[pyo3(get)]
    pub embeddings_generated: u64,

    #[pyo3(get)]
    pub elapsed_secs: f64,

    #[pyo3(get)]
    pub errors: Vec<String>,

    #[pyo3(get)]
    pub peak_rss_mb: Option<f64>,
}

impl PipelineReport {
    pub fn empty() -> Self {
        Self {
            files_processed: 0,
            files_skipped: 0,
            chunks_written: 0,
            embeddings_generated: 0,
            elapsed_secs: 0.0,
            errors: Vec::new(),
            peak_rss_mb: None,
        }
    }
}