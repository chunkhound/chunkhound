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

    /// Parse-time skips that are not errors: ``(path, skip_reason)``.
    /// Timeouts stay in ``errors`` so the coordinator can split them out.
    #[pyo3(get)]
    pub skipped_paths: Vec<(String, String)>,

    #[pyo3(get)]
    pub peak_rss_mb: Option<f64>,

    /// Mirrors Python's `DiskUsageLimitExceededError` contract: set to
    /// `Some((current_mb, limit_mb))` when a mid-run disk-usage check (see
    /// `db::check_disk_usage_limit`) tripped and the store thread stopped
    /// further writes; `None` otherwise. Reported as data, not a raised
    /// exception, matching `_check_disk_usage_limit`'s own contract. A
    /// single `Option<(f64, f64)>` rather than a `bool` plus two
    /// independent `Option<f64>` fields — the tripped/not-tripped state and
    /// its two numbers can't desync into an inconsistent combination this way.
    #[pyo3(get)]
    pub disk_limit: Option<(f64, f64)>,
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
            skipped_paths: Vec::new(),
            peak_rss_mb: None,
            disk_limit: None,
        }
    }
}
