//! Pipeline configuration — extracted from a Python dict at construction time.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Extract an optional field from a PyDict — returns None if key is absent or value is Python None.
fn extract_opt<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract()?)),
    }
}

/// Extract a field from a PyDict, falling back to `default` if the key is absent.
fn extract_or<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    default: T,
) -> PyResult<T> {
    match dict.get_item(key)? {
        Some(v) => v.extract(),
        None => Ok(default),
    }
}

/// Parsing-tuning flags pass through to the parse callback unchanged.
#[derive(Debug, Clone)]
pub(crate) struct PipelineConfig {
    // Project root (for relative path computation, like Python's _get_relative_path).
    pub project_root: PathBuf,

    // Storage
    pub db_path: PathBuf,
    pub db_batch_size: usize,
    pub compaction_batch_threshold: u32,
    pub compaction_threshold: f64,
    pub compaction_min_size_mb: u64,
    pub disk_usage_limit_mb: Option<f64>,

    // Pipeline parallelism
    pub parse_batch_size: usize,
    pub parse_thread_pool_size: usize,
    pub embed_thread_pool_size: usize,
    pub embed_batch_size: usize,

    // Change detection
    pub mtime_epsilon_seconds: f64,

    // Orphan cleanup (mirrors config.indexing.cleanup on the Python side)
    pub do_cleanup: bool,

    // Feature toggles
    pub skip_embeddings: bool,

    // Pass-through (parse callback)
    pub per_file_timeout_secs: f64,
    pub per_file_timeout_min_size_kb: u32,
    pub detect_embedded_sql: bool,
    pub config_file_size_threshold_kb: u32,

    // Pass-through (embed callback)
    pub embedding_provider: String,
    pub embedding_model: String,
}

impl PipelineConfig {
    /// Extract configuration from a Python dict.
    pub fn from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            project_root: extract_or(dict, "project_root", String::new())?.into(),
            db_path: extract_or(dict, "db_path", String::new())?.into(),
            db_batch_size: extract_or(dict, "db_batch_size", 100u64)? as usize,
            compaction_batch_threshold: extract_or(dict, "compaction_batch_threshold", 50u64)?
                as u32,
            compaction_threshold: extract_or(dict, "compaction_threshold", 0.30)?,
            compaction_min_size_mb: extract_or(dict, "compaction_min_size_mb", 50u64)?,
            disk_usage_limit_mb: extract_opt(dict, "disk_usage_limit_mb")?,

            parse_batch_size: extract_or(dict, "parse_batch_size", 200u64)? as usize,
            parse_thread_pool_size: extract_or(dict, "parse_thread_pool_size", 0u64)? as usize,
            embed_thread_pool_size: extract_or(dict, "embed_thread_pool_size", 0u64)? as usize,
            embed_batch_size: extract_or(dict, "embed_batch_size", 200u64)? as usize,

            mtime_epsilon_seconds: extract_or(dict, "mtime_epsilon_seconds", 0.01)?,
            do_cleanup: extract_or(dict, "do_cleanup", true)?,
            skip_embeddings: extract_or(dict, "skip_embeddings", false)?,

            per_file_timeout_secs: extract_or(dict, "per_file_timeout_secs", 3.0)?,
            per_file_timeout_min_size_kb: extract_or(dict, "per_file_timeout_min_size_kb", 128u64)?
                as u32,
            detect_embedded_sql: extract_or(dict, "detect_embedded_sql", true)?,
            config_file_size_threshold_kb: extract_or(dict, "config_file_size_threshold_kb", 20u64)?
                as u32,

            embedding_provider: extract_or(dict, "embedding_provider", String::new())?,
            embedding_model: extract_or(dict, "embedding_model", String::new())?,
        })
    }
}
