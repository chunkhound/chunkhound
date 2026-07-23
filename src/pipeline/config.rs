//! Pipeline configuration — extracted from a Python dict at construction time.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

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
    pub pipeline_parallel: bool,

    // Change detection
    pub force_reindex: bool,
    pub mtime_epsilon_seconds: f64,

    // Orphan cleanup
    pub skip_cleanup: bool,

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
            project_root: get_str_or(dict, "project_root", "")?.into(),
            db_path: get_str_or(dict, "db_path", "")?.into(),
            db_batch_size: get_u64_or(dict, "db_batch_size", 100)? as usize,
            compaction_batch_threshold: get_u64_or(dict, "compaction_batch_threshold", 50)? as u32,
            compaction_threshold: get_f64_or(dict, "compaction_threshold", 0.30)?,
            compaction_min_size_mb: get_u64_or(dict, "compaction_min_size_mb", 50)?,
            disk_usage_limit_mb: get_opt_f64(dict, "disk_usage_limit_mb")?,

            parse_batch_size: get_u64_or(dict, "parse_batch_size", 200)? as usize,
            parse_thread_pool_size: get_u64_or(dict, "parse_thread_pool_size", 0)? as usize,
            embed_thread_pool_size: get_u64_or(dict, "embed_thread_pool_size", 0)? as usize,
            embed_batch_size: get_u64_or(dict, "embed_batch_size", 200)? as usize,
            pipeline_parallel: get_bool_or(dict, "pipeline_parallel", false)?,

            force_reindex: get_bool_or(dict, "force_reindex", false)?,
            mtime_epsilon_seconds: get_f64_or(dict, "mtime_epsilon_seconds", 0.01)?,
            skip_cleanup: get_bool_or(dict, "skip_cleanup", false)?,
            skip_embeddings: get_bool_or(dict, "skip_embeddings", false)?,

            per_file_timeout_secs: get_f64_or(dict, "per_file_timeout_secs", 3.0)?,
            per_file_timeout_min_size_kb: get_u64_or(dict, "per_file_timeout_min_size_kb", 128)?
                as u32,
            detect_embedded_sql: get_bool_or(dict, "detect_embedded_sql", true)?,
            config_file_size_threshold_kb: get_u64_or(dict, "config_file_size_threshold_kb", 20)?
                as u32,

            embedding_provider: get_str_or(dict, "embedding_provider", "")?,
            embedding_model: get_str_or(dict, "embedding_model", "")?,
        })
    }
}

// ── Helper extractors ────────────────────────────────────────────

fn get_str_or(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> PyResult<String> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<String>()?),
        None => Ok(default.to_string()),
    }
}

fn get_u64_or(dict: &Bound<'_, PyDict>, key: &str, default: u64) -> PyResult<u64> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<u64>()?),
        None => Ok(default),
    }
}

fn get_f64_or(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> PyResult<f64> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<f64>()?),
        None => Ok(default),
    }
}

fn get_bool_or(dict: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match dict.get_item(key)? {
        Some(v) => Ok(v.extract::<bool>()?),
        None => Ok(default),
    }
}

fn get_opt_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    match dict.get_item(key)? {
        Some(v) => {
            if v.is_none() {
                Ok(None)
            } else {
                Ok(Some(v.extract::<f64>()?))
            }
        }
        None => Ok(None),
    }
}
