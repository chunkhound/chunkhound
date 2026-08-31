//! Pipeline configuration — extracted from a Python dict at construction time.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Secret wrapper whose derived/debugged parent structs cannot print the key.
#[derive(Clone)]
pub(crate) struct ApiKey(String);

impl ApiKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

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

/// Extract an optional field, but only defaults when the key is absent.
/// An explicitly-passed Python `None` is preserved as `None` (a real
/// "disabled" value) rather than being coerced to `default` — unlike
/// `extract_or`, which treats "absent" and "explicit None" the same way.
fn extract_opt_or_default<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
    default: T,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        None => Ok(Some(default)),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract()?)),
    }
}

/// Parsing-tuning flags pass through to the parse callback unchanged.
#[derive(Debug, Clone)]
pub(crate) struct PipelineConfig {
    // Storage
    pub db_path: PathBuf,
    pub db_batch_size: usize,
    /// `None` means auto-compaction is disabled (mirrors Python's
    /// `DatabaseConfig.fragmentation_threshold_pct = None`).
    pub compaction_threshold: Option<f64>,
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
    pub embedding_api_key: Option<ApiKey>,
    pub embedding_base_url: Option<String>,
    pub embedding_output_dims: Option<usize>,
    pub embedding_matryoshka: bool,
    pub embedding_client_side_truncation: bool,
    pub embedding_api_version: Option<String>,
    pub embedding_ssl_verify: bool,
    pub embedding_is_azure: bool,
    pub embedding_azure_endpoint: Option<String>,
    pub embedding_azure_deployment: Option<String>,
    pub embed_max_tokens_per_batch: usize,
}

impl PipelineConfig {
    /// Extract configuration from a Python dict.
    pub fn from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            db_path: extract_or(dict, "db_path", String::new())?.into(),
            db_batch_size: extract_or(dict, "db_batch_size", 100u64)? as usize,
            compaction_threshold: extract_opt_or_default(dict, "compaction_threshold", 0.30)?,
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
            embedding_api_key: extract_opt::<String>(dict, "embedding_api_key")?.map(ApiKey),
            embedding_base_url: extract_opt(dict, "embedding_base_url")?,
            embedding_output_dims: extract_opt(dict, "embedding_output_dims")?,
            embedding_matryoshka: extract_or(dict, "embedding_matryoshka", false)?,
            embedding_client_side_truncation: extract_or(
                dict,
                "embedding_client_side_truncation",
                false,
            )?,
            embedding_api_version: extract_opt(dict, "embedding_api_version")?,
            embedding_ssl_verify: extract_or(dict, "embedding_ssl_verify", true)?,
            embedding_is_azure: extract_or(dict, "embedding_is_azure", false)?,
            embedding_azure_endpoint: extract_opt(dict, "embedding_azure_endpoint")?,
            embedding_azure_deployment: extract_opt(dict, "embedding_azure_deployment")?,
            embed_max_tokens_per_batch: extract_or(dict, "embed_max_tokens_per_batch", 8191u64)?
                .max(1) as usize,
        })
    }

    pub fn embed_config(&self) -> crate::embed::EmbedConfig {
        crate::embed::EmbedConfig {
            provider: self.embedding_provider.clone(),
            model: self.embedding_model.clone(),
            api_key: self
                .embedding_api_key
                .as_ref()
                .map(|key| key.as_str().to_string()),
            base_url: self.embedding_base_url.clone(),
            output_dims: self.embedding_output_dims,
            matryoshka: self.embedding_matryoshka,
            client_side_truncation: self.embedding_client_side_truncation,
            api_version: self.embedding_api_version.clone(),
            ssl_verify: self.embedding_ssl_verify,
            is_azure: self.embedding_is_azure,
            azure_endpoint: self.embedding_azure_endpoint.clone(),
            azure_deployment: self.embedding_azure_deployment.clone(),
            max_tokens_per_batch: self.embed_max_tokens_per_batch,
            max_items_per_batch: self.embed_batch_size.max(1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_redacts_embedding_api_key() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("embedding_api_key", "secret-test-key")
                .expect("set key");
            let config = PipelineConfig::from_py_dict(&dict).expect("config");
            let debug = format!("{config:?}");
            assert!(!debug.contains("secret-test-key"));
            assert!(debug.contains("REDACTED"));
        });
    }
}
