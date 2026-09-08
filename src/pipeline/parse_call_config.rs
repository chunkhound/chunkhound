//! Per-call configuration handed to the Python parse callback.

use pyo3::prelude::*;

/// Bundles the config values `parse_batch_callback()` needs on the Python
/// side. Constructed once per `IndexingPipeline.run()` call from
/// `PipelineConfig` and passed to every `parse_batch_callback()` invocation
/// for that run — a single typed object instead of a growing list of
/// positional arguments, so adding a new field later doesn't require another
/// call-arity change (and the test-local callback stand-ins that mirror this
/// call's shape don't all need updating every time either).
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) struct ParseCallConfig {
    #[pyo3(get)]
    pub detect_embedded_sql: bool,

    #[pyo3(get)]
    pub per_file_timeout_secs: f64,

    #[pyo3(get)]
    pub per_file_timeout_min_size_kb: u32,

    #[pyo3(get)]
    pub config_file_size_threshold_kb: u32,

    #[pyo3(get)]
    pub parse_thread_pool_size: usize,
}

#[pymethods]
impl ParseCallConfig {
    #[new]
    #[pyo3(signature = (
        detect_embedded_sql=true,
        per_file_timeout_secs=3.0,
        per_file_timeout_min_size_kb=128,
        config_file_size_threshold_kb=20,
        parse_thread_pool_size=0,
    ))]
    fn new(
        detect_embedded_sql: bool,
        per_file_timeout_secs: f64,
        per_file_timeout_min_size_kb: u32,
        config_file_size_threshold_kb: u32,
        parse_thread_pool_size: usize,
    ) -> Self {
        Self {
            detect_embedded_sql,
            per_file_timeout_secs,
            per_file_timeout_min_size_kb,
            config_file_size_threshold_kb,
            parse_thread_pool_size,
        }
    }
}
