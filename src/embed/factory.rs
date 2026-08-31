use super::{EmbedBatchFn, PythonEmbedCallback};
use crate::error::PipelineError;
use pyo3::prelude::*;
use std::fmt;

#[derive(Clone)]
pub(crate) struct EmbedConfig {
    pub provider: String,
    pub model: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub output_dims: Option<usize>,
    pub matryoshka: bool,
    pub client_side_truncation: bool,
    pub api_version: Option<String>,
    pub ssl_verify: bool,
    pub is_azure: bool,
    pub azure_endpoint: Option<String>,
    pub azure_deployment: Option<String>,
    pub max_tokens_per_batch: usize,
    pub max_items_per_batch: usize,
}

impl fmt::Debug for EmbedConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbedConfig")
            .field("provider", &self.provider)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .field("base_url", &self.base_url)
            .field("output_dims", &self.output_dims)
            .field("matryoshka", &self.matryoshka)
            .field("client_side_truncation", &self.client_side_truncation)
            .field("api_version", &self.api_version)
            .field("ssl_verify", &self.ssl_verify)
            .field("is_azure", &self.is_azure)
            .field("azure_endpoint", &self.azure_endpoint)
            .field("azure_deployment", &self.azure_deployment)
            .field("max_tokens_per_batch", &self.max_tokens_per_batch)
            .field("max_items_per_batch", &self.max_items_per_batch)
            .finish()
    }
}

pub(crate) fn create_embed_fn(
    config: &EmbedConfig,
    callback: Option<Py<PyAny>>,
) -> Result<Box<dyn EmbedBatchFn>, String> {
    if config.is_azure != config.azure_endpoint.is_some() {
        return Err(PipelineError::BadRequest(
            "Azure embedding configuration is inconsistent".to_string(),
        )
        .to_string());
    }
    if config.is_azure && config.base_url.is_some() {
        return Err(PipelineError::BadRequest(
            "Azure embeddings cannot also set base_url".to_string(),
        )
        .to_string());
    }
    if config.is_azure && (config.api_version.is_none() || config.api_key.is_none()) {
        return Err(PipelineError::BadRequest(
            "Azure embeddings require api_version and api_key".to_string(),
        )
        .to_string());
    }

    match config.provider.as_str() {
        "openai" => Ok(Box::new(super::OpenAiProvider::new(config.clone())?)),
        "voyageai" => Ok(Box::new(super::VoyageAiProvider::new(config.clone())?)),
        _ => callback
            .map(|value| Box::new(PythonEmbedCallback::new(value)) as Box<dyn EmbedBatchFn>)
            .ok_or_else(|| {
                PipelineError::BadRequest("unknown embedding provider".to_string()).to_string()
            }),
    }
}
