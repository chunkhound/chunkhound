use super::common::{sanitize, HttpClientPool};
use super::factory::EmbedConfig;
use super::{EmbedBatchFn, EmbedBatchResult};
use crate::error::PipelineError;
use reqwest::blocking::Response;
use serde::Deserialize;
use std::sync::atomic::AtomicUsize;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Deserialize)]
struct OpenAiResponse {
    data: Vec<OpenAiEmbedding>,
}

#[derive(Deserialize)]
struct OpenAiEmbedding {
    index: usize,
    embedding: Vec<f64>,
}

pub(crate) struct OpenAiProvider {
    config: EmbedConfig,
    pool: HttpClientPool,
    // First-observed embedding dimension for this provider instance, shared
    // across every concurrent `embed_batch` call (one per rayon sub-batch).
    // 0 means "not yet established" -- real embedding dimensions are always
    // > 0, so 0 is safe to use as an unset sentinel.
    observed_dims: AtomicUsize,
}

impl OpenAiProvider {
    pub fn new(config: EmbedConfig) -> Result<Self, String> {
        if config.model.is_empty() {
            return Err(
                PipelineError::BadRequest("embedding model is empty".to_string()).to_string(),
            );
        }
        let pool = HttpClientPool::new(config.ssl_verify);
        Ok(Self {
            config,
            pool,
            observed_dims: AtomicUsize::new(0),
        })
    }

    fn url(&self) -> Result<String, PipelineError> {
        if self.config.is_azure {
            let endpoint = self.config.azure_endpoint.as_deref().ok_or_else(|| {
                PipelineError::BadRequest("Azure endpoint is missing".to_string())
            })?;
            let deployment = self
                .config
                .azure_deployment
                .as_deref()
                .unwrap_or(&self.config.model);
            let version = self.config.api_version.as_deref().ok_or_else(|| {
                PipelineError::BadRequest("Azure API version is missing".to_string())
            })?;
            return Ok(format!(
                "{}/openai/deployments/{}/embeddings?api-version={}",
                endpoint.trim_end_matches('/'),
                encode_path_segment(deployment)?,
                encode_query_value(version)?
            ));
        }
        Ok(format!(
            "{}/embeddings",
            self.config
                .base_url
                .as_deref()
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
        ))
    }

    fn request_once(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PipelineError> {
        let url = self.url()?;
        let mut body = serde_json::json!({
            "model": if self.config.is_azure {
                self.config.azure_deployment.as_deref().unwrap_or(&self.config.model)
            } else {
                &self.config.model
            },
            "input": texts,
        });
        let can_send_dimensions = self.config.output_dims.is_some()
            && !self.config.client_side_truncation
            && (self.config.matryoshka || self.config.is_azure || self.config.base_url.is_some());
        if can_send_dimensions {
            body["dimensions"] = serde_json::json!(self.config.output_dims);
        }

        self.pool.with_client(|client| {
            let mut request = client.post(url).json(&body);
            if let Some(key) = self.config.api_key.as_deref().filter(|k| !k.is_empty()) {
                if self.config.is_azure {
                    request = request.header("api-key", key);
                } else {
                    request = request.bearer_auth(key);
                }
            }
            let response = request.send().map_err(|e| {
                PipelineError::IoError(sanitize(e.to_string(), self.config.api_key.as_deref()))
            })?;
            parse_response(response, texts.len(), self.config.api_key.as_deref())
        })
    }

    fn request_with_retry(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PipelineError> {
        super::common::default_retry(|| self.request_once(texts))
    }
}

impl EmbedBatchFn for OpenAiProvider {
    fn embed_batch(&self, texts: &[String]) -> Result<EmbedBatchResult, String> {
        super::common::run_embed_batch(texts, &self.config, &self.observed_dims, |batch| {
            self.request_with_retry(batch)
        })
    }
}

fn parse_response(
    response: Response,
    expected: usize,
    secret: Option<&str>,
) -> Result<Vec<Vec<f32>>, PipelineError> {
    let status = response.status();
    if !status.is_success() {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let body = response.text().unwrap_or_default();
        let message = sanitize(body, secret);
        let message_lower = message.to_lowercase();
        return Err(match status.as_u16() {
            400 if message_lower.contains("context")
                || (message_lower.contains("token") && message_lower.contains("limit"))
                || (message_lower.contains("token")
                    && message_lower.contains("max")
                    && message_lower.contains("per request")) =>
            {
                PipelineError::ContextLengthExceeded
            }
            400 => PipelineError::BadRequest(message),
            401 | 403 => PipelineError::Auth,
            408 => PipelineError::ProviderError("HTTP 408 request timeout".to_string()),
            429 => PipelineError::RateLimited {
                retry_after_secs: retry_after,
            },
            500..=599 => {
                PipelineError::ProviderError(format!("HTTP {}: {message}", status.as_u16()))
            }
            _ => super::retry::classify_http_status(status.as_u16()),
        });
    }
    let payload: OpenAiResponse = response
        .json()
        .map_err(|e| PipelineError::ResponseFormat(sanitize(e.to_string(), secret)))?;
    if payload.data.len() != expected {
        return Err(PipelineError::ResponseFormat(format!(
            "returned {} vectors for {} inputs",
            payload.data.len(),
            expected
        )));
    }
    let mut vectors = vec![None; expected];
    for item in payload.data {
        if item.index >= expected || vectors[item.index].is_some() {
            return Err(PipelineError::ResponseFormat(
                "invalid or duplicate response index".to_string(),
            ));
        }
        vectors[item.index] = Some(item.embedding);
    }
    vectors
        .into_iter()
        .map(|v| {
            let v = v.ok_or_else(|| {
                PipelineError::ResponseFormat("missing response index".to_string())
            })?;
            if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
                return Err(PipelineError::ResponseFormat(
                    "empty or non-finite vector".to_string(),
                ));
            }
            Ok(v.into_iter().map(|x| x as f32).collect())
        })
        .collect()
}

fn encode_path_segment(value: &str) -> Result<String, PipelineError> {
    if value.is_empty() {
        return Err(PipelineError::BadRequest(
            "invalid Azure deployment".to_string(),
        ));
    }
    Ok(percent_encode(value))
}

fn encode_query_value(value: &str) -> Result<String, PipelineError> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(PipelineError::BadRequest(
            "invalid Azure API version".to_string(),
        ));
    }
    Ok(percent_encode(value))
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
                vec![b as char]
            } else {
                format!("%{b:02X}").chars().collect::<Vec<_>>()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EmbedBatchFn;

    fn config(base_url: String) -> EmbedConfig {
        EmbedConfig {
            provider: "openai".to_string(),
            model: "text-embedding-3-small".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(base_url),
            output_dims: None,
            matryoshka: false,
            client_side_truncation: false,
            api_version: None,
            ssl_verify: true,
            is_azure: false,
            azure_endpoint: None,
            azure_deployment: None,
            max_tokens_per_batch: 8191,
            max_items_per_batch: 100,
        }
    }

    #[test]
    fn response_indices_are_reordered_to_input_order() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [
                    {"index": 1, "embedding": [2.0, 3.0]},
                    {"index": 0, "embedding": [0.0, 1.0]}
                ]
            }));
        });
        let provider = OpenAiProvider::new(config(server.url(""))).expect("provider");
        let result = provider
            .embed_batch(&["first".to_string(), "second".to_string()])
            .expect("response");
        assert_eq!(result.vectors[0], Some(vec![0.0, 1.0]));
        assert_eq!(result.vectors[1], Some(vec![2.0, 3.0]));
        mock.assert();
    }

    #[test]
    fn embed_batch_rejects_dimension_drift_across_concurrent_calls() {
        // Simulates two rayon sub-batches calling `embed_batch` on the same
        // provider instance -- one per call, as `pipeline.rs`'s
        // `embed_batch_parallel` does via a shared `Arc<dyn EmbedBatchFn>`.
        // The provider must remember the dimension from the first call and
        // reject a differently-sized vector on the second call, even though
        // each call's own local batch is internally consistent.
        let server = httpmock::MockServer::start();
        let first_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_contains("first-batch");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}]
            }));
        });
        let second_mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/embeddings")
                .body_contains("second-batch");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]
            }));
        });
        let provider = OpenAiProvider::new(config(server.url(""))).expect("provider");

        let first = provider
            .embed_batch(&["first-batch".to_string()])
            .expect("first response");
        assert_eq!(first.vectors[0], Some(vec![1.0, 2.0]));
        assert!(first.errors.is_empty());

        let second = provider
            .embed_batch(&["second-batch".to_string()])
            .expect("second response");
        assert_eq!(second.vectors[0], None);
        assert_eq!(second.errors.len(), 1);
        assert!(second.errors[0].contains("invalid embedding vector"));

        first_mock.assert();
        second_mock.assert();
    }

    #[test]
    fn azure_url_contains_deployment_and_api_version_without_key_in_url() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/openai/deployments/my%2Fdeployment/embeddings")
                .query_param("api-version", "2024-02-01")
                .header("api-key", "test-key");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [1.0, 2.0]}]
            }));
        });
        let mut azure = config(server.url(""));
        azure.base_url = None;
        azure.is_azure = true;
        azure.azure_endpoint = Some(server.url(""));
        azure.azure_deployment = Some("my/deployment".to_string());
        azure.api_version = Some("2024-02-01".to_string());
        let provider = OpenAiProvider::new(azure).expect("provider");
        provider
            .embed_batch(&["text".to_string()])
            .expect("response");
        mock.assert();
    }
}
