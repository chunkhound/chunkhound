use super::common::{is_context_length_error, sanitize, HttpClientPool};
use super::factory::EmbedConfig;
use super::{EmbedBatchFn, EmbedBatchResult};
use crate::error::PipelineError;
use reqwest::blocking::Response;
use serde::Deserialize;
use std::sync::atomic::AtomicUsize;

const DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbedding>,
    #[serde(default)]
    usage: Option<VoyageUsage>,
}

#[derive(Deserialize)]
struct VoyageEmbedding {
    index: usize,
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct VoyageUsage {
    #[serde(default, deserialize_with = "super::common::lenient_total_tokens")]
    total_tokens: Option<u64>,
}

pub(crate) struct VoyageAiProvider {
    config: EmbedConfig,
    pool: HttpClientPool,
    // First-observed embedding dimension for this provider instance, shared
    // across every concurrent `embed_batch` call (one per rayon sub-batch).
    // 0 means "not yet established" -- real embedding dimensions are always
    // > 0, so 0 is safe to use as an unset sentinel.
    observed_dims: AtomicUsize,
}

impl VoyageAiProvider {
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

    /// One vendor call attempt. Records exactly one analytics event via
    /// [`super::common::record_embed_attempt`] — this is called once per
    /// `request_with_retry`'s retry-loop iteration, matching the design's
    /// "calls = every attempt including retries" semantics, with zero
    /// further threading needed in `request_with_retry`/`run_embed_batch`.
    fn request_once(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PipelineError> {
        let result = self.request_once_inner(texts);
        super::common::record_embed_attempt(&self.config, &result);
        result.map(|(vectors, _)| vectors)
    }

    fn request_once_inner(
        &self,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, Option<u64>), PipelineError> {
        let url = format!(
            "{}/embeddings",
            self.config
                .base_url
                .as_deref()
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
        );
        let mut body = serde_json::json!({
            "model": &self.config.model,
            "input": texts,
            "input_type": "document",
            "truncation": true,
        });
        if self.config.output_dims.is_some() && !self.config.client_side_truncation {
            body["output_dimension"] = serde_json::json!(self.config.output_dims);
        }
        self.pool.with_client(|client| {
            let mut request = client.post(url).json(&body);
            if let Some(key) = self.config.api_key.as_deref().filter(|k| !k.is_empty()) {
                request = request.bearer_auth(key);
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

impl EmbedBatchFn for VoyageAiProvider {
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
) -> Result<(Vec<Vec<f32>>, Option<u64>), PipelineError> {
    let status = response.status();
    if !status.is_success() {
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok());
        let body = response.text().unwrap_or_default();
        let body = sanitize(body, secret);
        let body_lower = body.to_lowercase();
        return Err(match status.as_u16() {
            400 if is_context_length_error(&body_lower) => PipelineError::ContextLengthExceeded,
            400 => PipelineError::BadRequest(body),
            401 | 403 => PipelineError::Auth,
            408 => PipelineError::ProviderError("HTTP 408 request timeout".to_string()),
            429 => PipelineError::RateLimited {
                retry_after_secs: retry_after,
            },
            500..=599 => PipelineError::ProviderError(format!("HTTP {}: {body}", status.as_u16())),
            _ => PipelineError::ProviderError(format!("HTTP {}", status.as_u16())),
        });
    }
    let payload: VoyageResponse = response
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
    let input_tokens = payload.usage.and_then(|u| u.total_tokens);
    let vectors: Result<Vec<Vec<f32>>, PipelineError> = vectors
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
        .collect();
    Ok((vectors?, input_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::EmbedBatchFn;

    fn config(base_url: String) -> EmbedConfig {
        EmbedConfig {
            provider: "voyageai".to_string(),
            model: "voyage-3".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: Some(base_url),
            output_dims: None,
            matryoshka: false,
            model_known: true,
            client_side_truncation: false,
            api_version: None,
            ssl_verify: true,
            is_azure: false,
            azure_endpoint: None,
            azure_deployment: None,
            max_tokens_per_batch: 8191,
            max_items_per_batch: 100,
            analytics: None,
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
        let provider = VoyageAiProvider::new(config(server.url(""))).expect("provider");
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
        let provider = VoyageAiProvider::new(config(server.url(""))).expect("provider");

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
    fn embed_batch_records_a_successful_provider_call_with_token_usage() {
        let dir = tempfile::tempdir().unwrap();
        let inner = crate::analytics::test_inner(dir.path());
        let handle = inner.test_start_command();

        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"total_tokens": 17}
            }));
        });
        let mut cfg = config(server.url(""));
        cfg.analytics = Some((inner.clone(), handle));
        let provider = VoyageAiProvider::new(cfg).expect("provider");

        provider
            .embed_batch(&["hello".to_string()])
            .expect("response");
        mock.assert();

        let state = inner.test_end_command(handle).unwrap();
        let stats = state
            .providers
            .get(&(
                "embedding".to_string(),
                "voyageai".to_string(),
                "voyage-3".to_string(),
            ))
            .unwrap();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.fails, 0);
        assert_eq!(stats.input_tokens, Some(17));
    }

    #[test]
    fn embed_batch_succeeds_when_usage_object_is_missing_total_tokens() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"total_tokens_used": 17}
            }));
        });
        let provider = VoyageAiProvider::new(config(server.url(""))).expect("provider");

        let response = provider
            .embed_batch(&["hello".to_string()])
            .expect("response");
        assert_eq!(response.vectors[0], Some(vec![0.1, 0.2]));
        mock.assert();
    }

    #[test]
    fn embed_batch_succeeds_when_total_tokens_has_the_wrong_json_type() {
        for malformed in [
            serde_json::json!("42"),
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::Value::Null,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let inner = crate::analytics::test_inner(dir.path());
            let handle = inner.test_start_command();

            let server = httpmock::MockServer::start();
            let mock = server.mock(|when, then| {
                when.method(httpmock::Method::POST).path("/embeddings");
                then.status(200).json_body(serde_json::json!({
                    "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"total_tokens": malformed.clone()}
                }));
            });
            let mut cfg = config(server.url(""));
            cfg.analytics = Some((inner.clone(), handle));
            let provider = VoyageAiProvider::new(cfg).expect("provider");

            let response = provider
                .embed_batch(&["hello".to_string()])
                .unwrap_or_else(|e| {
                    panic!("malformed total_tokens {malformed:?} discarded the response: {e}")
                });
            assert_eq!(response.vectors[0], Some(vec![0.1, 0.2]));
            mock.assert();

            let state = inner.test_end_command(handle).unwrap();
            let stats = state
                .providers
                .get(&(
                    "embedding".to_string(),
                    "voyageai".to_string(),
                    "voyage-3".to_string(),
                ))
                .unwrap();
            assert_eq!(stats.calls, 1);
            assert_eq!(stats.fails, 0);
            assert_eq!(
                stats.input_tokens, None,
                "malformed total_tokens must not be coerced into a bogus count"
            );
        }
    }

    #[test]
    fn embed_batch_records_a_failed_provider_call_with_error_type() {
        let dir = tempfile::tempdir().unwrap();
        let inner = crate::analytics::test_inner(dir.path());
        let handle = inner.test_start_command();

        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(401);
        });
        let mut cfg = config(server.url(""));
        cfg.analytics = Some((inner.clone(), handle));
        let provider = VoyageAiProvider::new(cfg).expect("provider");

        let _ = provider.embed_batch(&["hello".to_string()]);
        mock.assert();

        let state = inner.test_end_command(handle).unwrap();
        let stats = state
            .providers
            .get(&(
                "embedding".to_string(),
                "voyageai".to_string(),
                "voyage-3".to_string(),
            ))
            .unwrap();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.fails, 1);
        assert_eq!(stats.error_types.get("Auth"), Some(&1));
    }
}
