use super::factory::EmbedConfig;
use super::retry::{embed_with_retry, embed_with_split, RetryPolicy};
use super::token::{estimate_tokens, BatchBuilder, BatchConfig};
use super::{EmbedBatchFn, EmbedBatchResult};
use crate::error::PipelineError;
use rayon::current_thread_index;
use reqwest::blocking::{Client, Response};
use serde::Deserialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.voyageai.com/v1";
type ClientSlot = Arc<Mutex<Option<Client>>>;
type ClientSlots = Arc<Mutex<Vec<ClientSlot>>>;

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Deserialize)]
struct VoyageEmbedding {
    index: usize,
    embedding: Vec<f64>,
}

pub(crate) struct VoyageAiProvider {
    config: EmbedConfig,
    clients: ClientSlots,
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
        Ok(Self {
            config,
            clients: Arc::new(Mutex::new(Vec::new())),
            observed_dims: AtomicUsize::new(0),
        })
    }

    fn client_slot(&self) -> Result<Arc<Mutex<Option<Client>>>, PipelineError> {
        let index = current_thread_index().unwrap_or(0);
        let mut slots = self.clients.lock().map_err(|_| PipelineError::Cancelled)?;
        if slots.len() <= index {
            slots.resize_with(index + 1, || Arc::new(Mutex::new(None)));
        }
        Ok(Arc::clone(&slots[index]))
    }

    fn with_client<T>(
        &self,
        operation: impl FnOnce(&Client) -> Result<T, PipelineError>,
    ) -> Result<T, PipelineError> {
        let slot = self.client_slot()?;
        let mut client = slot.lock().map_err(|_| PipelineError::Cancelled)?;
        if client.is_none() {
            let mut builder = Client::builder()
                .timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(2);
            if !self.config.ssl_verify {
                builder = builder.danger_accept_invalid_certs(true);
            }
            *client = Some(
                builder
                    .build()
                    .map_err(|error| PipelineError::IoError(error.to_string()))?,
            );
        }
        operation(client.as_ref().ok_or(PipelineError::Cancelled)?)
    }

    fn request_once(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PipelineError> {
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
        self.with_client(|client| {
            let mut request = client.post(url).json(&body);
            if let Some(key) = self.config.api_key.as_deref().filter(|key| !key.is_empty()) {
                request = request.bearer_auth(key);
            }
            let response = request.send().map_err(|error| {
                PipelineError::IoError(sanitize(error.to_string(), self.config.api_key.as_deref()))
            })?;
            parse_response(response, texts.len(), self.config.api_key.as_deref())
        })
    }

    fn request_with_retry(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, PipelineError> {
        embed_with_retry(
            RetryPolicy {
                max_attempts: 3,
                base_delay: Duration::from_secs(1),
            },
            || self.request_once(texts),
        )
    }
}

impl EmbedBatchFn for VoyageAiProvider {
    fn embed_batch(&self, texts: &[String]) -> Result<EmbedBatchResult, String> {
        let mut result = EmbedBatchResult::empty(texts.len());
        let mut builder = BatchBuilder::new(BatchConfig {
            max_tokens: self.config.max_tokens_per_batch,
            max_items: self.config.max_items_per_batch,
        });
        let mut batches = Vec::new();
        for (index, text) in texts.iter().enumerate() {
            if estimate_tokens(text) > self.config.max_tokens_per_batch {
                result.errors.push(format!(
                    "input {index}: {}",
                    PipelineError::ContextLengthExceeded
                ));
                continue;
            }
            if let Some(batch) = builder.push(index, text.clone()) {
                batches.push(batch);
            }
        }
        if let Some(batch) = builder.finish() {
            batches.push(batch);
        }
        for batch in batches {
            log::trace!("embedding batch token estimate: {}", batch.tokens);
            let mut request = |items: &[String]| self.request_with_retry(items);
            for (offset, outcome) in embed_with_split(&batch.texts, &mut request)
                .into_iter()
                .enumerate()
            {
                let index = batch.indices[offset];
                let vector = match outcome {
                    Ok(vector) => vector,
                    Err(error) => {
                        result.errors.push(format!("input {index}: {error}"));
                        continue;
                    }
                };
                let Some(output) = self.validate_vector(vector) else {
                    result
                        .errors
                        .push(format!("input {index}: invalid embedding vector"));
                    continue;
                };
                result.vectors[index] = Some(output);
            }
        }
        Ok(result)
    }
}

impl VoyageAiProvider {
    fn validate_vector(&self, mut vector: Vec<f32>) -> Option<Vec<f32>> {
        if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
            return None;
        }
        if self.config.client_side_truncation {
            let target = self.config.output_dims?;
            if vector.len() < target {
                return None;
            }
            vector.truncate(target);
        } else if let Some(target) = self.config.output_dims {
            if vector.len() != target {
                return None;
            }
        }
        if !self.check_observed_dims(vector.len()) {
            return None;
        }
        Some(vector)
    }

    /// Establishes (on first call) or enforces (on every later call) a single
    /// embedding dimension across ALL concurrent `embed_batch` calls made on
    /// this provider instance for the lifetime of one indexing run.
    ///
    /// `embed_batch` runs concurrently on a shared `Arc<dyn EmbedBatchFn>` --
    /// one call per rayon sub-batch -- so this can't be a local variable
    /// inside `embed_batch`, or dimension drift across two different calls
    /// would never be caught. The compare-exchange makes "first observed
    /// dimension wins" atomic: on a race between two threads seeing the
    /// unset (0) sentinel, exactly one wins and sets the run's dimension,
    /// the other's compare_exchange fails and falls through to the `Err`
    /// arm, which then compares its own length against the winner's value.
    fn check_observed_dims(&self, len: usize) -> bool {
        match self
            .observed_dims
            .compare_exchange(0, len, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => true,
            Err(existing) => existing == len,
        }
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
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok());
        let body = response.text().unwrap_or_default();
        let body = sanitize(body, secret);
        return Err(match status.as_u16() {
            400 if body.to_lowercase().contains("context")
                || (body.to_lowercase().contains("token")
                    && body.to_lowercase().contains("limit")) =>
            {
                PipelineError::ContextLengthExceeded
            }
            401 | 403 => PipelineError::Auth,
            408 => PipelineError::ProviderError("HTTP 408 request timeout".to_string()),
            429 => PipelineError::RateLimited {
                retry_after_secs: retry_after,
            },
            400 => PipelineError::BadRequest(body),
            500..=599 => PipelineError::ProviderError(format!("HTTP {}: {body}", status.as_u16())),
            _ => PipelineError::ProviderError(format!("HTTP {}", status.as_u16())),
        });
    }
    let payload: VoyageResponse = response
        .json()
        .map_err(|error| PipelineError::ResponseFormat(sanitize(error.to_string(), secret)))?;
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
        .map(|vector| {
            let vector = vector.ok_or_else(|| {
                PipelineError::ResponseFormat("missing response index".to_string())
            })?;
            if vector.is_empty() || vector.iter().any(|value| !value.is_finite()) {
                return Err(PipelineError::ResponseFormat(
                    "empty or non-finite vector".to_string(),
                ));
            }
            Ok(vector.into_iter().map(|value| value as f32).collect())
        })
        .collect()
}

fn sanitize(value: String, secret: Option<&str>) -> String {
    let value = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let value = if let Some(secret) = secret.filter(|secret| !secret.is_empty()) {
        value.replace(secret, "[REDACTED]")
    } else {
        value
    };
    if value.len() <= 500 {
        value
    } else {
        format!("{}...", value.chars().take(500).collect::<String>())
    }
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
}
