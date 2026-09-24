use super::common::{is_context_length_error, sanitize, HttpClientPool};
use super::factory::EmbedConfig;
use super::{EmbedBatchFn, EmbedBatchResult};
use crate::error::PipelineError;
use reqwest::blocking::Response;
use serde::Deserialize;
use std::sync::atomic::AtomicUsize;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Mirrors Python's `openai_utils.is_official_openai_endpoint`: an unset
/// (or empty) `base_url` means the default official endpoint, and an explicit
/// URL counts as official only when it is exactly `https://api.openai.com` or
/// sits beneath it. The trailing-slash check is what keeps a look-alike host
/// such as `https://api.openai.com.evil.example/v1` from being trusted.
fn is_official_openai_endpoint(base_url: Option<&str>) -> bool {
    match base_url {
        None => true,
        Some("") => true,
        Some(url) => url == "https://api.openai.com" || url.starts_with("https://api.openai.com/"),
    }
}

/// Mirrors Python's `dimensions` gate in
/// `openai_provider._build_embedding_request_kwargs`: the param is withheld
/// only when the endpoint is trusted as official AND the model is a *known*,
/// non-matryoshka entry in `OPENAI_MODEL_CONFIG`. A non-official endpoint, a
/// matryoshka model, or an unrecognized model are all trusted to accept (or
/// reject) the parameter live rather than have it withheld based on a static
/// table lookup that can't speak for them.
///
/// `trust_runtime_output_dims` is Python's `_trust_runtime_output_dims()`,
/// i.e. `not is_official_openai_endpoint(base_url)`. It is deliberately *not*
/// "a `base_url` was configured": pointing `base_url` explicitly at
/// `https://api.openai.com/v1` is still the official endpoint, so the
/// withholding rule has to apply there exactly as it does when `base_url` is
/// unset.
///
/// Azure is likewise not a bypass. Azure configs leave `base_url` unset
/// (`factory.rs` rejects `is_azure` together with a `base_url`), so `None`
/// reads as official and the withholding rule applies to Azure exactly as it
/// does to api.openai.com.
fn should_send_dimensions(
    output_dims_set: bool,
    client_side_truncation: bool,
    matryoshka: bool,
    trust_runtime_output_dims: bool,
    model_known: bool,
) -> bool {
    output_dims_set
        && !client_side_truncation
        && (matryoshka || trust_runtime_output_dims || !model_known)
}

/// Apply [`should_send_dimensions`] to a concrete config. Kept separate from
/// the predicate so the endpoint-trust wiring itself is unit-testable: the
/// mock-server harness always points `base_url` at localhost and so can never
/// reach the official-endpoint branch.
fn should_send_dimensions_for(config: &EmbedConfig) -> bool {
    should_send_dimensions(
        config.output_dims.is_some(),
        config.client_side_truncation,
        config.matryoshka,
        !is_official_openai_endpoint(config.base_url.as_deref()),
        config.model_known,
    )
}

#[derive(Deserialize)]
struct OpenAiResponse {
    data: Vec<OpenAiEmbedding>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiEmbedding {
    index: usize,
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    #[serde(default, deserialize_with = "super::common::lenient_total_tokens")]
    total_tokens: Option<u64>,
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
        let url = self.url()?;
        let mut body = serde_json::json!({
            "model": if self.config.is_azure {
                self.config.azure_deployment.as_deref().unwrap_or(&self.config.model)
            } else {
                &self.config.model
            },
            "input": texts,
        });
        if should_send_dimensions_for(&self.config) {
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
) -> Result<(Vec<Vec<f32>>, Option<u64>), PipelineError> {
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
            400 if is_context_length_error(&message_lower) => PipelineError::ContextLengthExceeded,
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
        .flat_map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                vec![byte as char].into_iter().collect::<Vec<_>>()
            } else {
                format!("%{byte:02X}").chars().collect::<Vec<_>>()
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

    /// Build a config for the gate tests: known, non-matryoshka model with
    /// `output_dims` set, i.e. the one combination whose outcome depends
    /// entirely on whether the endpoint is judged official.
    fn ada_config(base_url: Option<&str>) -> EmbedConfig {
        EmbedConfig {
            model: "text-embedding-ada-002".to_string(),
            base_url: base_url.map(str::to_string),
            output_dims: Some(1536),
            matryoshka: false,
            model_known: true,
            ..config("https://example.invalid/v1".to_string())
        }
    }

    #[test]
    fn official_endpoint_recognizes_the_default_and_explicit_forms() {
        // Unset or empty means "default official endpoint", matching Python's
        // falsy `if not base_url` check.
        assert!(is_official_openai_endpoint(None));
        assert!(is_official_openai_endpoint(Some("")));
        assert!(is_official_openai_endpoint(Some("https://api.openai.com")));
        assert!(is_official_openai_endpoint(Some(
            "https://api.openai.com/v1"
        )));
        // A look-alike host must not inherit the official endpoint's trust
        // just because it shares a prefix.
        assert!(!is_official_openai_endpoint(Some(
            "https://api.openai.com.evil.example/v1"
        )));
        assert!(!is_official_openai_endpoint(Some(
            "http://api.openai.com/v1"
        )));
        assert!(!is_official_openai_endpoint(Some("https://proxy.local/v1")));
    }

    #[test]
    fn dimensions_withheld_for_known_model_on_explicitly_official_base_url() {
        // The regression this guards: a config may name the official endpoint
        // explicitly instead of leaving `base_url` unset. Python's
        // `is_official_openai_endpoint` still reads that as official, so a
        // known non-matryoshka model must NOT get `dimensions` -- real OpenAI
        // answers 400 "does not support specifying dimensions" for ada-002,
        // which is not retryable or splittable and costs the run every vector.
        // Gating on "a base_url was configured" instead of endpoint trust got
        // this wrong.
        assert!(!should_send_dimensions_for(&ada_config(Some(
            "https://api.openai.com/v1"
        ))));
        assert!(!should_send_dimensions_for(&ada_config(Some(
            "https://api.openai.com"
        ))));
        // Same rule when base_url is simply unset.
        assert!(!should_send_dimensions_for(&ada_config(None)));
        // A genuinely custom endpoint is trusted to judge the param itself.
        assert!(should_send_dimensions_for(&ada_config(Some(
            "https://proxy.local/v1"
        ))));
    }

    #[test]
    fn dimensions_gate_treats_azure_as_an_official_endpoint() {
        // Azure leaves `base_url` unset (factory.rs rejects is_azure together
        // with a base_url), so `is_official_openai_endpoint(None)` is true and
        // the known-model withholding rule applies to Azure exactly as it does
        // to api.openai.com. Sending `dimensions` for a known non-matryoshka
        // model is a live 400 from Azure.
        let azure = EmbedConfig {
            is_azure: true,
            azure_endpoint: Some("https://example.openai.azure.com".to_string()),
            azure_deployment: Some("ada".to_string()),
            api_version: Some("2024-02-01".to_string()),
            ..ada_config(None)
        };
        assert!(!should_send_dimensions_for(&azure));
        // An unknown model is still trusted at runtime, on Azure as elsewhere.
        assert!(should_send_dimensions_for(&EmbedConfig {
            model_known: false,
            ..azure
        }));
    }

    #[test]
    fn dimensions_withheld_for_known_model_on_official_endpoint() {
        // The only case where dimensions must be withheld: trusted official
        // endpoint, known model, not matryoshka.
        assert!(!should_send_dimensions(true, false, false, false, true));
    }

    #[test]
    fn dimensions_sent_for_unknown_model_on_official_endpoint() {
        // An unknown model on the official endpoint must still get
        // `dimensions`, matching Python's `model not in self._model_config`
        // early-return. The mock-server harness in test_embed_parity.py always
        // sets a custom base_url, so it structurally cannot reach this branch
        // -- only this unit test can.
        assert!(should_send_dimensions(true, false, false, false, false));
    }

    #[test]
    fn dimensions_sent_for_matryoshka_model_on_official_endpoint() {
        assert!(should_send_dimensions(true, false, true, false, true));
    }

    #[test]
    fn dimensions_sent_for_untrusted_endpoint_regardless_of_model_known() {
        assert!(should_send_dimensions(true, false, false, true, true));
        assert!(should_send_dimensions(true, false, false, true, false));
    }

    #[test]
    fn dimensions_withheld_when_output_dims_unset() {
        assert!(!should_send_dimensions(false, false, false, false, false));
    }

    #[test]
    fn dimensions_withheld_for_client_side_truncation() {
        assert!(!should_send_dimensions(true, true, true, true, false));
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
    fn embed_batch_records_a_successful_provider_call_with_token_usage() {
        let dir = tempfile::tempdir().unwrap();
        let inner = crate::analytics::test_inner(dir.path());
        let handle = inner.test_start_command();

        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"total_tokens": 42}
            }));
        });
        let mut cfg = config(server.url(""));
        cfg.analytics = Some((inner.clone(), handle));
        let provider = OpenAiProvider::new(cfg).expect("provider");

        provider
            .embed_batch(&["hello".to_string()])
            .expect("response");
        mock.assert();

        let state = inner.test_end_command(handle).unwrap();
        let stats = state
            .providers
            .get(&(
                "embedding".to_string(),
                "openai".to_string(),
                "text-embedding-3-small".to_string(),
            ))
            .unwrap();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.fails, 0);
        assert_eq!(stats.input_tokens, Some(42));
    }

    #[test]
    fn embed_batch_succeeds_when_usage_object_is_missing_total_tokens() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(serde_json::json!({
                "data": [{"index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 42}
            }));
        });
        let provider = OpenAiProvider::new(config(server.url(""))).expect("provider");

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
            let provider = OpenAiProvider::new(cfg).expect("provider");

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
                    "openai".to_string(),
                    "text-embedding-3-small".to_string(),
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
        let provider = OpenAiProvider::new(cfg).expect("provider");

        let _ = provider.embed_batch(&["hello".to_string()]);
        mock.assert();

        let state = inner.test_end_command(handle).unwrap();
        let stats = state
            .providers
            .get(&(
                "embedding".to_string(),
                "openai".to_string(),
                "text-embedding-3-small".to_string(),
            ))
            .unwrap();
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.fails, 1);
        assert_eq!(stats.error_types.get("Auth"), Some(&1));
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
