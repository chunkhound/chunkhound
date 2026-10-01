//! Shared primitives used by every native HTTP embedding provider.
//!
//! Extracted from the original per-provider modules to eliminate the
//! ~250-line duplication between `openai.rs` and `voyageai.rs`.

use super::factory::EmbedConfig;
use super::retry::{embed_with_retry, embed_with_split, RetryPolicy};
use super::token::{estimate_tokens, BatchBuilder, BatchConfig};
use super::EmbedBatchResult;
use crate::error::PipelineError;
use rayon::current_thread_index;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub(crate) type ClientSlot = Arc<Mutex<Option<Client>>>;
pub(crate) type ClientSlots = Arc<Mutex<Vec<ClientSlot>>>;

pub(crate) const MAX_ERROR_LEN: usize = 500;

// ── HTTP connection pool ────────────────────────────────────────────────────

/// Per-provider HTTP client pool with one lazily-initialised `reqwest` client
/// per rayon thread index.  Both `OpenAiProvider` and `VoyageAiProvider`
/// embed this instead of duplicating the client-management logic.
pub(crate) struct HttpClientPool {
    clients: ClientSlots,
    ssl_verify: bool,
}

impl HttpClientPool {
    pub(crate) fn new(ssl_verify: bool) -> Self {
        Self {
            clients: Arc::new(Mutex::new(Vec::new())),
            ssl_verify,
        }
    }

    pub(crate) fn client_slot(&self) -> Result<Arc<Mutex<Option<Client>>>, PipelineError> {
        let index = current_thread_index().unwrap_or(0);
        let mut slots = self.clients.lock().map_err(|_| PipelineError::Cancelled)?;
        if slots.len() <= index {
            slots.resize_with(index + 1, || Arc::new(Mutex::new(None)));
        }
        Ok(Arc::clone(&slots[index]))
    }

    /// Obtain (or lazily create) the per-thread reqwest [`Client`] and run
    /// `operation` against it.
    pub(crate) fn with_client<T>(
        &self,
        operation: impl FnOnce(&Client) -> Result<T, PipelineError>,
    ) -> Result<T, PipelineError> {
        let slot = self.client_slot()?;
        let mut client = slot.lock().map_err(|_| PipelineError::Cancelled)?;
        if client.is_none() {
            let mut builder = Client::builder()
                .timeout(Duration::from_secs(30))
                .pool_max_idle_per_host(2);
            if !self.ssl_verify {
                builder = builder.danger_accept_invalid_certs(true);
            }
            *client = Some(
                builder
                    .build()
                    .map_err(|e| PipelineError::IoError(sanitize(e.to_string(), None)))?,
            );
        }
        operation(client.as_ref().ok_or(PipelineError::Cancelled)?)
    }
}

// ── Dimension tracking ──────────────────────────────────────────────────────

/// Establish (on first call) or enforce (on every later call) a single
/// embedding dimension for the lifetime of one provider instance.
///
/// `embed_batch` is called concurrently on a shared `Arc<dyn EmbedBatchFn>` —
/// one call per rayon sub-batch.  Using `AtomicUsize` with compare-exchange
/// makes "first observed dimension wins" lock-free: exactly one thread sets
/// the sentinel from 0 to the real dimension; all subsequent calls compare
/// against that value.  Returns `false` if `len` differs from the already-
/// established dimension.
pub(crate) fn check_observed_dims(observed: &AtomicUsize, len: usize) -> bool {
    match observed.compare_exchange(0, len, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => true,
        Err(existing) => existing == len,
    }
}

// ── Vector validation ───────────────────────────────────────────────────────

/// Validate one embedding vector against the provider's config and the run's
/// established dimension, optionally truncating for client-side truncation.
///
/// Returns `None` if the vector is invalid (empty, non-finite, wrong size, or
/// dimension drift detected).
pub(crate) fn validate_vector(
    config: &EmbedConfig,
    observed_dims: &AtomicUsize,
    mut vector: Vec<f32>,
) -> Option<Vec<f32>> {
    if vector.is_empty() || vector.iter().any(|v| !v.is_finite()) {
        return None;
    }
    if config.client_side_truncation {
        let target = config.output_dims?;
        if vector.len() < target {
            return None;
        }
        vector.truncate(target);
    } else if let Some(target) = config.output_dims {
        if vector.len() != target {
            return None;
        }
    }
    if !check_observed_dims(observed_dims, vector.len()) {
        return None;
    }
    Some(vector)
}

// ── Batching orchestration ──────────────────────────────────────────────────

/// Run the full batch-build → split-on-context-length → validate loop.
///
/// Both `OpenAiProvider` and `VoyageAiProvider` delegate their `embed_batch`
/// implementation here; the only per-provider difference is the
/// `request_with_retry` closure that performs the actual HTTP call.
pub(crate) fn run_embed_batch(
    texts: &[String],
    config: &EmbedConfig,
    observed_dims: &AtomicUsize,
    mut request_with_retry: impl FnMut(&[String]) -> Result<Vec<Vec<f32>>, PipelineError>,
) -> Result<EmbedBatchResult, String> {
    let mut result = EmbedBatchResult::empty(texts.len());
    let mut builder = BatchBuilder::new(BatchConfig {
        max_tokens: config.max_tokens_per_batch,
        max_items: config.max_items_per_batch,
    });
    let mut batches = Vec::new();
    for (index, text) in texts.iter().enumerate() {
        if estimate_tokens(text) > config.max_tokens_per_batch {
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
        for (offset, outcome) in embed_with_split(&batch.texts, &mut request_with_retry)
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
            let Some(output) = validate_vector(config, observed_dims, vector) else {
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

// ── Analytics ────────────────────────────────────────────────────────────────

/// Record one vendor call attempt against the command bound to `config`, if
/// any. Called once per `request_with_retry` retry-loop iteration, matching
/// the design's "calls = every attempt including retries" semantics.
///
/// Shared by both providers on purpose: `OpenAiProvider`/`VoyageAiProvider`'s
/// `request_once` bodies were byte-for-byte identical here, and a per-provider
/// copy is exactly the kind of thing a future analytics-schema change would
/// only land in one file, per this module's own precedent
/// (`is_context_length_error`'s doc comment above).
pub(crate) fn record_embed_attempt(
    config: &EmbedConfig,
    result: &Result<(Vec<Vec<f32>>, Option<u64>), PipelineError>,
) {
    let Some((analytics, handle)) = &config.analytics else {
        return;
    };
    let call = match result {
        Ok((_, input_tokens)) => crate::analytics::ProviderCall {
            kind: "embedding",
            provider: &config.provider,
            model: &config.model,
            success: true,
            error_type: None,
            input_tokens: *input_tokens,
            output_tokens: None,
        },
        Err(e) => crate::analytics::ProviderCall {
            kind: "embedding",
            provider: &config.provider,
            model: &config.model,
            success: false,
            error_type: Some(e.analytics_error_type()),
            input_tokens: None,
            output_tokens: None,
        },
    };
    analytics.record_provider_call(*handle, call);
}

/// Tolerate any JSON shape for a response's `usage.total_tokens` field --
/// missing, `null`, a string, negative, or a float all decode to `None`
/// rather than failing deserialization of the surrounding response struct.
/// A malformed token count must never discard an otherwise-valid batch of
/// embedding vectors just because analytics couldn't parse the usage field.
/// Shared by both providers' `*Usage` structs via `#[serde(deserialize_with
/// = "...")]`, per this module's own precedent (see `record_embed_attempt`
/// above).
pub(crate) fn lenient_total_tokens<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(serde_json::Value::deserialize(deserializer)?.as_u64())
}

// ── Retry wrapper ───────────────────────────────────────────────────────────

/// Convenience wrapper used by both providers' `request_with_retry` methods.
pub(crate) fn default_retry<T>(
    request: impl FnMut() -> Result<T, PipelineError>,
) -> Result<T, PipelineError> {
    embed_with_retry(
        RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_secs(1),
        },
        request,
    )
}

// ── Error classification ────────────────────────────────────────────────────

/// Whether a 400 response body reports a per-request context/token-length
/// overflow, i.e. the one failure `embed_with_split` can actually recover from.
///
/// Mirrors the three conjunctive predicates in Python's
/// `openai_provider._embed_batch_internal`. Deliberately *not* a bare
/// "context" or "token"+"limit" match: those also fire on quota and
/// malformed-request 400s, which splitting cannot fix. Misclassifying one
/// costs `2N-1` requests for an N-input batch (the recursive halving in
/// `embed_with_split`, which is excluded from retry so each node is a single
/// call) and then reports "input exceeds the provider context limit" for every
/// chunk, burying the real cause.
///
/// Expects an already-lowercased body. Python's predicates are case-sensitive;
/// matching case-insensitively is a deliberate widening, since it only adds
/// bodies that differ from the known wording by capitalisation and it errs
/// toward the recoverable path.
///
/// Shared by both providers on purpose: this classifier has needed three
/// upstream revisions (most recently #404), and a per-provider copy is the
/// likeliest way a future fix lands in one file only.
pub(crate) fn is_context_length_error(body_lower: &str) -> bool {
    (body_lower.contains("maximum context length") && body_lower.contains("tokens"))
        || (body_lower.contains("tokens")
            && body_lower.contains("max")
            && body_lower.contains("per request"))
        || (body_lower.contains("input length exceeds") && body_lower.contains("context length"))
}

// ── Error sanitisation ──────────────────────────────────────────────────────

/// Replace control characters, redact the API secret, and truncate long
/// messages so they are safe to surface in error strings.
pub(crate) fn sanitize(value: String, secret: Option<&str>) -> String {
    let value = value
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>();
    let value = if let Some(secret) = secret.filter(|s| !s.is_empty()) {
        value.replace(secret, "[REDACTED]")
    } else {
        value
    };
    if value.len() <= MAX_ERROR_LEN {
        value
    } else {
        format!(
            "{}...",
            value.chars().take(MAX_ERROR_LEN).collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three phrasings Python recognises, lowercased as `parse_response`
    /// hands them over. Each must reach the recoverable split path.
    #[test]
    fn recognizes_every_python_context_length_phrasing() {
        assert!(is_context_length_error(
            "this model's maximum context length is 8192 tokens, however you \
             requested 10000 tokens."
        ));
        assert!(is_context_length_error(
            "your request has 9001 tokens which exceeds the 8192 max tokens per request."
        ));
        assert!(is_context_length_error(
            "input length exceeds context length limit of 8192"
        ));
    }

    /// The regression this guards: a bare "context" or "token"+"limit" match
    /// also fires on quota and malformed-request 400s. Splitting cannot fix
    /// those -- it costs 2N-1 requests against an endpoint that already said
    /// no, and then reports a context-length failure for every chunk, hiding
    /// the real cause.
    #[test]
    fn rejects_non_length_400_bodies() {
        assert!(!is_context_length_error(
            "api token limit exceeded for this organization"
        ));
        assert!(!is_context_length_error(
            "unsupported parameter 'dimensions' for this model in the embeddings context"
        ));
        assert!(!is_context_length_error("invalid api token"));
        assert!(!is_context_length_error(
            "the response was filtered due to the content management policy"
        ));
        assert!(!is_context_length_error(""));
    }

    /// Python's predicates are case-sensitive; ours are not. That widening is
    /// deliberate -- it only adds bodies differing by capitalisation, and errs
    /// toward the recoverable path.
    #[test]
    fn matching_is_case_insensitive_on_the_lowercased_body() {
        let body = "This Model's Maximum Context Length Is 8192 Tokens".to_lowercase();
        assert!(is_context_length_error(&body));
    }
}
