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
