use crate::error::PipelineError;
use rand::Rng;
use std::thread;
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryPolicy {
    pub max_attempts: usize,
    pub base_delay: Duration,
}

pub(crate) fn classify_http_status(status: u16) -> PipelineError {
    match status {
        400 => PipelineError::BadRequest("invalid embedding request".to_string()),
        401 | 403 => PipelineError::Auth,
        408 => PipelineError::ProviderError("HTTP 408 request timeout".to_string()),
        429 => PipelineError::RateLimited {
            retry_after_secs: None,
        },
        _ => PipelineError::ProviderError(format!("HTTP {status}")),
    }
}

pub(crate) fn should_retry(error: &PipelineError) -> bool {
    matches!(error, PipelineError::RateLimited { .. })
        || matches!(error, PipelineError::ProviderError(message) if message.contains("HTTP 408") || message.starts_with("HTTP 5"))
        || matches!(error, PipelineError::IoError(_))
}

pub(crate) fn sleep_with_jitter(delay: Duration) {
    let factor = rand::thread_rng().gen_range(0.75..=1.25);
    thread::sleep(delay.mul_f64(factor));
}

pub(crate) fn embed_with_retry<T, F>(
    policy: RetryPolicy,
    mut request: F,
) -> Result<T, PipelineError>
where
    F: FnMut() -> Result<T, PipelineError>,
{
    let attempts = policy.max_attempts.max(1);
    let mut delay = policy.base_delay;
    let mut last_error = None;
    for attempt in 0..attempts {
        match request() {
            Ok(value) => return Ok(value),
            Err(error) if should_retry(&error) && attempt + 1 < attempts => {
                let retry_after = error.retry_after_secs();
                last_error = Some(error);
                sleep_with_jitter(retry_after.map(Duration::from_secs).unwrap_or(delay));
                delay = delay.saturating_mul(2);
            }
            Err(error) => return Err(error),
        }
    }
    Err(last_error.unwrap_or(PipelineError::Cancelled))
}

/// Apply a request to a batch, recursively halving only when the provider
/// explicitly reports a context-length failure. Returns one result per input,
/// in input order, so a failure on one half (a singleton that still can't fit,
/// or any other error) never discards vectors already obtained for the other
/// half -- only the indices belonging to the failed half are marked `Err`.
pub(crate) fn embed_with_split<T, F>(
    texts: &[String],
    request: &mut F,
) -> Vec<Result<T, PipelineError>>
where
    F: FnMut(&[String]) -> Result<Vec<T>, PipelineError>,
{
    match request(texts) {
        Ok(values) => values.into_iter().map(Ok).collect(),
        Err(PipelineError::ContextLengthExceeded) if texts.len() > 1 => {
            let midpoint = texts.len() / 2;
            let mut left = embed_with_split(&texts[..midpoint], request);
            left.extend(embed_with_split(&texts[midpoint..], request));
            left
        }
        Err(error) => texts.iter().map(|_| Err(error.clone())).collect(),
    }
}
