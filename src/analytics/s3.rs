//! SigV4-signed `PutObject` against an S3-compatible bucket (self-hosted
//! MinIO), reusing the crate's existing blocking `reqwest` client. No
//! `ListObject`/`GetObject`/multipart calls are ever made — one presigned
//! PUT per flush, matching the write-only credential model in `AGENTS.md`'s
//! Authorization & Trust Model section.

use rusty_s3::{Bucket, Credentials, S3Action, UrlStyle};
use std::time::Duration;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum S3Error {
    #[error("invalid analytics S3 endpoint/bucket config: {0}")]
    Config(String),
    // Deliberately not `#[from] reqwest::Error` -- reqwest's Display impl
    // appends " for url (...)" for request-level failures (connection
    // refused/timeout/DNS), and the presigned SigV4 URL embeds
    // `X-Amz-Credential=<ACCESS_KEY_ID>` in its query string. Every call
    // site must construct this via `.without_url()` (see put_object below)
    // so the shared write-only access key never lands in a log line.
    #[error("analytics upload request failed: {0}")]
    Request(reqwest::Error),
    #[error("analytics upload rejected by server: HTTP {0}")]
    Rejected(u16),
}

pub(crate) struct S3Target {
    bucket: Bucket,
    credentials: Credentials,
}

impl S3Target {
    pub fn new(
        endpoint_url: &str,
        bucket_name: &str,
        access_key: &str,
        secret_key: &str,
    ) -> Result<Self, S3Error> {
        let endpoint = endpoint_url
            .parse()
            .map_err(|e| S3Error::Config(format!("invalid endpoint URL: {e}")))?;
        // MinIO's default addressing is path-style (endpoint/bucket/key), not
        // virtual-host-style (bucket.endpoint/key) — a self-hosted instance
        // is not expected to have per-bucket DNS/TLS set up.
        let bucket = Bucket::new(
            endpoint,
            UrlStyle::Path,
            bucket_name.to_string(),
            "us-east-1",
        )
        .map_err(|e| S3Error::Config(format!("invalid bucket config: {e}")))?;
        let credentials = Credentials::new(access_key, secret_key);
        Ok(Self {
            bucket,
            credentials,
        })
    }

    pub fn put_object(
        &self,
        client: &reqwest::blocking::Client,
        object_key: &str,
        body: Vec<u8>,
    ) -> Result<(), S3Error> {
        let action = self.bucket.put_object(Some(&self.credentials), object_key);
        // Short-lived presigned URL covering exactly one flush's PUT — never
        // held for the process lifetime, never persisted.
        let url = action.sign(Duration::from_secs(60));
        let response = client
            .put(url)
            .header("content-type", "application/x-ndjson")
            .body(body)
            .send()
            .map_err(|e| S3Error::Request(e.without_url()))?;
        if !response.status().is_success() {
            return Err(S3Error::Rejected(response.status().as_u16()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_failure_error_never_contains_the_access_key_or_url() {
        // Port 1 (tcpmux) is never listening -- this forces a genuine
        // request-level failure (connection refused), the exact case where
        // reqwest::Error's Display impl would otherwise append " for url
        // (...)", and with it the presigned SigV4
        // X-Amz-Credential=<ACCESS_KEY_ID> query parameter.
        let target = S3Target::new(
            "http://127.0.0.1:1",
            "analytics-bucket",
            "super-secret-access-key-id",
            "super-secret-secret-key",
        )
        .unwrap();
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();

        let err = target
            .put_object(&client, "some-key.jsonl", b"data".to_vec())
            .unwrap_err();

        let message = err.to_string();
        assert!(
            !message.contains("super-secret-access-key-id"),
            "error message must never leak the S3 access key: {message}"
        );
        assert!(
            !message.to_lowercase().contains("for url"),
            "error message must not embed the request URL at all: {message}"
        );
    }
}
