//! Per-user usage analytics: buffer one `command_summary` JSONL event per
//! MCP tool call / bounded CLI command, periodically flush the buffer as one
//! object directly to an S3-compatible bucket. See `AGENTS.md` (root) for the
//! design rationale and `src/AGENTS.md` for how this fits into the crate.
//!
//! This module must never disrupt a host command: every public method on
//! `AnalyticsRecorder` catches and logs its own internal failures rather than
//! propagating a `PyErr`, and constructing the recorder with `enabled=false`
//! (or invalid config) yields an inert no-op rather than an error.

mod command;
mod identity;
mod recorder;
mod repository;
mod s3;

pub(crate) use command::ProviderCall;
pub use recorder::AnalyticsRecorder;
pub(crate) use recorder::Inner;

#[cfg(test)]
pub(crate) use recorder::test_inner;
