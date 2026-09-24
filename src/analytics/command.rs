//! Command bookkeeping: no `contextvars`/thread-local anywhere — callers
//! (Python or Rust) hold the `u64` handle returned by `start()` explicitly
//! and pass it back into every subsequent call. This is what lets the same
//! table be used correctly from both ordinary single-threaded Python call
//! chains and the Rust rayon thread pool during indexing, with no
//! cross-thread propagation problem to solve.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Default, Clone)]
pub(crate) struct ProviderStats {
    pub calls: u64,
    pub fails: u64,
    pub error_types: HashMap<String, u64>,
    /// `None` until the first call reports a token count; stays `None` if no
    /// call ever does (e.g. reranker calls never report tokens at all, and
    /// embedding calls never report `output_tokens`) so the serialized event
    /// emits `null` rather than a misleading `0`. See wiki's `providers[]`
    /// schema: these are `integer | null`, not always-present integers.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

/// Key: (kind, provider, model) e.g. ("embedding", "openai", "text-embedding-3-small").
pub(crate) type ProviderKey = (String, String, String);

/// One provider-call outcome to fold into a command's rollup. Bundled into a
/// struct (rather than seven positional args) both to sidestep
/// `clippy::too_many_arguments` and because "one attempt's outcome" is a
/// meaningful unit on its own — this is what both a Python hook site and a
/// Rust-internal caller (the native embed adapters) construct per attempt.
pub(crate) struct ProviderCall<'a> {
    pub kind: &'a str,
    pub provider: &'a str,
    pub model: &'a str,
    pub success: bool,
    pub error_type: Option<&'a str>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct CommandState {
    pub command: String,
    pub source: String,
    /// Pre-serialized JSON object (validated at construction; falls back to
    /// `{}` on parse failure so a bad action payload can never break a host
    /// command). See `AnalyticsRecorder::start_command`.
    pub action: serde_json::Value,
    pub started_at: Instant,
    pub providers: HashMap<ProviderKey, ProviderStats>,
    pub internal_error_type: Option<String>,
}

impl CommandState {
    fn new(command: String, source: String, action: serde_json::Value) -> Self {
        Self {
            command,
            source,
            action,
            started_at: Instant::now(),
            providers: HashMap::new(),
            internal_error_type: None,
        }
    }

    fn record_provider_call(&mut self, call: ProviderCall<'_>) {
        let key = (
            call.kind.to_string(),
            call.provider.to_string(),
            call.model.to_string(),
        );
        let stats = self.providers.entry(key).or_default();
        stats.calls += 1;
        if call.success {
            if let Some(t) = call.input_tokens {
                stats.input_tokens = Some(stats.input_tokens.unwrap_or(0) + t);
            }
            if let Some(t) = call.output_tokens {
                stats.output_tokens = Some(stats.output_tokens.unwrap_or(0) + t);
            }
        } else {
            stats.fails += 1;
            if let Some(et) = call.error_type {
                *stats.error_types.entry(et.to_string()).or_insert(0) += 1;
            }
        }
    }

    /// Only meaningful when no provider entry has recorded a fail — a
    /// vendor-caused failure already explains the command's outcome, so an
    /// internal error recorded after one is a deliberate no-op, not a bug.
    fn record_internal_error(&mut self, error_type: String) {
        let any_provider_failed = self.providers.values().any(|s| s.fails > 0);
        if !any_provider_failed {
            self.internal_error_type = Some(error_type);
        }
    }
}

/// Table of currently-open commands, safe to share across Rust threads (the
/// rayon pool) and across concurrent Python asyncio tasks via PyO3 calls.
#[derive(Default)]
pub(crate) struct CommandTable {
    next_id: AtomicU64,
    open: Mutex<HashMap<u64, CommandState>>,
}

impl CommandTable {
    pub fn start(&self, command: String, source: String, action: serde_json::Value) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.lock()
            .insert(id, CommandState::new(command, source, action));
        id
    }

    /// Unknown handle is silently dropped — "shouldn't normally happen" per
    /// design, and must never turn into a panic or a raised error for
    /// something as incidental as a stale/mistaken handle.
    pub fn record_provider_call(&self, handle: u64, call: ProviderCall<'_>) {
        if let Some(state) = self.lock().get_mut(&handle) {
            state.record_provider_call(call);
        }
    }

    pub fn record_internal_error(&self, handle: u64, error_type: String) {
        if let Some(state) = self.lock().get_mut(&handle) {
            state.record_internal_error(error_type);
        }
    }

    /// Merges new keys into a still-open command's action fields, overwriting
    /// on conflict. For commands whose action fields aren't fully known at
    /// `start()` time (e.g. `index`'s `file_count`/`total_chunks`, only known
    /// once the run completes) -- `start()` records what's known upfront,
    /// this fills in the rest before `end()`. Unknown handle or a
    /// non-object `action` value is silently dropped.
    pub fn update_action(&self, handle: u64, action: serde_json::Value) {
        let serde_json::Value::Object(new_fields) = action else {
            return;
        };
        if let Some(state) = self.lock().get_mut(&handle) {
            if let serde_json::Value::Object(existing) = &mut state.action {
                existing.extend(new_fields);
            }
        }
    }

    /// Finalizes and removes the command, returning its accumulated state
    /// for the caller (`AnalyticsRecorder::end_command`) to turn into a
    /// `command_summary` event. Unknown handle returns `None` silently.
    pub fn end(&self, handle: u64) -> Option<CommandState> {
        self.lock().remove(&handle)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<u64, CommandState>> {
        self.open.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::thread;

    fn call<'a>(
        kind: &'a str,
        provider: &'a str,
        model: &'a str,
        success: bool,
    ) -> ProviderCall<'a> {
        ProviderCall {
            kind,
            provider,
            model,
            success,
            error_type: None,
            input_tokens: None,
            output_tokens: None,
        }
    }

    #[test]
    fn lifecycle_produces_a_correctly_shaped_rollup() {
        let table = CommandTable::default();
        let handle = table.start(
            "search".to_string(),
            "mcp".to_string(),
            json!({"query": "x"}),
        );
        table.record_provider_call(handle, call("llm", "anthropic", "claude", true));
        table.record_provider_call(handle, call("llm", "anthropic", "claude", true));
        table.record_provider_call(
            handle,
            ProviderCall {
                error_type: Some("RateLimitError"),
                ..call("embedding", "voyageai", "voyage-3", false)
            },
        );

        let state = table.end(handle).expect("command must still be open");
        assert_eq!(state.command, "search");
        assert_eq!(state.source, "mcp");
        assert_eq!(state.action, json!({"query": "x"}));

        let llm = state
            .providers
            .get(&(
                "llm".to_string(),
                "anthropic".to_string(),
                "claude".to_string(),
            ))
            .unwrap();
        assert_eq!(llm.calls, 2);
        assert_eq!(llm.fails, 0);

        let embedding = state
            .providers
            .get(&(
                "embedding".to_string(),
                "voyageai".to_string(),
                "voyage-3".to_string(),
            ))
            .unwrap();
        assert_eq!(embedding.calls, 1);
        assert_eq!(embedding.fails, 1);
        assert_eq!(embedding.error_types.get("RateLimitError"), Some(&1));
    }

    #[test]
    fn unknown_handle_is_silently_dropped_everywhere() {
        let table = CommandTable::default();
        // None of these may panic despite handle 999 never having been started.
        table.record_provider_call(999, call("llm", "openai", "gpt", true));
        table.record_internal_error(999, "KeyError".to_string());
        table.update_action(999, json!({"file_count": 1}));
        assert!(table.end(999).is_none());
    }

    #[test]
    fn update_action_merges_fields_known_only_after_the_command_runs() {
        // Mirrors the index command: `mode` is known at start_command time,
        // `file_count`/`total_chunks` are only known once indexing finishes.
        let table = CommandTable::default();
        let handle = table.start(
            "index".to_string(),
            "cli".to_string(),
            json!({"mode": "initial"}),
        );
        table.update_action(handle, json!({"file_count": 42, "total_chunks": 1337}));

        let state = table.end(handle).unwrap();
        assert_eq!(
            state.action,
            json!({"mode": "initial", "file_count": 42, "total_chunks": 1337})
        );
    }

    #[test]
    fn update_action_on_unknown_handle_is_a_silent_noop() {
        let table = CommandTable::default();
        table.update_action(999, json!({"file_count": 1}));
        // No panic; nothing further to assert since there's no state to read.
    }

    #[test]
    fn internal_error_is_recorded_only_when_no_provider_failed() {
        let table = CommandTable::default();

        let clean_failure = table.start("search".to_string(), "cli".to_string(), json!({}));
        table.record_internal_error(clean_failure, "KeyError".to_string());
        let state = table.end(clean_failure).unwrap();
        assert_eq!(state.internal_error_type, Some("KeyError".to_string()));

        let vendor_failure = table.start("search".to_string(), "cli".to_string(), json!({}));
        table.record_provider_call(
            vendor_failure,
            ProviderCall {
                error_type: Some("TimeoutError"),
                ..call("llm", "openai", "gpt", false)
            },
        );
        table.record_internal_error(vendor_failure, "KeyError".to_string());
        let state = table.end(vendor_failure).unwrap();
        assert_eq!(
            state.internal_error_type, None,
            "a vendor-caused failure already explains the outcome; internal_error_type must stay null"
        );
    }

    #[test]
    fn concurrent_handles_from_multiple_threads_never_cross_contaminate() {
        // This is the Rust-native replacement for what a Python-only design
        // would have needed a contextvars-across-rayon-threads test for:
        // each thread here owns a distinct handle (as the native embed
        // adapters will), and correctness is enforced by the explicit
        // handle, not by any thread-local/contextvars magic.
        let table = Arc::new(CommandTable::default());
        let handles: Vec<u64> = (0..8)
            .map(|i| table.start(format!("index-{i}"), "cli".to_string(), json!({"n": i})))
            .collect();

        let workers: Vec<_> = handles
            .iter()
            .copied()
            .enumerate()
            .map(|(i, handle)| {
                let table = Arc::clone(&table);
                thread::spawn(move || {
                    for _ in 0..50 {
                        table.record_provider_call(
                            handle,
                            call("embedding", "openai", "text-embedding-3", true),
                        );
                    }
                    let _ = i;
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }

        for handle in handles {
            let state = table.end(handle).unwrap();
            let stats = state
                .providers
                .get(&(
                    "embedding".to_string(),
                    "openai".to_string(),
                    "text-embedding-3".to_string(),
                ))
                .unwrap();
            assert_eq!(
                stats.calls, 50,
                "each command's tally must reflect only its own thread's calls"
            );
        }
    }
}
