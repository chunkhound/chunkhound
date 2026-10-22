//! `AnalyticsRecorder` — the PyO3-exposed core. Owns the local buffer file,
//! a background flush thread, and the S3 upload. Every public method here
//! (Python-facing or Rust-internal) must be panic-free and must never let an
//! internal I/O/config failure disrupt the caller — see `mod.rs`.

use super::command::CommandTable;
use super::identity;
use super::repository::resolve_repository_name;
use super::s3::S3Target;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::FromPyObject;
use rand::RngCore;
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

struct Config {
    flush_interval: Duration,
    flush_batch_size: usize,
    max_buffer_bytes: u64,
    max_upload_retries: usize,
    buffer_dir: PathBuf,
    s3_target: Option<S3Target>,
}

struct BufferState {
    active_path: PathBuf,
    lines: usize,
    bytes: u64,
    last_flush: Instant,
}

pub(crate) struct Inner {
    config: Config,
    commands: CommandTable,
    repository: String,
    payload_identity: Option<String>,
    key_segment: String,
    /// The *Python package's* version (`chunkhound.__version__`, hatch-vcs
    /// derived), passed in from Python at construction time — deliberately
    /// not `env!("CARGO_PKG_VERSION")`, which is this crate's own fixed
    /// `Cargo.toml` version and not what the wiki's `chunkhound_version`
    /// field means.
    chunkhound_version: String,
    buffer: Mutex<BufferState>,
    http: reqwest::blocking::Client,
    shutdown: AtomicBool,
}

#[pyclass]
pub struct AnalyticsRecorder {
    /// `None` when analytics is disabled or misconfigured — every method
    /// below degrades to a silent no-op in that case, by construction.
    inner: Option<Arc<Inner>>,
    flush_thread: Mutex<Option<JoinHandle<()>>>,
}

#[pymethods]
impl AnalyticsRecorder {
    /// `config` keys: enabled (bool), privacy_mode (str), s3_endpoint_url
    /// (str|None), s3_bucket (str|None), s3_access_key (str|None),
    /// s3_secret_key (str|None), flush_interval_seconds (int),
    /// flush_batch_size (int), max_upload_retries (int — failed-upload
    /// attempts before a buffered file is dropped, floored at 1), buffer_dir
    /// (str), salt_path (str), repository_dir (str), os_username (str),
    /// chunkhound_version (str — pass `chunkhound.__version__`, not this
    /// crate's own version).
    #[new]
    fn new(py: Python<'_>, config: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Extracting from the PyDict needs the GIL, but everything
        // build_inner_from_raw() does with the extracted (owned) RawConfig
        // -- resolving the repository name via a synchronous `git` shell-out,
        // creating the buffer dir, building the reqwest client, reading/
        // writing the salt file -- is blocking I/O with no reason to hold
        // the GIL for it, so it runs under allow_threads(), same pattern as
        // shutdown().
        let raw = extract_raw_config(config);
        let inner = py.allow_threads(|| raw.and_then(build_inner_from_raw));
        let mut recorder = Self {
            inner: inner.map(Arc::new),
            flush_thread: Mutex::new(None),
        };
        recorder.spawn_flush_thread();
        Ok(recorder)
    }

    /// Returns an opaque handle id. Pass it into every subsequent call for
    /// this command. `action` must be a JSON-serialized object (built with
    /// `json.dumps` on the Python side) — a malformed payload degrades to
    /// `{}` rather than raising, since a bad action string must never break
    /// the host command.
    #[pyo3(signature = (command, source, action))]
    fn start_command(&self, command: String, source: String, action: String) -> u64 {
        let Some(inner) = &self.inner else { return 0 };
        let action_value: serde_json::Value =
            serde_json::from_str(&action).unwrap_or_else(|_| json!({}));
        inner.commands.start(command, source, action_value)
    }

    #[pyo3(signature = (handle, kind, provider, model, success, error_type=None, input_tokens=None, output_tokens=None))]
    #[allow(clippy::too_many_arguments)] // Mirrors the wiki's own record_provider_call() signature.
    fn record_provider_call(
        &self,
        handle: u64,
        kind: String,
        provider: String,
        model: String,
        success: bool,
        error_type: Option<String>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    ) {
        let Some(inner) = &self.inner else { return };
        inner.commands.record_provider_call(
            handle,
            super::command::ProviderCall {
                kind: &kind,
                provider: &provider,
                model: &model,
                success,
                error_type: error_type.as_deref(),
                input_tokens,
                output_tokens,
            },
        );
    }

    fn record_internal_error(&self, handle: u64, error_type: String) {
        let Some(inner) = &self.inner else { return };
        inner.commands.record_internal_error(handle, error_type);
    }

    /// Merges `action` (a JSON-serialized object) into a still-open
    /// command's action fields -- for fields only known after the command
    /// runs (e.g. `index`'s `file_count`/`total_chunks`). See
    /// `CommandTable::update_action`.
    fn update_action(&self, handle: u64, action: String) {
        let Some(inner) = &self.inner else { return };
        if let Ok(value) = serde_json::from_str(&action) {
            inner.commands.update_action(handle, value);
        }
    }

    fn end_command(&self, py: Python<'_>, handle: u64, success: bool) {
        let Some(inner) = &self.inner else { return };
        let Some(state) = inner.commands.end(handle) else {
            return;
        };
        let event = build_event(inner, &state, success);
        // Deliberately does NOT call maybe_flush()/flush_active() here: that
        // path can do a synchronous, blocking HTTPS PUT to S3 (upload_and_
        // cleanup), which is unbounded by anything shorter than the http
        // client's 30s timeout -- flushing inline would stall the calling
        // host command, and on a single-threaded asyncio MCP server every
        // *other* concurrent tool call too, directly violating this
        // module's own "must never disrupt a host command" invariant (see
        // mod.rs). The background thread spawned by spawn_flush_thread()
        // polls maybe_flush(false) every second on its own OS thread with
        // no GIL involved at all, so a batch-size- or interval-triggered
        // flush still happens -- just within ~1s instead of inline here,
        // which is a negligible delay for an hours-scale-default feature.
        //
        // record_event() itself still does a blocking local-disk open+
        // append (bounded by disk I/O, not network, but still I/O this
        // pymethod must not do while holding the GIL -- a slow/contended/
        // network-mounted buffer dir would otherwise stall every other
        // concurrent tool call the same way an inline S3 PUT would).
        // Release the GIL for exactly that call, same pattern as
        // shutdown()/new() above -- `inner`/`event` are owned/Arc-shared
        // data, no Python objects are touched while the GIL is released.
        py.allow_threads(|| inner.record_event(event));
    }

    /// Best-effort final flush, bounded by `timeout_ms` so a slow/unreachable
    /// S3 endpoint can never hang process exit. Call from the CLI's
    /// `finally` and the MCP server's shutdown path.
    #[pyo3(signature = (timeout_ms=3000))]
    fn shutdown(&self, py: Python<'_>, timeout_ms: u64) {
        let Some(inner) = self.inner.clone() else {
            return;
        };
        py.allow_threads(|| shutdown_blocking(inner, timeout_ms));
    }
}

impl AnalyticsRecorder {
    fn spawn_flush_thread(&mut self) {
        let Some(inner) = self.inner.clone() else {
            return;
        };
        let handle = std::thread::spawn(move || {
            while !inner.shutdown.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_secs(1));
                inner.maybe_flush(false);
            }
        });
        *self.flush_thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
    }

    /// Extracts a clone of the shared inner state for Rust-internal callers
    /// that already hold a reference to this recorder from Python (e.g.
    /// `IndexingPipeline::run()`, which receives `Py<AnalyticsRecorder>`).
    /// Once extracted, `Inner::record_provider_call` can be called directly
    /// from deep inside the native embedding pipeline — including from
    /// rayon worker threads — with zero further PyO3/GIL round-trips per
    /// attempt. `None` when analytics is disabled, matching every other
    /// method's silent-no-op behavior.
    pub(crate) fn inner_arc(&self) -> Option<Arc<Inner>> {
        self.inner.clone()
    }
}

/// Best-effort final flush bounded by `timeout_ms`. A free function over an
/// owned `Arc<Inner>` (not a `&self` method) specifically so the flush can
/// run on a genuinely detached `thread::spawn` thread — a scoped thread
/// (`thread::scope`) would block this call until the flush finishes
/// regardless of the timeout, defeating the entire point. If the timeout
/// elapses first, the flush keeps running in the background (best-effort,
/// not cancelled) while this call returns anyway. Marks `inner.shutdown`
/// first so the background flush-loop thread (`spawn_flush_thread`) stops
/// polling once this call is underway. Pure Rust, no `py`/GIL needed —
/// `AnalyticsRecorder::shutdown` just calls this inside `py.allow_threads()`,
/// which lets this exact timeout-bounding behavior be exercised directly by
/// `cargo test`.
fn shutdown_blocking(inner: Arc<Inner>, timeout_ms: u64) {
    inner.shutdown.store(true, Ordering::SeqCst);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        inner.maybe_flush(true);
        let _ = tx.send(());
    });
    let _ = rx.recv_timeout(Duration::from_millis(timeout_ms));
}

/// Plain-Rust mirror of the Python config dict — kept separate from the
/// PyO3 boundary specifically so `build_inner_from_raw` (the actual
/// construction logic) can be exercised by `cargo test` without a Python
/// interpreter/GIL involved at all.
struct RawConfig {
    enabled: bool,
    privacy_mode: String,
    s3_endpoint_url: Option<String>,
    s3_bucket: Option<String>,
    s3_access_key: Option<String>,
    s3_secret_key: Option<String>,
    flush_interval_seconds: u64,
    flush_batch_size: usize,
    max_buffer_bytes: u64,
    max_upload_retries: usize,
    buffer_dir: PathBuf,
    salt_path: PathBuf,
    repository_dir: PathBuf,
    os_username: String,
    /// `chunkhound.__version__` as passed from Python. Defaulting to
    /// `"unknown"` (not this crate's own version) if never supplied, since
    /// a caller that omits it has no other correct value to fall back to.
    chunkhound_version: String,
}

impl Default for RawConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            privacy_mode: "full".to_string(),
            s3_endpoint_url: None,
            s3_bucket: None,
            s3_access_key: None,
            s3_secret_key: None,
            flush_interval_seconds: 21600,
            flush_batch_size: 500,
            max_buffer_bytes: 10 * 1024 * 1024,
            max_upload_retries: 10,
            buffer_dir: PathBuf::from("."),
            salt_path: PathBuf::from("salt"),
            repository_dir: PathBuf::from("."),
            os_username: "unknown".to_string(),
            chunkhound_version: "unknown".to_string(),
        }
    }
}

/// GIL-bound extraction only -- no I/O. Must run before `py.allow_threads()`
/// since it borrows the PyDict; the returned `RawConfig` is fully owned so
/// the actual (blocking) construction work in `build_inner_from_raw` can run
/// with the GIL released. See `AnalyticsRecorder::new`.
fn extract_raw_config(config: &Bound<'_, PyDict>) -> Option<RawConfig> {
    let buffer_dir: String = get(config, "buffer_dir")?;
    let salt_path: String = get(config, "salt_path")?;
    Some(RawConfig {
        enabled: get(config, "enabled").unwrap_or(false),
        privacy_mode: get(config, "privacy_mode").unwrap_or_else(|| "full".to_string()),
        s3_endpoint_url: get(config, "s3_endpoint_url"),
        s3_bucket: get(config, "s3_bucket"),
        s3_access_key: get(config, "s3_access_key"),
        s3_secret_key: get(config, "s3_secret_key"),
        flush_interval_seconds: get(config, "flush_interval_seconds").unwrap_or(21600),
        flush_batch_size: get(config, "flush_batch_size").unwrap_or(500),
        max_upload_retries: get(config, "max_upload_retries").unwrap_or(10),
        buffer_dir: PathBuf::from(buffer_dir),
        salt_path: PathBuf::from(salt_path),
        repository_dir: PathBuf::from(
            get(config, "repository_dir").unwrap_or_else(|| ".".to_string()),
        ),
        os_username: get(config, "os_username").unwrap_or_else(|| "unknown".to_string()),
        chunkhound_version: get(config, "chunkhound_version")
            .unwrap_or_else(|| "unknown".to_string()),
        ..RawConfig::default()
    })
}

fn build_inner_from_raw(raw: RawConfig) -> Option<Inner> {
    if !raw.enabled {
        return None;
    }

    let s3_target = match (
        &raw.s3_endpoint_url,
        &raw.s3_bucket,
        &raw.s3_access_key,
        &raw.s3_secret_key,
    ) {
        (Some(url), Some(bucket), Some(key), Some(secret)) => {
            match S3Target::new(url, bucket, key, secret) {
                Ok(target) => Some(target),
                Err(e) => {
                    log::warn!("analytics: disabling upload, invalid S3 config: {e}");
                    None
                }
            }
        }
        _ => {
            log::debug!("analytics: no S3 target configured, buffering locally only");
            None
        }
    };

    if let Err(e) = fs::create_dir_all(&raw.buffer_dir) {
        log::warn!("analytics: disabling, cannot create buffer dir: {e}");
        return None;
    }
    let repository = resolve_repository_name(&raw.repository_dir);
    let payload_identity =
        identity::payload_identity(&raw.privacy_mode, &raw.os_username, &raw.salt_path);
    let key_segment = identity::object_key_segment(&raw.privacy_mode, &payload_identity);

    let active_path = new_active_buffer_path(&raw.buffer_dir);
    let http = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(client) => client,
        Err(e) => {
            log::warn!("analytics: disabling, failed to build HTTP client: {e}");
            return None;
        }
    };

    Some(Inner {
        config: Config {
            flush_interval: Duration::from_secs(raw.flush_interval_seconds),
            flush_batch_size: raw.flush_batch_size,
            max_buffer_bytes: raw.max_buffer_bytes,
            max_upload_retries: raw.max_upload_retries.max(1),
            buffer_dir: raw.buffer_dir,
            s3_target,
        },
        commands: CommandTable::default(),
        repository,
        payload_identity,
        key_segment,
        chunkhound_version: raw.chunkhound_version,
        buffer: Mutex::new(BufferState {
            active_path,
            lines: 0,
            bytes: 0,
            last_flush: Instant::now(),
        }),
        http,
        shutdown: AtomicBool::new(false),
    })
}

fn get<'py, T: FromPyObject<'py>>(config: &Bound<'py, PyDict>, key: &str) -> Option<T> {
    config.get_item(key).ok().flatten()?.extract().ok()
}

fn new_active_buffer_path(buffer_dir: &Path) -> PathBuf {
    let pid = std::process::id();
    let start_ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros())
        .unwrap_or(0);
    buffer_dir.join(format!("buffer-{pid}-{start_ts}.jsonl"))
}

fn build_event(
    inner: &Inner,
    state: &super::command::CommandState,
    success: bool,
) -> serde_json::Value {
    let mut providers = serde_json::Map::new();
    for ((kind, provider, model), stats) in &state.providers {
        let entry = json!({
            "provider": provider,
            "model": model,
            "calls": stats.calls,
            "fails": stats.fails,
            "error_types": stats.error_types,
            "input_tokens": stats.input_tokens,
            "output_tokens": stats.output_tokens,
        });
        providers
            .entry(kind.clone())
            .or_insert_with(|| serde_json::Value::Array(Vec::new()))
            .as_array_mut()
            .expect("provider bucket is always initialized as an array")
            .push(entry);
    }
    json!({
        "type": "command_summary",
        "user": inner.payload_identity,
        "ts": iso8601_now(),
        "repository": inner.repository,
        "os": python_style_os_name(),
        "chunkhound_version": inner.chunkhound_version,
        "command": state.command,
        "source": state.source,
        "duration_ms": state.started_at.elapsed().as_millis() as u64,
        "success": success,
        "action": state.action,
        "providers": providers,
        "internal_error_type": state.internal_error_type,
    })
}

/// Matches Python's `platform.system()` output (`"Linux"`/`"Darwin"`/
/// `"Windows"`), not Rust's `std::env::consts::OS` (`"linux"`/`"macos"`/
/// `"windows"`) — the wiki's examples and any downstream consumer expect
/// the Python spelling, and this field must be identical regardless of
/// which language recorded a given event.
fn python_style_os_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "Darwin",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    }
}

fn iso8601_now() -> String {
    use time::format_description::well_known::Rfc3339;
    time::OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string())
}

impl Inner {
    /// Rust-internal equivalent of `AnalyticsRecorder::record_provider_call`
    /// — for callers that already hold an `Arc<Inner>` (via `inner_arc()`)
    /// rather than a `Py<AnalyticsRecorder>`, so no GIL is needed per call.
    /// Unknown handle is a silent no-op, matching every other call site.
    pub(crate) fn record_provider_call(&self, handle: u64, call: super::command::ProviderCall<'_>) {
        self.commands.record_provider_call(handle, call);
    }

    /// Test-only: lets `embed::openai`/`embed::voyageai`'s own tests open a
    /// command and inspect its accumulated `providers` rollup directly,
    /// without going through the PyO3 boundary at all.
    #[cfg(test)]
    pub(crate) fn test_start_command(&self) -> u64 {
        self.commands
            .start("test".to_string(), "test".to_string(), json!({}))
    }

    #[cfg(test)]
    pub(crate) fn test_end_command(&self, handle: u64) -> Option<super::command::CommandState> {
        self.commands.end(handle)
    }

    fn record_event(&self, event: serde_json::Value) {
        let line = match serde_json::to_string(&event) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("analytics: failed to serialize event, dropping: {e}");
                return;
            }
        };
        let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
        let added = line.len() as u64 + 1;
        if buffer.bytes + added > self.config.max_buffer_bytes {
            log::debug!("analytics: buffer size cap reached, dropping event");
            return;
        }
        let result = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&buffer.active_path)
            .and_then(|mut f| writeln!(f, "{line}"));
        match result {
            Ok(()) => {
                buffer.lines += 1;
                buffer.bytes += added;
            }
            Err(e) => log::warn!("analytics: failed to write buffer file: {e}"),
        }
    }

    fn maybe_flush(&self, force: bool) {
        let should_flush = {
            let buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            force
                || buffer.lines >= self.config.flush_batch_size
                || (buffer.lines > 0 && buffer.last_flush.elapsed() >= self.config.flush_interval)
        };
        if should_flush {
            self.flush_active();
        }
        self.sweep_orphans();
    }

    /// Atomic rename-before-upload, not read-then-delete: any `record_event`
    /// racing with this rename either lands in the file before the rename
    /// (included in this flush) or starts a fresh active file after it
    /// (included in the next flush) — never lost, never interleaved with the
    /// upload read.
    fn flush_active(&self) {
        if self.config.s3_target.is_none() {
            // No upload target: rotating would just pile up ".pending-*"
            // files that upload_and_cleanup()'s early return never touches
            // (see below), growing without bound on a long-running process.
            // Stay on the active file instead -- record_event()'s
            // max_buffer_bytes check already bounds its size -- and bump
            // last_flush so maybe_flush() doesn't re-enter here every second
            // once the interval elapses.
            self.buffer
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .last_flush = Instant::now();
            return;
        }
        let (old_active, rotated_path) = {
            let mut buffer = self.buffer.lock().unwrap_or_else(|e| e.into_inner());
            let old_active = buffer.active_path.clone();
            let rotated_path = rotated_path_for(&old_active, 1);
            buffer.active_path = new_active_buffer_path(&self.config.buffer_dir);
            buffer.lines = 0;
            buffer.bytes = 0;
            buffer.last_flush = Instant::now();
            (old_active, rotated_path)
        };
        match fs::rename(&old_active, &rotated_path) {
            Ok(()) => self.upload_and_cleanup(&rotated_path),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                // Nothing was ever written this cycle; not an error.
            }
            Err(e) => log::warn!("analytics: failed to rotate buffer file: {e}"),
        }
    }

    /// Picks up buffer files left behind by a crashed prior process — never
    /// this recorder's own current active file. Matched by filename pattern
    /// and idle mtime, not PID liveness, since PIDs get reused.
    fn sweep_orphans(&self) {
        let Ok(entries) = fs::read_dir(&self.config.buffer_dir) else {
            return;
        };
        let own_active = self
            .buffer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .active_path
            .clone();
        let idle_threshold = self.config.flush_interval * 2;
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if path == own_active {
                continue;
            }
            let is_pending = name.contains(".pending-");
            let is_active_looking =
                name.starts_with("buffer-") && name.ends_with(".jsonl") && !is_pending;
            if !is_pending && !is_active_looking {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            let Ok(age) = SystemTime::now().duration_since(modified) else {
                continue;
            };
            if age < idle_threshold {
                continue;
            }
            if is_pending {
                self.upload_and_cleanup(&path);
            } else {
                let rotated = rotated_path_for(&path, 1);
                match fs::rename(&path, &rotated) {
                    Ok(()) => self.upload_and_cleanup(&rotated),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        // Another process's sweep won the race; harmless.
                    }
                    Err(e) => log::warn!("analytics: failed to rotate orphan buffer file: {e}"),
                }
            }
        }
    }

    fn upload_and_cleanup(&self, path: &Path) {
        let Some(target) = &self.config.s3_target else {
            // No S3 configured: leave the file on disk. flush_active() no
            // longer rotates into a fresh pending file on every cycle in
            // this mode (see its own early return), so this path is now
            // only reached for a stale orphan left by a crashed prior
            // process (sweep_orphans) -- a harmless, bounded no-op until a
            // target is configured, deliberately not subject to the retry
            // cap below since there's no misconfiguration to recover from
            // here, just an intentional local-only mode.
            return;
        };
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                log::warn!("analytics: failed to read buffer file for upload: {e}");
                return;
            }
        };
        if data.is_empty() {
            let _ = fs::remove_file(path);
            return;
        }
        // Computed before the move into put_object() below -- only actually
        // used if this attempt fails and turns out to be the last one.
        let event_count = data.iter().filter(|&&b| b == b'\n').count();
        let key = self.object_key();
        match target.put_object(&self.http, &key, data) {
            Ok(()) => {
                if let Err(e) = fs::remove_file(path) {
                    log::warn!("analytics: uploaded but failed to remove local buffer: {e}");
                }
            }
            Err(e) => {
                let attempt = pending_attempt(path);
                if attempt >= self.config.max_upload_retries {
                    log::warn!(
                        "analytics: dropping buffer file after {attempt} failed upload \
                         attempt(s), {event_count} event(s) lost ({}): {e}",
                        path.display()
                    );
                    let _ = fs::remove_file(path);
                } else {
                    log::warn!(
                        "analytics: upload failed (attempt {attempt} of {}), will retry: {e}",
                        self.config.max_upload_retries
                    );
                    let bumped = rotated_path_for(path, attempt + 1);
                    if let Err(rename_err) = fs::rename(path, &bumped) {
                        log::warn!(
                            "analytics: failed to bump retry-attempt filename: {rename_err}"
                        );
                    }
                }
            }
        }
    }

    fn object_key(&self) -> String {
        use time::format_description::well_known::Rfc3339;
        let now = time::OffsetDateTime::now_utc();
        const DATE_FORMAT: &[time::format_description::BorrowedFormatItem<'_>] =
            time::macros::format_description!("[year]/[month]/[day]");
        let date = now
            .format(DATE_FORMAT)
            .unwrap_or_else(|_| "1970/01/01".to_string());
        let ts = now
            .format(&Rfc3339)
            .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_string());
        let mut suffix = [0u8; 8];
        rand::thread_rng().fill_bytes(&mut suffix);
        let suffix_hex: String = suffix.iter().map(|b| format!("{b:02x}")).collect();
        format!(
            "analytics/{}/{}/{}/{}_{}.jsonl",
            self.repository, self.key_segment, date, ts, suffix_hex
        )
    }
}

/// Strips an existing `.pending-attemptN-{hex}` suffix, if present, so the
/// same helper can both rotate a fresh active file (`attempt = 1`) and
/// re-rotate an already-pending file onto its next attempt number.
fn base_before_pending(path: &Path) -> PathBuf {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return path.to_path_buf();
    };
    match name.find(".pending-") {
        Some(idx) => path.with_file_name(&name[..idx]),
        None => path.to_path_buf(),
    }
}

/// The failed-upload attempt count encoded in a `.pending-attemptN-{hex}`
/// filename, read back out so a retry surviving a flush/sweep cycle (or a
/// crash + restart -- `sweep_orphans()` finds the same file) picks up where
/// it left off instead of resetting to 0. A file with no attempt marker
/// (e.g. a legacy orphan from before this scheme existed) is treated as
/// attempt 1, the same as a file rotated for the first time.
fn pending_attempt(path: &Path) -> usize {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return 1;
    };
    let Some(idx) = name.find(".pending-attempt") else {
        return 1;
    };
    let rest = &name[idx + ".pending-attempt".len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(1)
}

fn rotated_path_for(path: &Path, attempt: usize) -> PathBuf {
    let base = base_before_pending(path);
    let mut suffix = [0u8; 4];
    rand::thread_rng().fill_bytes(&mut suffix);
    let suffix_hex: String = suffix.iter().map(|b| format!("{b:02x}")).collect();
    let mut rotated = base.into_os_string();
    rotated.push(format!(".pending-attempt{attempt}-{suffix_hex}"));
    PathBuf::from(rotated)
}

/// Test-only helper for other modules (e.g. `embed::openai`, `embed::voyageai`)
/// that need a real, enabled `Inner` to exercise their own analytics wiring
/// without going through the PyO3 boundary at all.
#[cfg(test)]
pub(crate) fn test_inner(dir: &Path) -> Arc<Inner> {
    let raw = RawConfig {
        buffer_dir: dir.to_path_buf(),
        salt_path: dir.join("salt"),
        repository_dir: dir.to_path_buf(),
        ..RawConfig::default()
    };
    Arc::new(build_inner_from_raw(raw).expect("enabled RawConfig always builds an Inner"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;
    use tempfile::tempdir;

    fn raw_config(buffer_dir: &Path) -> RawConfig {
        RawConfig {
            buffer_dir: buffer_dir.to_path_buf(),
            salt_path: buffer_dir.join("salt"),
            repository_dir: buffer_dir.to_path_buf(),
            ..RawConfig::default()
        }
    }

    fn count_buffer_files(dir: &Path) -> usize {
        fs::read_dir(dir).unwrap().count()
    }

    #[test]
    fn disabled_config_yields_no_inner() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.enabled = false;
        assert!(build_inner_from_raw(raw).is_none());
    }

    #[test]
    fn record_event_appends_a_line_and_updates_counters() {
        let dir = tempdir().unwrap();
        let inner = build_inner_from_raw(raw_config(dir.path())).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.record_event(json!({"type": "command_summary", "n": 2}));

        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(buffer.lines, 2);
        let contents = fs::read_to_string(&buffer.active_path).unwrap();
        assert_eq!(contents.lines().count(), 2);
    }

    #[test]
    fn end_command_never_flushes_inline_even_when_batch_threshold_is_hit() {
        // Regression guard for the Critical GIL-blocking bug: end_command()
        // must not call maybe_flush()/flush_active() itself, since that path
        // can do a synchronous S3 PUT while the caller (a pymethod) still
        // holds the GIL. This replicates end_command()'s exact body --
        // commands.end() -> build_event() -> record_event(), nothing else --
        // against a threshold that would trigger an immediate flush if
        // maybe_flush() were (still, or ever again) called from here.
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.flush_batch_size = 1;
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = build_inner_from_raw(raw).unwrap();

        let handle = inner
            .commands
            .start("search".to_string(), "mcp".to_string(), json!({}));
        let state = inner.commands.end(handle).unwrap();
        let event = build_event(&inner, &state, true);
        inner.record_event(event);

        // No rotation/upload happened: the active buffer file is still the
        // active file, and no ".pending-*" file was ever created.
        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(buffer.lines, 1);
        assert!(buffer.active_path.exists());
        drop(buffer);
        // Just the active buffer file -- "full" privacy mode never touches
        // the salt file, and nothing else has been created yet.
        assert_eq!(count_buffer_files(dir.path()), 1);

        // The background flush thread's own trigger still works against the
        // same state -- this isn't a dead threshold, it's just not called
        // from end_command() anymore. An S3 target is configured here (unlike
        // the shared raw_config() default) specifically so this exercises the
        // rotate-and-upload path -- the no-target/no-rotate behavior has its
        // own dedicated test in no_s3_target_configured_never_rotates_the_active_file.
        inner.maybe_flush(false);
        mock.assert();
        assert_eq!(
            fs::read_dir(dir.path())
                .unwrap()
                .filter(|e| e
                    .as_ref()
                    .unwrap()
                    .path()
                    .to_string_lossy()
                    .contains(".pending-"))
                .count(),
            0,
            "successful upload removes the rotated pending file"
        );
    }

    #[test]
    fn size_cap_drops_events_without_blocking() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.max_buffer_bytes = 10; // smaller than a single serialized event
        let inner = build_inner_from_raw(raw).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(
            buffer.lines, 0,
            "event over the cap must be dropped, not written"
        );
    }

    #[test]
    fn flush_active_rotates_and_uploads_then_clears_local_buffer() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active();

        mock.assert();
        assert_eq!(
            count_buffer_files(dir.path()),
            0, // the rotated file was uploaded and removed; the fresh active
            // path isn't created on disk until the next record_event()
            "the uploaded rotated file must be removed after a successful upload"
        );
    }

    #[test]
    fn flush_leaves_buffer_intact_when_upload_fails() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(500);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active();

        mock.assert();
        let pending: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .collect();
        assert_eq!(
            pending.len(),
            1,
            "a failed upload must leave the rotated file on disk for the next retry"
        );
        // The very first failure immediately bumps attempt 1 -> 2 (see
        // upload_and_cleanup()'s retry-cap branch) -- attempt numbering
        // itself is covered in detail by
        // failed_upload_bumps_the_attempt_number_across_retries below.
        assert!(pending[0]
            .file_name()
            .to_string_lossy()
            .contains(".pending-attempt2-"));
    }

    #[test]
    fn failed_upload_bumps_the_attempt_number_across_retries() {
        // A file that fails upload, survives one retry cycle (attempt 1 ->
        // 2), and is picked back up by the orphan sweep (standing in for a
        // later flush tick or a crash+restart) must carry its attempt count
        // forward rather than resetting to 1.
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(500);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        raw.max_upload_retries = 5;
        raw.flush_interval_seconds = 0; // idle_threshold = 0: sweep picks it up immediately
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active(); // attempt 1 fails, renamed to attempt2

        let after_first_failure: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .collect();
        assert_eq!(after_first_failure.len(), 1);
        assert!(after_first_failure[0]
            .file_name()
            .to_string_lossy()
            .contains(".pending-attempt2-"));

        inner.sweep_orphans(); // attempt 2 fails, renamed to attempt3

        mock.assert_hits(2);
        let after_second_failure: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .collect();
        assert_eq!(after_second_failure.len(), 1);
        assert!(after_second_failure[0]
            .file_name()
            .to_string_lossy()
            .contains(".pending-attempt3-"));
    }

    #[test]
    fn upload_failures_drop_the_file_once_max_retries_is_exceeded() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(500);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        raw.max_upload_retries = 2;
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();

        inner.record_event(json!({"type": "command_summary", "n": 1}));
        inner.flush_active(); // attempt 1 fails -> renamed to attempt2
        inner.sweep_orphans(); // attempt 2 fails, == max_upload_retries -> dropped

        mock.assert_hits(2);
        assert_eq!(
            count_buffer_files(dir.path()),
            0,
            "a buffer file must be dropped, not retried forever, once it exceeds max_upload_retries"
        );
    }

    #[test]
    fn pending_attempt_reads_the_filename_and_defaults_to_one() {
        assert_eq!(
            pending_attempt(Path::new("buffer-1-2.jsonl.pending-attempt3-ab12cd34")),
            3
        );
        assert_eq!(
            pending_attempt(Path::new("buffer-1-2.jsonl.pending-attempt1-ab12cd34")),
            1
        );
        // No attempt marker at all (e.g. a legacy orphan) defaults to 1,
        // same as a file rotated for the first time.
        assert_eq!(
            pending_attempt(Path::new("buffer-1-2.jsonl.pending-deadbeef")),
            1
        );
        assert_eq!(pending_attempt(Path::new("buffer-1-2.jsonl")), 1);
    }

    #[test]
    fn no_s3_target_configured_never_rotates_the_active_file() {
        // Regression guard for the unbounded-local-disk bug: with no S3
        // target, flush_active() must not rotate into a fresh ".pending-*"
        // file on every cycle -- that would create one new orphaned file
        // per flush_interval/flush_batch_size tick for the life of a
        // long-running process. It should stay on the single active file
        // instead, which record_event()'s own max_buffer_bytes cap bounds.
        let dir = tempdir().unwrap();
        let inner = build_inner_from_raw(raw_config(dir.path())).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));
        let active_path_before = inner.buffer.lock().unwrap().active_path.clone();

        inner.flush_active();
        inner.flush_active();
        inner.flush_active();

        let pending_count = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".pending-"))
            .count();
        assert_eq!(pending_count, 0, "local-only mode must never rotate");
        let buffer = inner.buffer.lock().unwrap();
        assert_eq!(
            buffer.active_path, active_path_before,
            "active file must be unchanged across repeated flushes"
        );
        assert!(buffer.active_path.exists());
        assert_eq!(
            fs::read_to_string(&buffer.active_path)
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn orphan_sweep_uploads_a_stale_active_looking_file() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        // idle_threshold = 2 * flush_interval; zero makes every existing
        // file "old enough" immediately, avoiding any need to fake mtimes.
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();

        // Simulate a file left behind by a crashed prior process: not
        // inner's own active file, matching the active-file naming pattern.
        let orphan_path = dir.path().join("buffer-999999-123.jsonl");
        fs::write(&orphan_path, "{\"type\":\"command_summary\"}\n").unwrap();

        inner.sweep_orphans();

        mock.assert();
        assert!(
            !orphan_path.exists(),
            "orphan must be rotated away and uploaded"
        );
    }

    #[test]
    fn orphan_sweep_uploads_a_leftover_pending_file() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();

        // Simulates a process that rotated but crashed before uploading.
        let pending_path = dir.path().join("buffer-999999-123.jsonl.pending-deadbeef");
        fs::write(&pending_path, "{\"type\":\"command_summary\"}\n").unwrap();

        inner.sweep_orphans();

        mock.assert();
        assert!(!pending_path.exists());
    }

    #[test]
    fn orphan_sweep_never_touches_its_own_active_file() {
        let dir = tempdir().unwrap();
        let mut raw = raw_config(dir.path());
        raw.flush_interval_seconds = 0;
        let inner = build_inner_from_raw(raw).unwrap();
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        inner.sweep_orphans();

        let buffer = inner.buffer.lock().unwrap();
        assert!(
            buffer.active_path.exists(),
            "the recorder's own live active file must never be swept as an orphan"
        );
    }

    #[test]
    fn concurrent_record_event_never_loses_a_write_across_a_racing_flush() {
        // Direct test of the atomic-rename-before-upload guarantee: every
        // event recorded either lands in the file rename picks up, or in
        // the fresh file created immediately after -- never both, never
        // neither.
        let dir = tempdir().unwrap();
        let inner = StdArc::new(build_inner_from_raw(raw_config(dir.path())).unwrap());

        let writer_inner = StdArc::clone(&inner);
        let writer = std::thread::spawn(move || {
            for i in 0..500 {
                writer_inner.record_event(json!({"type": "command_summary", "n": i}));
            }
        });
        for _ in 0..20 {
            inner.flush_active();
            std::thread::sleep(Duration::from_micros(200));
        }
        writer.join().unwrap();
        inner.flush_active(); // pick up whatever's left in the final active file

        let mut seen = std::collections::HashSet::new();
        for entry in fs::read_dir(dir.path()).unwrap().flatten() {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            if !name.starts_with("buffer-") {
                continue;
            }
            for line in fs::read_to_string(&path).unwrap().lines() {
                let value: serde_json::Value = serde_json::from_str(line).unwrap();
                seen.insert(value["n"].as_u64().unwrap());
            }
        }
        assert_eq!(
            seen.len(),
            500,
            "every recorded event must appear exactly once, across all flushes"
        );
    }

    #[test]
    fn shutdown_blocking_flushes_the_buffer() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200);
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = StdArc::new(build_inner_from_raw(raw).unwrap());
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        shutdown_blocking(inner.clone(), 3000);

        mock.assert();
        assert!(inner.shutdown.load(Ordering::SeqCst));
        assert_eq!(count_buffer_files(dir.path()), 0);
    }

    #[test]
    fn shutdown_blocking_never_waits_past_its_timeout() {
        let dir = tempdir().unwrap();
        let server = httpmock::MockServer::start();
        // Slower than the timeout below -- proves shutdown_blocking returns
        // on schedule instead of waiting for the (best-effort, still
        // in-flight) flush to finish.
        let _mock = server.mock(|when, then| {
            when.method(httpmock::Method::PUT);
            then.status(200).delay(Duration::from_millis(300));
        });
        let mut raw = raw_config(dir.path());
        raw.s3_endpoint_url = Some(server.url(""));
        raw.s3_bucket = Some("analytics-bucket".to_string());
        raw.s3_access_key = Some("key".to_string());
        raw.s3_secret_key = Some("secret".to_string());
        let inner = StdArc::new(build_inner_from_raw(raw).unwrap());
        inner.record_event(json!({"type": "command_summary", "n": 1}));

        let started = Instant::now();
        shutdown_blocking(inner, 50);
        assert!(
            started.elapsed() < Duration::from_millis(250),
            "shutdown_blocking must return at its timeout, not wait for the slow flush: took {:?}",
            started.elapsed()
        );
    }
}
