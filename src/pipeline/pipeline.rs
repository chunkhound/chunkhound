//! Unified Rust indexing pipeline — main orchestration class.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::config::PipelineConfig;
use super::differ::DiffResult;
use super::report::PipelineReport;

use crate::db::{check_disk_usage_limit, create_backend, DbBackend, DbConfig};
use crate::error::DbError;
use crate::types::{ChunkRecord, DbFileEntry, DbWriterBatch, FileRecord};

/// The main PyO3 class — Python calls `.run()` from `asyncio.to_thread`.
#[pyclass]
#[derive(Debug)]
pub(crate) struct IndexingPipeline {
    config: PipelineConfig,
}

/// Helper: call `progress_callback(phase, current, total)` if provided.
fn emit_progress(py: Python<'_>, cb: &Option<Py<PyAny>>, phase: &str, current: u64, total: u64) {
    if let Some(ref cb) = cb {
        let _ = cb.bind(py).call1((phase, current, total));
    }
}

/// Helper: call `progress_callback(phase, current, total)` from a
/// non-parse/embed worker thread, acquiring the GIL for the duration
/// of the call.
fn emit_progress_gil(cb: &Option<Py<PyAny>>, phase: &str, current: u64, total: u64) {
    if let Some(ref cb) = cb {
        Python::with_gil(|py| {
            let _ = cb.bind(py).call1((phase, current, total));
        });
    }
}

/// Like `emit_progress_gil` but passes a 4th `chunks` value (cumulative chunks
/// written so far). Used by the write-data phase so the Python progress bar can
/// display true throughput as chunks/s instead of cumulative batches/min. The
/// Python callback takes `chunks` as an optional trailing arg, so 3-arg callers
/// are unaffected.
fn emit_progress_gil_chunks(
    cb: &Option<Py<PyAny>>,
    phase: &str,
    current: u64,
    total: u64,
    chunks: u64,
) {
    if let Some(ref cb) = cb {
        Python::with_gil(|py| {
            let _ = cb.bind(py).call1((phase, current, total, chunks));
        });
    }
}

/// Result of the store thread in the 3-stage streaming pipeline.
struct StoreOutcome {
    chunks_written: u64,
    embeddings_written: u64,
    /// Per-file parse AND embed errors, collected from the parse/embed
    /// threads — merged in after all three threads join (see
    /// `pipeline_parse_embed_store`). Named `errors` (not `parse_errors`)
    /// because it carries both kinds, matching `PipelineReport.errors`.
    errors: Vec<String>,
    /// Set when a mid-run disk-usage check tripped: `(current_mb, limit_mb)`.
    disk_limit_exceeded: Option<(f64, f64)>,
}

/// Result of one `embed_batch_parallel` call — one streamed batch.
struct EmbedBatchOutcome {
    /// Count of chunks that actually received an embedding vector (excludes
    /// chunks in sub-batches whose callback invocation failed).
    embedded: u64,
    /// One formatted `"{file}: embedding failed for {n} chunk(s): {message}"`
    /// entry per (file, failed sub-batch) — mirrors the parse thread's
    /// `parse_errors` format so both flow into `PipelineReport.errors`
    /// identically on the Python side (see pipeline_bridge.py's
    /// `_split_rust_error`).
    errors: Vec<String>,
}

#[pymethods]
impl IndexingPipeline {
    /// Create a new pipeline from a Python configuration dict.
    #[new]
    fn new(config_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let config = PipelineConfig::from_py_dict(config_dict)?;
        Ok(Self { config })
    }

    /// Run the full indexing pipeline synchronously.
    ///
    /// Called from Python via `asyncio.to_thread(pipeline.run, ...)`.
    ///
    /// Pipeline order: parse ∥ embed ∥ store, streamed through bounded
    /// channels (see `pipeline_parse_embed_store`).
    ///
    /// progress_callback receives ``(phase: str, current: int, total: int)``
    /// at phase transitions.  Phases: ``"diff"``, ``"parse"``, ``"embed"``,
    /// ``"write-prepare"``, ``"write-data"``, ``"write-index"``,
    /// ``"write-compact"``, ``"write-done"``, ``"done"``. Only one of
    /// ``"write-index"``/``"write-compact"`` fires per run — compaction
    /// rebuilds indexes as part of its own rewrite, so the two never both
    /// run.
    #[pyo3(signature = (files, parse_batch_callback, embed_batch_callback=None, progress_callback=None, incremental=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        // (absolute_path, relative_key) pairs. relative_key is computed once
        // by Python's get_relative_path_safe() — the single symlink-aware
        // source of truth (git-worktree support: a symlink's logical path is
        // preserved even when its target resolves outside project_root) —
        // instead of being re-derived here via a naive strip_prefix, which
        // previously diverged from Python's DB-write path for symlinked
        // files and could misclassify their DB rows as removed.
        files: Vec<(String, String)>,
        parse_batch_callback: Py<PyAny>,
        embed_batch_callback: Option<Py<PyAny>>,
        progress_callback: Option<Py<PyAny>>,
        incremental: bool,
    ) -> PyResult<PipelineReport> {
        let started = Instant::now();

        if files.is_empty() {
            // Only short-circuit when there's no DB yet to clean up (first-ever
            // run on an empty directory). Otherwise an empty file list must
            // still flow through the incremental diff + streaming pipeline
            // below — files removed from disk since the last run (down to
            // zero) need their orphaned DB rows deleted. See the
            // `pending_delete_paths` flush in `pipeline_parse_embed_store`'s
            // embed thread.
            let no_db_yet = (self.config.db_path.as_os_str().is_empty()
                || self.config.db_path.as_os_str() == ":memory:")
                || !self.config.db_path.join("chunks.db").exists();
            if no_db_yet {
                return Ok(PipelineReport::empty());
            }
        }

        let mut file_count = files.len() as u64;
        let mut batch_paths: Vec<PathBuf> = files.iter().map(|(p, _)| PathBuf::from(p)).collect();
        // Applies to every file for the whole run, independent of the diff's
        // own verdict — built once here rather than routed through
        // `DiffResult`, since `build_db_batch` needs it for files the diff
        // never touches too (e.g. the "no DB yet" early-return path above
        // doesn't apply, but a fresh DB with no prior rows still needs every
        // file's relative key to write its first row).
        let rel_keys: std::collections::HashMap<PathBuf, String> = files
            .into_iter()
            .map(|(p, rel)| (PathBuf::from(p), rel))
            .collect();

        // ── Incremental diff (Phase 3) ─────────────────────────
        let delete_paths: Vec<String>;
        // Content hash for every scanned file the diff already had a DB row
        // for (see `differ::compute_diff`) — freshly computed for files that
        // must be reprocessed, or the DB's existing hash carried forward
        // unchanged otherwise. Covers files that never reach `batch_paths`
        // too (mtime-unchanged, or hash-confirmed unchanged despite a
        // differing mtime) so a force-reindex caller — which reprocesses
        // every file regardless — still writes back the correct hash for
        // one it didn't need to change, instead of nulling it out.
        let new_hashes: std::collections::HashMap<PathBuf, String>;
        let existing_ids: std::collections::HashMap<PathBuf, i64>;
        // (size_bytes, mtime) per scanned file — produced once by the diff
        // phase and reused by parse_one_batch to avoid a third stat pass.
        let disk_stats: std::collections::HashMap<PathBuf, (u64, f64)>;
        let mut files_skipped_by_hash = 0u64;
        if incremental {
            let diff =
                self.compute_diff_blocking(py, &progress_callback, &batch_paths, &rel_keys)?;
            // Only process changed files
            batch_paths = diff.changed;
            // Respect do_cleanup flag: skip orphan deletion when cleanup is disabled.
            delete_paths = if self.config.do_cleanup {
                diff.removed
            } else {
                Vec::new()
            };
            new_hashes = diff.new_hashes;
            existing_ids = diff.existing_ids;
            disk_stats = diff.disk_stats;
            files_skipped_by_hash = diff.skipped_by_hash;
            // Update file_count to reflect what will actually be processed
            file_count = batch_paths.len() as u64;
        } else {
            // force_reindex re-indexes every file, but must still detect and
            // remove orphaned DB rows (files deleted from disk since the last
            // run) — unless cleanup is disabled by config.
            if self.config.do_cleanup {
                let diff =
                    self.compute_diff_blocking(py, &progress_callback, &batch_paths, &rel_keys)?;
                delete_paths = diff.removed;
                // compute_diff_blocking populates new_hashes/existing_ids/disk_stats
                // for every scanned file it already had a DB row for — not just
                // ones it decided need reprocessing — precisely so this branch can
                // reuse that work: batch_paths stays the full file list (every
                // file is reprocessed regardless of the diff's own verdict), and
                // without this, a file the diff correctly deemed unchanged would
                // still get re-stat()'d here and, worse, have its content_hash
                // written back as NULL purely because this branch discarded the
                // hash compute_diff had (or already had, from the DB) on hand.
                new_hashes = diff.new_hashes;
                existing_ids = diff.existing_ids;
                disk_stats = diff.disk_stats;
            } else {
                delete_paths = Vec::new();
                new_hashes = std::collections::HashMap::new();
                existing_ids = std::collections::HashMap::new();
                disk_stats = std::collections::HashMap::new();
            }
        }

        // ── Resolve directory→db file path (shared by both write paths) ──
        let db_file: PathBuf = if self.config.db_path.as_os_str().is_empty()
            || self.config.db_path.as_os_str() == ":memory:"
        {
            PathBuf::from(":memory:")
        } else {
            self.config.db_path.join("chunks.db")
        };

        let db_config = DbConfig {
            db_path: db_file.to_string_lossy().into_owned(),
            compaction_threshold: self.config.compaction_threshold,
            compaction_min_size_bytes: self.config.compaction_min_size_mb * 1024 * 1024,
            insert_batch_size: self.config.db_batch_size.max(1),
        };

        // Ensure parent directory exists (DuckDB doesn't auto-create it).
        if let Some(parent) = db_file.parent() {
            if !parent.as_os_str().is_empty() && parent != PathBuf::from(":memory:").as_path() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to create db directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
        }

        // ── Parse ∥ Embed ∥ Store: 3-stage streaming pipeline ──
        let provider = self.config.embedding_provider.clone();
        let model = self.config.embedding_model.clone();
        let total_files = batch_paths.len() as u64;

        emit_progress(py, &progress_callback, "parse", 0, total_files);

        // Built once per run (invariant across every batch) and handed to
        // every parse_batch_callback() call — see ParseCallConfig's doc
        // comment for why this is a single typed object rather than a
        // growing list of positional arguments.
        let parse_config: Py<super::parse_call_config::ParseCallConfig> = Py::new(
            py,
            super::parse_call_config::ParseCallConfig {
                detect_embedded_sql: self.config.detect_embedded_sql,
                per_file_timeout_secs: self.config.per_file_timeout_secs,
                per_file_timeout_min_size_kb: self.config.per_file_timeout_min_size_kb,
                config_file_size_threshold_kb: self.config.config_file_size_threshold_kb,
                parse_thread_pool_size: self.config.parse_thread_pool_size,
            },
        )?;

        // Clone Python references before releasing the GIL — Py<T>::clone()
        // panics without the GIL held, and this whole call runs inside
        // py.allow_threads() below.
        let parse_cb = parse_batch_callback.clone_ref(py);
        let embed_cb = embed_batch_callback.as_ref().map(|cb| cb.clone_ref(py));
        let progress_cb = progress_callback.as_ref().map(|cb| cb.clone_ref(py));
        let store_progress_cb = progress_callback.as_ref().map(|cb| cb.clone_ref(py));
        let t_run = Instant::now();
        let outcome = py
            .allow_threads(|| {
                self.pipeline_parse_embed_store(
                    &batch_paths,
                    parse_cb,
                    parse_config,
                    embed_cb,
                    &provider,
                    &model,
                    progress_cb,
                    store_progress_cb,
                    delete_paths,
                    db_config,
                    new_hashes,
                    existing_ids,
                    disk_stats,
                    rel_keys,
                )
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        log::info!(
            "pipeline parse+embed+store: {:.2}s",
            t_run.elapsed().as_secs_f64()
        );

        let total_secs = started.elapsed().as_secs_f64();
        emit_progress(py, &progress_callback, "done", file_count, file_count);
        log::info!(
            "pipeline total: {total_secs:.2}s (files={file_count}, chunks={}, embeds={})",
            outcome.chunks_written,
            outcome.embeddings_written
        );

        Ok(PipelineReport {
            files_processed: file_count,
            files_skipped: files_skipped_by_hash,
            chunks_written: outcome.chunks_written,
            embeddings_generated: outcome.embeddings_written,
            elapsed_secs: total_secs,
            errors: outcome.errors,
            peak_rss_mb: None,
            disk_limit: outcome.disk_limit_exceeded,
        })
    }
}

// ── Internal helpers ────────────────────────────────────────────

impl IndexingPipeline {
    /// Read the DB state and compute which files changed.
    ///
    /// Runs synchronously on the same thread/GIL guard as `run()` (called
    /// before `py.allow_threads(...)` there), so `progress_callback` can be
    /// invoked directly via `emit_progress` — no `clone_ref`/`with_gil`
    /// needed. If this is ever moved onto a separate thread, it will need
    /// the same `clone_ref` + `Python::with_gil` treatment as the
    /// parse/embed/store callbacks below.
    fn compute_diff_blocking(
        &self,
        py: Python<'_>,
        progress_callback: &Option<Py<PyAny>>,
        files: &[PathBuf],
        rel_keys: &std::collections::HashMap<PathBuf, String>,
    ) -> PyResult<DiffResult> {
        let total_files = files.len() as u64;
        emit_progress(py, progress_callback, "diff", 0, total_files);

        let db_file = if self.config.db_path.as_os_str().is_empty()
            || self.config.db_path.as_os_str() == ":memory:"
        {
            emit_progress(py, progress_callback, "diff", total_files, total_files);
            return Ok(DiffResult {
                changed: files.to_vec(),
                removed: Vec::new(),
                ..Default::default()
            });
        } else {
            self.config.db_path.join("chunks.db")
        };

        if !db_file.exists() {
            emit_progress(py, progress_callback, "diff", total_files, total_files);
            return Ok(DiffResult {
                changed: files.to_vec(),
                removed: Vec::new(),
                ..Default::default()
            });
        }

        let db_config = DbConfig {
            db_path: db_file.to_string_lossy().into_owned(),
            compaction_threshold: self.config.compaction_threshold,
            compaction_min_size_bytes: self.config.compaction_min_size_mb * 1024 * 1024,
            insert_batch_size: self.config.db_batch_size.max(1),
        };
        let backend: Box<dyn DbBackend> = create_backend(db_config);
        let db_entries: Vec<DbFileEntry> = backend.read_file_states()?;

        // Read mtime AND size in a single pass so that compute_diff (which
        // needs mtime for change detection) and parse_one_batch (which needs
        // mtime + size for ParsedFile) can both reuse these values instead of
        // calling stat() again — collapsing two stat passes into one.
        //
        // `db_entries`' mtime already reverses the write-side local-timezone
        // cast (see `FILE_STATE_SELECT` in duckdb_backend.rs), so it's
        // directly comparable to these on-disk values with no further
        // normalization needed.
        let precomputed_stats: std::collections::HashMap<std::path::PathBuf, (u64, f64)> = files
            .iter()
            .filter_map(|p| {
                let meta = std::fs::metadata(p).ok()?;
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                Some((p.clone(), (meta.len(), mtime)))
            })
            .collect();

        let result = super::differ::compute_diff(
            &db_entries,
            files,
            rel_keys,
            self.config.mtime_epsilon_seconds,
            Some(&precomputed_stats),
            Some(&mut |current, total| {
                emit_progress(py, progress_callback, "diff", current as u64, total as u64);
            }),
        );

        // Unconditional final tick: small diffs (< DIFF_TICK_INTERVAL files)
        // may never hit the in-loop modulo, so the bar wouldn't otherwise
        // reach 100%.
        emit_progress(py, progress_callback, "diff", total_files, total_files);

        Ok(result)
    }

    // ── Pipeline-parallel parse+embed+store (Phase 8/9) ───────────

    /// Run the full 3-stage streaming pipeline: parse ∥ embed ∥ store.
    ///
    /// Three persistent OS threads, connected by two *bounded* channels
    /// (capacity 2 each): a parse thread batches files and parses them via
    /// the batch callback; a dedicated embed thread receives parsed
    /// batches, embeds them (via its own long-lived rayon pool, built once
    /// for the whole run), converts each embedded batch into a
    /// `DbWriterBatch`, and forwards it to a dedicated store thread —  so
    /// while the store thread writes batch N to DuckDB (and, on the final
    /// batch, rebuilds HNSW indexes and compacts), the embed thread is
    /// already embedding batch N+1, and the parse thread is already
    /// parsing batch N+2. Both bounds provide backpressure: parsed batches
    /// hold source text, and embedded batches additionally hold float
    /// vectors — both are memory-heavy.
    ///
    /// The store thread owns the single DB connection for the whole run,
    /// using the same HNSW "bulk mode" bracket
    /// (`drop_all_hnsw_indexes()` → N incremental writes →
    /// `ensure_all_hnsw_indexes()`) that the Python path already uses for
    /// bulk indexing, so no new DB-layer mechanism is required.
    ///
    /// **Caller must release the GIL** before entering this method.
    /// Each thread re-acquires the GIL independently via ``Python::with_gil()``.
    // Each parameter is moved or borrowed into a different one of the three
    // spawned threads (parse/embed/store) with its own ownership needs
    // (owned `Py<PyAny>` callbacks moved into one specific thread each,
    // borrowed `&str`, owned collections consumed once) -- already collapsed
    // where possible (`parse_config` bundles five scalars into one typed
    // object, see `ParseCallConfig`'s doc comment). Grouping the rest into a
    // struct would just move the same fields behind one more layer without
    // reducing what each thread actually needs to take ownership of.
    #[allow(clippy::too_many_arguments)]
    fn pipeline_parse_embed_store(
        &self,
        files: &[PathBuf],
        parse_cb: Py<PyAny>,
        parse_config: Py<super::parse_call_config::ParseCallConfig>,
        embed_cb: Option<Py<PyAny>>,
        provider: &str,
        model: &str,
        progress_cb: Option<Py<PyAny>>,
        store_progress_cb: Option<Py<PyAny>>,
        delete_paths: Vec<String>,
        db_config: DbConfig,
        new_hashes: std::collections::HashMap<PathBuf, String>,
        existing_ids: std::collections::HashMap<PathBuf, i64>,
        disk_stats: std::collections::HashMap<PathBuf, (u64, f64)>,
        rel_keys: std::collections::HashMap<PathBuf, String>,
    ) -> Result<StoreOutcome, String> {
        use std::sync::mpsc;
        use std::sync::{Arc, Mutex};

        let batch_size = self.config.parse_batch_size.max(1);
        let embed_thread_pool_size = self.config.embed_thread_pool_size;
        let embed_batch_size = self.config.embed_batch_size.max(1);
        let skip_embeddings = self.config.skip_embeddings;
        let disk_usage_limit_mb = self.config.disk_usage_limit_mb;
        let provider = provider.to_string();
        let model = model.to_string();

        // Separate clones of the progress callback for the parse and embed
        // threads — the original `progress_cb` is kept in this function's
        // own scope to emit the final landing progress after both threads
        // join.
        let progress_cb_parse = progress_cb
            .as_ref()
            .map(|cb| Python::with_gil(|py| cb.clone_ref(py)));
        let progress_cb_embed = progress_cb
            .as_ref()
            .map(|cb| Python::with_gil(|py| cb.clone_ref(py)));

        // Build file batches (indices into `files` slice).
        let batches: Vec<(usize, Vec<PathBuf>)> = files
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.to_vec()))
            .collect();
        let batch_count = batches.len();
        let batch_count_u64 = batch_count as u64;
        let total_files = files.len() as u64;

        // Bounded backpressure keeps any one stage from running arbitrarily
        // far ahead of the next — parsed batches hold source text, embedded
        // batches additionally hold float vectors, and both are memory-heavy.
        // The embed->store hop is deeper (8): instrumentation showed embed and
        // store were mutually starving (~600-800s each) through a 2-slot buffer,
        // so a deeper buffer lets embed run ahead during store's write bursts
        // and absorbs embed's large per-batch time variance (2s-28s).
        let (parse_tx, parse_rx) = mpsc::sync_channel::<(usize, Vec<super::types::ParsedFile>)>(2);
        let (store_tx, store_rx) = mpsc::sync_channel::<DbWriterBatch>(8);
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        // Per-file parse errors (e.g. one file's parse callback raised) —
        // these don't abort the run, unlike `error` above, which is for
        // whole-batch-callback failures.
        let parse_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        // Per-file embed errors (e.g. an embed API call for one sub-batch
        // failed) — same non-fatal treatment as `parse_errors`: those chunks
        // are written without an embedding rather than aborting the run.
        let embed_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        // ── Parse thread ──────────────────────────────────────
        let parse_handle = {
            let error = Arc::clone(&error);
            let parse_errors = Arc::clone(&parse_errors);
            std::thread::spawn(move || {
                let mut parsed_files_count: u64 = 0;
                for (batch_idx, batch) in batches {
                    if error.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
                        break;
                    }

                    let t_batch = Instant::now();
                    let paths: Vec<String> = batch
                        .iter()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();

                    let parsed = match Python::with_gil(|gil_py| {
                        Self::parse_one_batch(
                            gil_py,
                            &parse_cb,
                            &parse_config,
                            &paths,
                            &batch,
                            &new_hashes,
                            &disk_stats,
                            &rel_keys,
                        )
                    }) {
                        Ok(parsed) => parsed,
                        Err(e) => {
                            let mut err = error.lock().unwrap_or_else(|e| e.into_inner());
                            if err.is_none() {
                                *err = Some(e);
                            }
                            break;
                        }
                    };

                    {
                        let mut errs = parse_errors.lock().unwrap_or_else(|e| e.into_inner());
                        for pf in &parsed {
                            if let Some(e) = &pf.error {
                                errs.push(format!("{}: {}", pf.path.display(), e));
                            }
                        }
                    }

                    log::debug!(
                        "[parse] batch {batch_idx} done in {:.3}s",
                        t_batch.elapsed().as_secs_f64()
                    );

                    // Report real parse throughput as soon as this batch is
                    // parsed — not gated on the embed thread consuming it —
                    // so the "parse" progress bar reflects actual parse
                    // completion instead of tracking embed's consumption
                    // rate (see the removed piggyback emit in the embed
                    // thread's loop below).
                    parsed_files_count += batch.len() as u64;
                    emit_progress_gil(&progress_cb_parse, "parse", parsed_files_count, total_files);

                    if parse_tx.send((batch_idx, parsed)).is_err() {
                        // Receiver dropped (main thread error) — stop.
                        break;
                    }
                }
            })
        };

        // ── Store thread ───────────────────────────────────────
        // Owns the single DB connection for the whole run. Uses the same
        // HNSW bulk-mode bracket the Python path already relies on for bulk
        // indexing: drop all HNSW indexes once, write each streamed batch
        // incrementally (prepare_write's own HNSW-drop step is a no-op
        // while bulk mode is on), then either compact (which rebuilds
        // indexes as part of its own rewrite) or rebuild indexes directly
        // once at the very end — never both, see the comment below.
        let store_handle: std::thread::JoinHandle<Result<StoreOutcome, String>> = {
            std::thread::spawn(move || {
                // Captured before `db_config` is moved into the backend —
                // used only to stat the DB and WAL files for per-batch size
                // tracing below.
                let db_path = db_config.db_path.clone();
                let mut backend: Box<dyn DbBackend> = create_backend(db_config);
                let open_result = backend.open();
                Self::close_backend_on_err(backend.as_mut(), open_result)?;
                // Dropping HNSW indexes is a catalog-only DDL operation (no
                // data scan), so it's always sub-10ms in practice — not
                // worth a dedicated progress bar (it would flash past
                // unnoticed). Log it instead for the rare case it's slow.
                let t_hnsw_drop = Instant::now();
                let drop_hnsw_result = backend.drop_all_hnsw_indexes();
                Self::close_backend_on_err(backend.as_mut(), drop_hnsw_result)?;
                log::info!(
                    "[hnsw-drop] done in {:.3}s",
                    t_hnsw_drop.elapsed().as_secs_f64()
                );
                emit_progress_gil(&store_progress_cb, "write-prepare", 0, batch_count_u64);

                // Write each batch as soon as it arrives (no window). The
                // 10-batch window was left over from a reverted multi-batch
                // transaction; it now only serves to make the store thread
                // block collecting a full window before writing, which
                // instrumentation showed cost ~793s of store idle time and
                // ping-ponged with the embed stage. Per-batch commits (the
                // default write path) are unchanged. checkpoint_threshold —
                // not commit frequency — now governs WAL checkpoint cost.
                const STORE_COMMIT_INTERVAL: usize = 1;

                let mut chunks_written = 0u64;
                let mut embeddings_written = 0u64;
                let mut batch_no = 0usize;
                // Pipeline diagnostic: cumulative time the store thread spends
                // BLOCKED in recv() waiting for the embed stage to produce.
                let mut store_wait_s = 0f64;
                // Set when a mid-run disk-usage check trips (see below).
                let mut disk_limit_hit: Option<(f64, f64)> = None;

                let write_result: Result<(), String> = (|| {
                    let mut window: Vec<crate::types::DbWriterBatch> =
                        Vec::with_capacity(STORE_COMMIT_INTERVAL);

                    // Labeled so the disk-usage check below can break out of
                    // the whole store loop, not just the inner per-window
                    // `for`.
                    'store_loop: loop {
                        // Pre-write guard, mirroring Python's
                        // _check_disk_usage_limit: is the DB already over the
                        // limit from prior commits, before this window writes
                        // anything? Never inspects whether this window's own
                        // writes would push it over. Checked once per window
                        // — BEFORE any batch in it is received — rather than
                        // once per received batch, so this is safe regardless
                        // of STORE_COMMIT_INTERVAL: either the whole window is
                        // left completely untouched, or it's fully processed
                        // (prepare_write's deletes AND the matching inserts).
                        // Checking mid-fill instead would let a tripped check
                        // land after some batches in the window already had
                        // prepare_write's un-rollback-able deletes applied but
                        // before write_batches_in_one_txn ever inserted their
                        // replacements — silent data loss the moment this
                        // constant is ever raised above 1.
                        if let Some(hit) =
                            check_disk_usage_limit(Path::new(&db_path), disk_usage_limit_mb)
                        {
                            log::warn!(
                                "[store] disk usage limit exceeded: \
                                 {:.1} MB >= {:.1} MB — stopping further writes",
                                hit.0,
                                hit.1
                            );
                            disk_limit_hit = Some(hit);
                            break 'store_loop;
                        }

                        // Fill window: prepare_write per batch (auto-commit),
                        // block on recv until window full or channel closes.
                        window.clear();
                        for _ in 0..STORE_COMMIT_INTERVAL {
                            let t_recv = Instant::now();
                            let recv = store_rx.recv();
                            store_wait_s += t_recv.elapsed().as_secs_f64();
                            match recv {
                                Ok(batch) => {
                                    let t_prep = Instant::now();
                                    backend.prepare_write(&batch).map_err(|e| e.to_string())?;
                                    let prep_ms = t_prep.elapsed().as_secs_f64() * 1e3;
                                    log::debug!(
                                        "[store] prep batch {} in {prep_ms:.1}ms",
                                        window.len() + 1,
                                    );
                                    window.push(batch);
                                    // Emit progress after each prepare_write so the
                                    // chunks/s rate stays live during accumulation.
                                    // chunks_written is the cumulative total through
                                    // the previous window (this batch not yet written).
                                    emit_progress_gil_chunks(
                                        &store_progress_cb,
                                        "write-data",
                                        (batch_no + window.len()) as u64,
                                        batch_count_u64,
                                        chunks_written,
                                    );
                                }
                                Err(_) => break, // channel closed
                            }
                        }
                        if window.is_empty() {
                            break;
                        }

                        // Write the whole window in one transaction.
                        let t_write = Instant::now();
                        let results = backend
                            .write_batches_in_one_txn(&window)
                            .map_err(|e| e.to_string())?;
                        let write_s = t_write.elapsed().as_secs_f64();

                        let db_mib =
                            std::fs::metadata(&db_path).map(|m| m.len()).unwrap_or(0) / (1 << 20);
                        let wal_mib = std::fs::metadata(format!("{db_path}.wal"))
                            .map(|m| m.len())
                            .unwrap_or(0)
                            / (1 << 20);

                        let window_start = batch_no + 1;
                        for result in &results {
                            chunks_written += result.chunks_written;
                            embeddings_written += result.embeddings_written;
                            batch_no += 1;
                        }
                        // Emit final progress for this window now that batch_no and
                        // chunks_written are accurate.
                        emit_progress_gil_chunks(
                            &store_progress_cb,
                            "write-data",
                            batch_no as u64,
                            batch_count_u64,
                            chunks_written,
                        );
                        log::debug!(
                            "[store] window {window_start}-{batch_no} done in {write_s:.3}s \
                             (chunks={chunks_written} embeds={embeddings_written} \
                             db={db_mib}MiB wal={wal_mib}MiB)",
                        );
                    }
                    Ok(())
                })();
                log::debug!("[pipe] store blocked: embed-recv {store_wait_s:.1}s");

                // Force the parse/embed threads' existing cooperative-shutdown
                // cascade (the same mechanism a real store error already
                // triggers): dropping store_rx makes embed's store_tx.send()
                // fail fast, which drops parse_rx, making parse's
                // parse_tx.send() fail — no new shutdown machinery needed.
                // Deliberately stops promptly here rather than mirroring
                // Python's own behavior of continuing to parse/embed the rest
                // of the file list after the trip (a known Python-side
                // inefficiency, not a contract worth reproducing).
                if disk_limit_hit.is_some() {
                    drop(store_rx);
                }

                // Check compaction need BEFORE building the HNSW index, not
                // after. `run_compaction()`'s EXPORT/IMPORT rewrite copies
                // `files`/`chunks`/`embeddings_*` into a fresh database with
                // no indexes, then rebuilds them via `reopen()` — on both
                // its success path (`run_attach_copy_compaction`) and its
                // failure-fallback path (`reopen_after_compaction_failure`).
                // So `run_compaction()` *always* leaves the database with a
                // rebuilt HNSW index. Calling `ensure_all_hnsw_indexes()`
                // beforehand would just mean building the (CPU-intensive,
                // see `SET threads = 8` there) index once, throwing it away
                // during compaction, then building it again from scratch —
                // do not restore that call here without also removing the
                // `run_compaction()` branch's reliance on `reopen()`.
                //
                // Never compact after a disk-limit trip: the EXPORT/IMPORT
                // rewrite needs a full second copy of the database on disk
                // (old + new coexist until the swap), transiently roughly
                // doubling disk usage — the exact thing this run just
                // determined it can't afford. Falls back to the cheaper
                // ensure_all_hnsw_indexes() over whatever was already
                // written; actual compaction is deferred to a future run,
                // where indexing_coordinator.py's pre-run check will catch
                // the still-over-limit condition before any writes start.
                //
                // Chained onto `write_result` (rather than a separate `?`
                // per step) so that a failure at any point — the write loop
                // itself, or compaction/index-rebuild after it — funnels
                // through the single best-effort `backend.close()` below
                // instead of returning early with the connection left open
                // and HNSW indexes un-restored (Invariant 14).
                let post_write_result: Result<(), String> = write_result.and_then(|()| {
                    let needs_compaction = disk_limit_hit.is_none()
                        && backend.needs_compaction().map_err(|e| e.to_string())?;
                    if needs_compaction {
                        emit_progress_gil(&store_progress_cb, "write-compact", 0, 1);
                        backend.run_compaction().map_err(|e| e.to_string())?;
                    } else {
                        if disk_limit_hit.is_some() {
                            log::info!(
                                "[store] skipping compaction after disk-limit trip; \
                                 rebuilding HNSW indexes only"
                            );
                        }
                        emit_progress_gil(&store_progress_cb, "write-index", 0, 1);
                        backend
                            .ensure_all_hnsw_indexes()
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                });

                if let Err(e) = post_write_result {
                    // Best-effort: restore HNSW indexes over whatever was
                    // already committed before surfacing the error
                    // (mirrors the single-shot path's Invariant 14 restore).
                    let _ = backend.close();
                    return Err(e);
                }

                backend.close().map_err(|e| e.to_string())?;
                emit_progress_gil(&store_progress_cb, "write-done", 1, 1);

                Ok(StoreOutcome {
                    chunks_written,
                    embeddings_written,
                    // Filled in by the caller after all three threads join —
                    // the store thread has no access to the parse/embed
                    // threads' shared error accumulators.
                    errors: Vec::new(),
                    disk_limit_exceeded: disk_limit_hit,
                })
            })
        };

        // ── Embed thread ───────────────────────────────────────
        // Owns one rayon pool for the life of the run (built once, not
        // rebuilt per batch), so worker OS threads — and the Python-side
        // per-thread embedding-provider/HTTP-client cache keyed on them
        // (see chunkhound/pipeline_bridge.py's `threading.local()` usage)
        // — are reused across every streamed batch instead of being torn
        // down and rebuilt on each one.
        let embed_handle: std::thread::JoinHandle<Result<u64, String>> = {
            let error = Arc::clone(&error);
            let embed_errors = Arc::clone(&embed_errors);
            std::thread::spawn(move || {
                let pool = Self::build_embed_pool(embed_thread_pool_size)?;
                let mut pending_delete_paths = Some(delete_paths);
                let mut received_files: u64 = 0;
                let mut seen_chunks: u64 = 0;
                let mut embedded_chunks: u64 = 0;

                // Anchor the embed-rate clock before any batch is processed
                // — the Python side starts its speed-tracking timer on the
                // first "embed" call it receives, so an early (0, 0) call
                // here (mirroring the "parse" phase's own early call in
                // `run()`) ensures that timer starts before batch 0's
                // embedding work, not after it. Without this, the first
                // real per-batch call already carries batch 0's full chunk
                // count with ~0 elapsed time behind it, producing a
                // nonsensical "chunks/s" figure.
                if !skip_embeddings {
                    emit_progress_gil(&progress_cb_embed, "embed", 0, 0);
                }

                // Pipeline diagnostic: cumulative time the embed thread spends
                // BLOCKED on its channels — recv (upstream parse empty) and send
                // (downstream store full). Together with the store thread's
                // recv-wait this reveals which stage actually gates the pipeline.
                let mut embed_wait_parse = 0f64;
                let mut embed_wait_store = 0f64;
                loop {
                    let t_recv = Instant::now();
                    let recv = parse_rx.recv();
                    embed_wait_parse += t_recv.elapsed().as_secs_f64();
                    let (batch_idx, mut parsed_files) = match recv {
                        Ok(v) => v,
                        Err(_) => break,
                    };
                    if error.lock().unwrap_or_else(|e| e.into_inner()).is_some() {
                        break;
                    }

                    let t_batch = Instant::now();
                    received_files += parsed_files.len() as u64;

                    if !skip_embeddings && !parsed_files.is_empty() {
                        // Embed this batch's chunks.
                        let mut embed_targets: Vec<(usize, usize, String)> = Vec::new();
                        for (fi, pf) in parsed_files.iter().enumerate() {
                            if pf.error.is_some() {
                                continue;
                            }
                            for (ci, ch) in pf.chunks.iter().enumerate() {
                                let text = ch.embed_text.as_deref().unwrap_or(&ch.code).to_string();
                                embed_targets.push((fi, ci, text));
                            }
                        }
                        seen_chunks += embed_targets.len() as u64;

                        if !embed_targets.is_empty() {
                            if let Some(ref batch_cb) = embed_cb {
                                // `None` progress arg is deliberate:
                                // embed_batch_parallel's internal progress
                                // callback reports a total scoped to just
                                // this batch's chunk count, which isn't
                                // useful for a cross-batch running estimate
                                // — progress is instead computed and
                                // emitted once per batch below, using
                                // running totals.
                                match Self::embed_batch_parallel(
                                    &pool,
                                    embed_batch_size,
                                    &mut parsed_files,
                                    &embed_targets,
                                    batch_cb,
                                    &provider,
                                    &model,
                                    None,
                                ) {
                                    Ok(outcome) => {
                                        if !outcome.errors.is_empty() {
                                            embed_errors
                                                .lock()
                                                .unwrap_or_else(|e| e.into_inner())
                                                .extend(outcome.errors);
                                        }
                                        embedded_chunks += outcome.embedded;
                                    }
                                    Err(e) => {
                                        let mut err =
                                            error.lock().unwrap_or_else(|e| e.into_inner());
                                        if err.is_none() {
                                            *err = Some(e);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    if !skip_embeddings {
                        // Exact chunk total isn't known upfront in a
                        // pipelined/streaming design — estimate it from the
                        // chunk density observed so far, refined every batch.
                        // Converges to the exact count as received_files
                        // approaches total_files (exact on the final batch).
                        let estimated_total = if received_files > 0 {
                            ((seen_chunks as f64 / received_files as f64) * total_files as f64)
                                .round() as u64
                        } else {
                            seen_chunks
                        }
                        .max(embedded_chunks);
                        emit_progress_gil(
                            &progress_cb_embed,
                            "embed",
                            embedded_chunks,
                            estimated_total,
                        );
                    }

                    log::debug!(
                        "[embed] batch {batch_idx} done in {:.3}s",
                        t_batch.elapsed().as_secs_f64()
                    );

                    // Attach the original delete_paths (from the incremental
                    // diff) to the first streamed batch only — deleting is
                    // idempotent, but there is no need to repeat it per batch.
                    let batch_delete_paths = pending_delete_paths.take().unwrap_or_default();
                    let db_batch =
                        Self::build_db_batch(&parsed_files, batch_delete_paths, &existing_ids);
                    let t_send = Instant::now();
                    let sent = store_tx.send(db_batch);
                    embed_wait_store += t_send.elapsed().as_secs_f64();
                    if sent.is_err() {
                        // Store thread exited early (DB error) — stop feeding
                        // it; the real error surfaces via store_handle.join().
                        break;
                    }
                }
                log::debug!(
                    "[pipe] embed blocked: parse-recv {embed_wait_parse:.1}s  \
                     store-send {embed_wait_store:.1}s"
                );

                // No batch ever flowed through to carry `delete_paths` — this
                // happens when there are zero files to parse (e.g. re-indexing
                // a directory that's gone from "has files" to empty, or every
                // file is unchanged). Send a files-less batch so orphaned rows
                // still get deleted; `prepare_write()` handles `delete_paths`
                // independently of `batch.files` being non-empty.
                if let Some(paths) = pending_delete_paths.take() {
                    if !paths.is_empty() {
                        let db_batch = DbWriterBatch {
                            files: Vec::new(),
                            delete_paths: paths,
                        };
                        let _ = store_tx.send(db_batch);
                    }
                }

                Ok(embedded_chunks)
            })
        };

        // Join every stage before inspecting results, so the store thread's
        // DB connection is always closed (it closes the backend in both its
        // success and error paths) regardless of where an upstream error
        // occurred — never leak the connection.
        let parse_join = parse_handle.join();
        let embed_join = embed_handle.join();
        let store_join = store_handle.join();

        // A parse-thread error takes priority — it explains the incomplete
        // run even though later stages may have run to partial completion
        // (or hit their own errors) on whatever batches were already sent.
        // A panic (as opposed to a returned Err stashed in `error`) never
        // sets `error`, so it must be checked explicitly here — otherwise
        // the run falls through and reports success on an incomplete parse.
        if parse_join.is_err() {
            return Err("pipeline parse thread panicked".to_string());
        }
        if let Some(e) = Arc::try_unwrap(error)
            .expect("parse/embed threads already joined above and drop their clone on exit, so this is the sole remaining owner")
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            return Err(e);
        }
        // Per-file parse errors don't abort the run (unlike `error` above) —
        // collected here so they can be merged into the final report below.
        let file_parse_errors = Arc::try_unwrap(parse_errors)
            .expect("parse thread already joined above and drops its clone on exit, so this is the sole remaining owner")
            .into_inner()
            .unwrap_or_else(|e| e.into_inner());

        let embedded_chunks = match embed_join {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("pipeline embed thread panicked".to_string()),
        };
        // Per-batch embed errors don't abort the run (unlike `error` above) —
        // collected here so they can be merged into the final report below,
        // instead of only reaching a log::warn! line as before.
        let file_embed_errors = Arc::try_unwrap(embed_errors)
            .expect("embed thread already joined above and drops its clone on exit, so this is the sole remaining owner")
            .into_inner()
            .unwrap_or_else(|e| e.into_inner());

        // Emit final parse/embed progress — both stages are fully done by
        // the time every batch has been received and embedded here, even
        // though the store thread may still be writing the last batch(es).
        // These land the bars on their exact final counts (the per-batch
        // embed estimate above should already match, but this is a cheap,
        // guaranteed-exact landing rather than relying on that convergence).
        emit_progress_gil(&progress_cb, "parse", total_files, total_files);
        if !skip_embeddings {
            emit_progress_gil(&progress_cb, "embed", embedded_chunks, embedded_chunks);
        }

        match store_join {
            Ok(Ok(mut result)) => {
                result.errors = file_parse_errors;
                result.errors.extend(file_embed_errors);
                Ok(result)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err("pipeline store thread panicked".to_string()),
        }
    }

    /// Run a fallible backend setup step (`open`/`drop_all_hnsw_indexes`),
    /// closing the backend on failure so this step doesn't skip Invariant 14
    /// — no failure path may leave the connection open or HNSW indexes
    /// un-restored — mirroring the pattern the write/compact steps below
    /// already use via `post_write_result`'s best-effort `backend.close()`.
    fn close_backend_on_err<T>(
        backend: &mut dyn DbBackend,
        result: Result<T, DbError>,
    ) -> Result<T, String> {
        result.map_err(|e| {
            let _ = backend.close();
            e.to_string()
        })
    }

    /// Convert parsed files into a `DbWriterBatch`. Each file's relative
    /// path (`pf.rel_path`) was already computed by Python's
    /// `get_relative_path_safe()` and carried through parsing on
    /// `ParsedFile` — this function no longer re-derives it, which used to
    /// diverge from Python's DB-write path for symlinked files (see
    /// `differ::compute_diff`'s doc comment for the full history). Shared by
    /// the single-shot write path and the 3-stage streaming path, which
    /// calls this once per batch.
    fn build_db_batch(
        parsed: &[super::types::ParsedFile],
        delete_paths: Vec<String>,
        existing_ids: &std::collections::HashMap<PathBuf, i64>,
    ) -> DbWriterBatch {
        let mut file_records = Vec::with_capacity(parsed.len());

        for pf in parsed {
            // Files with a parse error or with zero chunks and no detected
            // language (images, binaries, generated data — anything with
            // nothing to index) still get a row here, with empty `chunks`
            // and `pf.skip_reason` set. Without this, such a file has no DB
            // row at all, so every future run's diff phase (which only ever
            // asks "does a row with a matching mtime exist for this path?")
            // rediscovers it as new and reprocesses it forever.
            let chunks: Vec<ChunkRecord> = pf
                .chunks
                .iter()
                .map(|ch| ChunkRecord {
                    chunk_type: ch.chunk_type.clone(),
                    symbol: ch.symbol.clone(),
                    code: ch.code.clone(),
                    start_line: ch.start_line,
                    end_line: ch.end_line,
                    start_byte: ch.start_byte,
                    end_byte: ch.end_byte,
                    language: ch.language.clone(),
                    metadata: ch.metadata.clone(),
                    embedding: ch.embedding.clone(),
                    provider: ch.provider.clone(),
                    model: ch.model.clone(),
                })
                .collect();

            file_records.push(FileRecord {
                existing_file_id: existing_ids.get(&pf.path).copied(),
                path: pf.rel_path.clone(),
                mtime: Some(pf.mtime),
                size_bytes: Some(pf.file_size as i64),
                content_hash: if pf.content_hash.is_empty() {
                    None
                } else {
                    Some(pf.content_hash.clone())
                },
                language: pf.language.clone(),
                skip_reason: pf.skip_reason.clone(),
                chunks,
            });
        }

        DbWriterBatch {
            files: file_records,
            delete_paths,
        }
    }

    /// Call the Python batch callback and extract ParsedFile results.
    // `rel_keys` pushed this to 8 args — each is a distinct read-only lookup
    // map produced once by the diff phase (new_hashes/disk_stats/rel_keys)
    // or a borrowed callback/path slice; bundling them into a struct would
    // just move the same fields behind one more layer without reducing what
    // this function actually needs, matching the same tradeoff already made
    // for `pipeline_parse_embed_store` and `embed_batch_parallel` below.
    #[allow(clippy::too_many_arguments)]
    fn parse_one_batch(
        py: Python<'_>,
        cb: &Py<PyAny>,
        parse_config: &Py<super::parse_call_config::ParseCallConfig>,
        paths: &[String],
        batch: &[PathBuf],
        new_hashes: &std::collections::HashMap<PathBuf, String>,
        disk_stats: &std::collections::HashMap<PathBuf, (u64, f64)>,
        rel_keys: &std::collections::HashMap<PathBuf, String>,
    ) -> Result<Vec<super::types::ParsedFile>, String> {
        let cb = cb.bind(py);
        let py_paths = PyList::new_bound(py, paths);
        let ret = cb
            .call1((py_paths, parse_config.bind(py)))
            .map_err(|e| e.to_string())?;

        let tuple_list: &Bound<'_, PyList> = ret.downcast::<PyList>().map_err(|e| e.to_string())?;

        if tuple_list.len() != batch.len() {
            return Err(format!(
                "parse callback returned {} results for {} paths",
                tuple_list.len(),
                batch.len()
            ));
        }

        let mut parsed = Vec::with_capacity(batch.len());

        for (i, item) in tuple_list.iter().enumerate() {
            let path = &batch[i];

            let lang: String = item
                .get_item(0)
                .ok()
                .and_then(|v| v.extract::<String>().ok())
                .unwrap_or_default();

            // Optional 3rd tuple element: per-file error message from the
            // Python callback (e.g. `_parse_one_file` caught an exception
            // for this file). A `None` Python value, or a callback that only
            // returns 2-tuples, both fall through to `None` here — extracting
            // `String` from `None` fails, same lenient pattern as the other
            // per-item helpers below.
            let py_error: Option<String> = item
                .get_item(2)
                .ok()
                .and_then(|v| v.extract::<String>().ok());

            if let Some(err) = py_error {
                let (file_size, mtime) = Self::disk_stats_or_stat(path, disk_stats);
                parsed.push(super::types::ParsedFile {
                    path: path.clone(),
                    rel_path: rel_keys.get(path).cloned().unwrap_or_default(),
                    language: None,
                    file_size,
                    mtime,
                    content_hash: new_hashes.get(path).cloned().unwrap_or_default(),
                    chunks: Vec::new(),
                    skip_reason: Some(Self::truncate_skip_reason(&format!("parse_error: {err}"))),
                    error: Some(err),
                });
                continue;
            }

            let chunks_py = match item
                .get_item(1)
                .ok()
                .and_then(|v| v.downcast_into::<PyList>().ok())
            {
                Some(l) => l,
                None => {
                    let (file_size, mtime) = Self::disk_stats_or_stat(path, disk_stats);
                    parsed.push(super::types::ParsedFile {
                        path: path.clone(),
                        rel_path: rel_keys.get(path).cloned().unwrap_or_default(),
                        language: None,
                        file_size,
                        mtime,
                        content_hash: new_hashes.get(path).cloned().unwrap_or_default(),
                        chunks: Vec::new(),
                        skip_reason: Some("parse_error: invalid chunk list".into()),
                        error: Some("invalid chunk list".into()),
                    });
                    continue;
                }
            };

            let chunks = Self::extract_chunks(py, &chunks_py);

            // Reuse the size + mtime already read by the diff phase — avoids
            // a third stat() pass per changed file. Falls back to a fresh
            // metadata() call for non-incremental runs (where disk_stats is
            // empty) or any file that wasn't in the precomputed map.
            let (file_size, mtime) = Self::disk_stats_or_stat(path, disk_stats);
            let language = if lang.is_empty() { None } else { Some(lang) };
            let skip_reason = if chunks.is_empty() && language.is_none() {
                Some("unrecognized_or_empty".to_string())
            } else {
                None
            };

            parsed.push(super::types::ParsedFile {
                path: path.clone(),
                rel_path: rel_keys.get(path).cloned().unwrap_or_default(),
                language,
                file_size,
                mtime,
                // Computed by the diff phase (`differ::compute_diff`), which
                // already read this file's bytes once to decide it needed
                // reprocessing — reused here instead of hashing again.
                content_hash: new_hashes.get(path).cloned().unwrap_or_default(),
                chunks,
                skip_reason,
                error: None,
            });
        }

        Ok(parsed)
    }

    /// Resolve a file's (size, mtime) from the diff phase's precomputed map,
    /// falling back to a fresh `stat()` when absent (non-incremental runs,
    /// where the map is empty, or any file the diff phase didn't cover).
    /// Shared by every `ParsedFile` construction site in `parse_one_batch` —
    /// including the error branches, which must populate real values here
    /// too: a `FileRecord` written with a zeroed mtime would never match the
    /// file's real on-disk mtime on a later run, defeating the whole point
    /// of persisting a skip row (see `build_db_batch`).
    fn disk_stats_or_stat(
        path: &std::path::Path,
        disk_stats: &std::collections::HashMap<PathBuf, (u64, f64)>,
    ) -> (u64, f64) {
        disk_stats.get(path).copied().unwrap_or_else(|| {
            let meta = std::fs::metadata(path).ok();
            let mtime = meta
                .as_ref()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            (meta.map_or(0, |m| m.len()), mtime)
        })
    }

    /// Bound a skip_reason string's length — parse error messages can
    /// theoretically echo file content or a long traceback.
    fn truncate_skip_reason(reason: &str) -> String {
        const MAX_LEN: usize = 500;
        if reason.len() <= MAX_LEN {
            reason.to_string()
        } else {
            let mut truncated = reason.chars().take(MAX_LEN).collect::<String>();
            truncated.push_str("...");
            truncated
        }
    }

    /// Size and build the rayon thread pool used for parallel embed
    /// dispatch.
    ///
    /// Built once per pipeline run (by the embed thread) and reused across
    /// every streamed batch — avoids spinning up a fresh OS thread pool
    /// (and discarding the Python-side per-thread embedding-provider/
    /// HTTP-client cache) on every batch.
    fn build_embed_pool(embed_thread_pool_size: usize) -> Result<rayon::ThreadPool, String> {
        // Respect embed_thread_pool_size if set, else cap to half of CPUs
        // to avoid overwhelming the embedding API with concurrent requests.
        let embed_threads = if embed_thread_pool_size > 0 {
            embed_thread_pool_size
        } else {
            let cpu = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (cpu / 2).clamp(1, 4)
        };

        rayon::ThreadPoolBuilder::new()
            .num_threads(embed_threads)
            .build()
            .map_err(|e| e.to_string())
    }

    /// Dispatch embed batches via rayon to a Python batch callback.
    ///
    /// Each batch calls Python's ``embed_batch_callback(texts) → List[List[float]]``.
    /// Results are collected and applied to ``parsed`` after all batches complete.
    ///
    /// **Caller must release the GIL** before entering this method (via
    /// ``py.allow_threads()``).  Each rayon thread re-acquires the GIL
    /// independently via ``Python::with_gil()``.
    // `pool`/`embed_batch_size` were split out of `&self` so this fn can be
    // called from a thread that doesn't hold `&IndexingPipeline` — that's
    // the 8th argument; splitting further would obscure the parameter list.
    #[allow(clippy::too_many_arguments)]
    fn embed_batch_parallel(
        pool: &rayon::ThreadPool,
        embed_batch_size: usize,
        parsed: &mut [super::types::ParsedFile],
        targets: &[(usize, usize, String)],
        callback: &Py<PyAny>,
        provider: &str,
        model: &str,
        progress_callback: Option<Py<PyAny>>,
    ) -> Result<EmbedBatchOutcome, String> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Mutex;

        let batch_size = embed_batch_size.max(1);
        let total = targets.len() as u64;
        let completed = AtomicU64::new(0);

        let all_results: Mutex<Vec<(usize, usize, Vec<f32>)>> = Mutex::new(Vec::new());
        // Per-batch callback failures don't abort the run (like Python's
        // asyncio.gather(return_exceptions=True) path) — collected here
        // instead of only reaching a log::warn! line, so the chunks that
        // ended up written without an embedding are still visible to the
        // caller in `PipelineReport.errors`.
        let batch_errors: Mutex<Vec<String>> = Mutex::new(Vec::new());
        // Cloned once upfront (read-only, cheap — paths only) so the failure
        // path below can name the affected file without holding a borrow of
        // `parsed` across the parallel section (`parsed` is mutated after).
        let file_paths: Vec<PathBuf> = parsed.iter().map(|pf| pf.path.clone()).collect();

        pool.install(|| {
            targets.par_chunks(batch_size).for_each(|batch| {
                let texts: Vec<String> = batch.iter().map(|(_, _, t)| t.clone()).collect();
                let indices: Vec<(usize, usize)> =
                    batch.iter().map(|(fi, ci, _)| (*fi, *ci)).collect();
                let batch_len = batch.len() as u64;

                match Python::with_gil(|gil_py| {
                    let cb = callback.bind(gil_py);
                    let ret = cb.call1((texts,))?;
                    let vectors: Vec<Vec<f64>> = ret.extract()?;
                    Ok::<_, pyo3::PyErr>(vectors)
                }) {
                    Ok(vectors) => {
                        let mut batch_results = Vec::with_capacity(batch.len());
                        // Indices for which the callback returned no vector at
                        // all (a well-formed but short response — e.g. a
                        // provider that silently truncates — rather than a
                        // raised exception, which is handled below instead).
                        let mut missing: Vec<(usize, usize)> = Vec::new();
                        for (i, (fi, ci)) in indices.iter().enumerate() {
                            if let Some(vec) = vectors.get(i) {
                                batch_results.push((
                                    *fi,
                                    *ci,
                                    vec.iter().map(|x| *x as f32).collect(),
                                ));
                            } else {
                                missing.push((*fi, *ci));
                            }
                        }
                        all_results
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .extend(batch_results);
                        if !missing.is_empty() {
                            log::warn!(
                                "embed batch returned {} vector(s) for {} chunk(s), {} missing",
                                vectors.len(),
                                batch_len,
                                missing.len()
                            );
                            let reason = format!(
                                "provider returned {} vector(s) for {} requested",
                                vectors.len(),
                                batch_len
                            );
                            batch_errors
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .extend(Self::format_batch_errors(&missing, &file_paths, &reason));
                        }
                    }
                    Err(e) => {
                        log::warn!("embed batch failed ({} chunks), continuing: {e}", batch_len);
                        batch_errors
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .extend(Self::format_batch_errors(
                                &indices,
                                &file_paths,
                                &e.to_string(),
                            ));
                    }
                };

                // Always advance progress — even on failure, like Python's
                // asyncio.gather(return_exceptions=True) path does.
                let done = completed.fetch_add(batch_len, Ordering::Relaxed) + batch_len;
                if let Some(ref prog) = progress_callback {
                    Python::with_gil(|gil_py| {
                        let cb = prog.bind(gil_py);
                        let _ = cb.call1(("embed", done, total));
                    });
                }
            });
        });

        // Apply embeddings to parsed chunks (single-threaded, after all batches).
        // Mutating pure-Rust Vec<f32> fields — no GIL required.
        let results = all_results.into_inner().unwrap_or_else(|e| e.into_inner());
        let embedded = results.len() as u64;
        for (fi, ci, vec) in results {
            let chunk = &mut parsed[fi].chunks[ci];
            chunk.embedding = Some(vec);
            chunk.provider = Some(provider.to_string());
            chunk.model = Some(model.to_string());
        }

        Ok(EmbedBatchOutcome {
            embedded,
            errors: batch_errors.into_inner().unwrap_or_else(|e| e.into_inner()),
        })
    }

    /// Group `(file_index, chunk_index)` pairs by file and format one
    /// `"{path}: embedding failed for {n} chunk(s): {reason}"` line per
    /// affected file — shared by `embed_batch_parallel`'s two failure paths
    /// (a raised exception, and a well-formed-but-short response) so both
    /// produce errors at the same per-file granularity as `parse_errors`.
    fn format_batch_errors(
        indices: &[(usize, usize)],
        file_paths: &[PathBuf],
        reason: &str,
    ) -> Vec<String> {
        let mut per_file: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (fi, _ci) in indices {
            *per_file.entry(*fi).or_insert(0) += 1;
        }
        per_file
            .into_iter()
            .map(|(fi, count)| {
                let path = file_paths
                    .get(fi)
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|| "<unknown>".to_string());
                format!("{path}: embedding failed for {count} chunk(s): {reason}")
            })
            .collect()
    }

    /// Extract NewChunk structs from a Python list[dict].
    fn extract_chunks(
        py: Python<'_>,
        chunks_py: &Bound<'_, PyList>,
    ) -> Vec<super::types::NewChunk> {
        let mut chunks = Vec::with_capacity(chunks_py.len());

        for item in chunks_py.iter() {
            let cd: &Bound<'_, PyDict> = match item.downcast::<PyDict>() {
                Ok(d) => d,
                Err(_) => continue,
            };

            chunks.push(super::types::NewChunk {
                chunk_type: Self::str_from_dict(py, cd, "chunk_type"),
                symbol: Self::opt_str_from_dict(py, cd, "symbol"),
                code: Self::str_from_dict(py, cd, "code"),
                start_line: Self::opt_i64_from_dict(py, cd, "start_line"),
                end_line: Self::opt_i64_from_dict(py, cd, "end_line"),
                start_byte: Self::opt_i64_from_dict(py, cd, "start_byte"),
                end_byte: Self::opt_i64_from_dict(py, cd, "end_byte"),
                language: Self::opt_str_from_dict(py, cd, "language"),
                metadata: Self::opt_str_from_dict(py, cd, "metadata"),
                embed_text: Self::opt_str_from_dict(py, cd, "embed_text"),
                embedding: None,
                provider: None,
                model: None,
            });
        }

        chunks
    }

    fn str_from_dict(_py: Python<'_>, dict: &Bound<'_, PyDict>, key: &str) -> String {
        dict.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default()
    }

    fn opt_str_from_dict(_py: Python<'_>, dict: &Bound<'_, PyDict>, key: &str) -> Option<String> {
        dict.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
    }

    fn opt_i64_from_dict(_py: Python<'_>, dict: &Bound<'_, PyDict>, key: &str) -> Option<i64> {
        dict.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<i64>().ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BatchResult;
    use std::cell::Cell;

    /// Minimal `DbBackend` double for testing `close_backend_on_err` in
    /// isolation — no PyO3/GIL/pipeline machinery needed, since the function
    /// under test only touches `&mut dyn DbBackend` and a `Result`.
    struct FakeBackend {
        fail_open: bool,
        fail_drop_hnsw: bool,
        close_called: Cell<bool>,
    }

    impl DbBackend for FakeBackend {
        fn open(&mut self) -> Result<(), DbError> {
            if self.fail_open {
                Err(DbError::Other("simulated open failure".into()))
            } else {
                Ok(())
            }
        }

        fn close(&mut self) -> Result<(), DbError> {
            self.close_called.set(true);
            Ok(())
        }

        fn write_batch(&mut self, _batch: &DbWriterBatch) -> Result<BatchResult, DbError> {
            unreachable!("not exercised by these tests")
        }

        fn needs_compaction(&self) -> Result<bool, DbError> {
            Ok(false)
        }

        fn run_compaction(&mut self) -> Result<(), DbError> {
            Ok(())
        }

        fn drop_all_hnsw_indexes(&mut self) -> Result<(), DbError> {
            if self.fail_drop_hnsw {
                Err(DbError::Other(
                    "simulated drop_all_hnsw_indexes failure".into(),
                ))
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn test_open_failure_still_closes_backend() {
        let mut backend = FakeBackend {
            fail_open: true,
            fail_drop_hnsw: false,
            close_called: Cell::new(false),
        };
        let open_result = backend.open();
        let result = IndexingPipeline::close_backend_on_err(&mut backend, open_result);
        assert!(result.is_err());
        assert!(
            backend.close_called.get(),
            "close() must run even when open() fails (Invariant 14)"
        );
    }

    #[test]
    fn test_drop_hnsw_failure_still_closes_backend() {
        let mut backend = FakeBackend {
            fail_open: false,
            fail_drop_hnsw: true,
            close_called: Cell::new(false),
        };
        let drop_result = backend.drop_all_hnsw_indexes();
        let result = IndexingPipeline::close_backend_on_err(&mut backend, drop_result);
        assert!(result.is_err());
        assert!(
            backend.close_called.get(),
            "close() must run even when drop_all_hnsw_indexes() fails (Invariant 14)"
        );
    }

    #[test]
    fn test_success_path_does_not_close_early() {
        let mut backend = FakeBackend {
            fail_open: false,
            fail_drop_hnsw: false,
            close_called: Cell::new(false),
        };
        let open_result = backend.open();
        let result = IndexingPipeline::close_backend_on_err(&mut backend, open_result);
        assert!(result.is_ok());
        assert!(
            !backend.close_called.get(),
            "close() must not run on the success path — the store thread closes \
             explicitly later, and closing here would wrongly trigger a premature \
             ensure_all_hnsw_indexes() before any batches are written"
        );
    }
}
