//! Unified Rust indexing pipeline — main orchestration class.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::config::PipelineConfig;
use super::differ::{DbFileEntry, DiffResult};
use super::report::PipelineReport;

use crate::db::{create_backend, DbBackend, DbConfig};
use crate::types::{ChunkRecord, DbWriterBatch, FileRecord};

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

/// Result of the store thread in the 3-stage streaming pipeline.
struct StoreOutcome {
    chunks_written: u64,
    embeddings_written: u64,
    /// Per-file parse errors collected from the parse thread — merged in
    /// after all three threads join (see `pipeline_parse_embed_store`).
    parse_errors: Vec<String>,
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
    /// at phase transitions.  Phases: ``"parse"``, ``"embed"``,
    /// ``"write-prepare"``, ``"write-data"``, ``"write-index"``,
    /// ``"write-compact"``, ``"write-done"``, ``"done"``. Only one of
    /// ``"write-index"``/``"write-compact"`` fires per run — compaction
    /// rebuilds indexes as part of its own rewrite, so the two never both
    /// run.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (files, parse_batch_callback, embed_batch_callback=None, progress_callback=None, incremental=false))]
    fn run(
        &mut self,
        py: Python<'_>,
        files: Vec<String>,
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
        let mut batch_paths: Vec<PathBuf> = files.into_iter().map(PathBuf::from).collect();

        // ── Incremental diff (Phase 3) ─────────────────────────
        let delete_paths: Vec<String>;
        if incremental {
            let diff = self.compute_diff_blocking(&batch_paths)?;
            // Only process changed files
            batch_paths = diff.changed;
            delete_paths = diff.removed;
            // Update file_count to reflect what will actually be processed
            file_count = batch_paths.len() as u64;
        } else {
            delete_paths = Vec::new();
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
            compaction_batch_threshold: self.config.compaction_batch_threshold,
            compaction_threshold: self.config.compaction_threshold,
            compaction_min_size_bytes: self.config.compaction_min_size_mb * 1024 * 1024,
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
                    embed_cb,
                    &provider,
                    &model,
                    progress_cb,
                    store_progress_cb,
                    delete_paths,
                    db_config,
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
            files_skipped: 0,
            chunks_written: outcome.chunks_written,
            embeddings_generated: outcome.embeddings_written,
            elapsed_secs: total_secs,
            errors: outcome.parse_errors,
            peak_rss_mb: None,
        })
    }
}

// ── Internal helpers ────────────────────────────────────────────

impl IndexingPipeline {
    /// Read the DB state and compute which files changed.
    fn compute_diff_blocking(&self, files: &[PathBuf]) -> PyResult<DiffResult> {
        let db_file = if self.config.db_path.as_os_str().is_empty()
            || self.config.db_path.as_os_str() == ":memory:"
        {
            return Ok(DiffResult {
                changed: files.to_vec(),
                removed: Vec::new(),
                files_scanned: files.len(),
            });
        } else {
            self.config.db_path.join("chunks.db")
        };

        if !db_file.exists() {
            return Ok(DiffResult {
                changed: files.to_vec(),
                removed: Vec::new(),
                files_scanned: files.len(),
            });
        }

        let conn = duckdb::Connection::open(&db_file)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut stmt = conn
            .prepare("SELECT path, EXTRACT(EPOCH FROM modified_time) FROM files")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let db_entries: Vec<DbFileEntry> = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let mtime: f64 = row.get(1)?;
                Ok(DbFileEntry { path, mtime })
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .filter_map(|r| r.ok())
            .collect();

        // DuckDB stores to_timestamp(epoch) using local time, so EXTRACT(EPOCH)
        // returns a value shifted by the timezone offset.  Compute the median
        // offset (db_mtime - disk_mtime) and normalize.
        let disk_mtimes: Vec<Option<f64>> = files
            .iter()
            .map(|p| {
                std::fs::metadata(p)
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .map(|t| {
                        t.duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs_f64())
                            .unwrap_or(0.0)
                    })
            })
            .collect();

        let mut offsets: Vec<f64> = Vec::new();
        for e in &db_entries {
            // Find matching disk mtime by relative path
            for (fp, dm_opt) in files.iter().zip(disk_mtimes.iter()) {
                if let Some(dm) = dm_opt {
                    let rel = fp
                        .strip_prefix(&self.config.project_root)
                        .ok()
                        .map(|r| r.to_string_lossy().replace('\\', "/"));
                    if rel.as_deref() == Some(&e.path) {
                        offsets.push(e.mtime - dm);
                        break;
                    }
                }
            }
        }

        offsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let tz_offset = if offsets.len() >= 2 {
            offsets[offsets.len() / 2]
        } else {
            offsets.first().copied().unwrap_or(0.0)
        };

        let normalized: Vec<DbFileEntry> = db_entries
            .iter()
            .map(|e| DbFileEntry {
                path: e.path.clone(),
                mtime: e.mtime - tz_offset,
            })
            .collect();

        let result = super::differ::compute_diff(
            &normalized,
            files,
            &self.config.project_root,
            self.config.mtime_epsilon_seconds,
        );

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
    #[allow(clippy::too_many_arguments)]
    fn pipeline_parse_embed_store(
        &self,
        files: &[PathBuf],
        parse_cb: Py<PyAny>,
        embed_cb: Option<Py<PyAny>>,
        provider: &str,
        model: &str,
        progress_cb: Option<Py<PyAny>>,
        store_progress_cb: Option<Py<PyAny>>,
        delete_paths: Vec<String>,
        db_config: DbConfig,
    ) -> Result<StoreOutcome, String> {
        use std::sync::mpsc;
        use std::sync::{Arc, Mutex};

        let batch_size = self.config.parse_batch_size.max(1);
        let detect_sql = self.config.detect_embedded_sql;
        let embed_thread_pool_size = self.config.embed_thread_pool_size;
        let embed_batch_size = self.config.embed_batch_size.max(1);
        let skip_embeddings = self.config.skip_embeddings;
        let project_root = self.config.project_root.clone();
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

        // Bounded at every hop (capacity 2): backpressure keeps any one
        // stage from running arbitrarily far ahead of the next — parsed
        // batches hold source text, embedded batches additionally hold
        // float vectors, and both are memory-heavy.
        let (parse_tx, parse_rx) = mpsc::sync_channel::<(usize, Vec<super::types::ParsedFile>)>(2);
        let (store_tx, store_rx) = mpsc::sync_channel::<DbWriterBatch>(2);
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        // Per-file parse errors (e.g. one file's parse callback raised) —
        // these don't abort the run, unlike `error` above, which is for
        // whole-batch-callback failures.
        let parse_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        // ── Parse thread ──────────────────────────────────────
        let parse_handle = {
            let error = Arc::clone(&error);
            let parse_errors = Arc::clone(&parse_errors);
            std::thread::spawn(move || {
                let mut parsed_files_count: u64 = 0;
                for (batch_idx, batch) in batches {
                    if error.lock().unwrap().is_some() {
                        break;
                    }

                    let t_batch = Instant::now();
                    let paths: Vec<String> = batch
                        .iter()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();

                    let parsed = match Python::with_gil(|gil_py| {
                        Self::parse_one_batch(gil_py, &parse_cb, &paths, &batch, detect_sql)
                    }) {
                        Ok(parsed) => parsed,
                        Err(e) => {
                            let mut err = error.lock().unwrap();
                            if err.is_none() {
                                *err = Some(e);
                            }
                            break;
                        }
                    };

                    {
                        let mut errs = parse_errors.lock().unwrap();
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
                let mut backend: Box<dyn DbBackend> = create_backend(db_config);
                backend.open().map_err(|e| e.to_string())?;
                backend.drop_all_hnsw_indexes().map_err(|e| e.to_string())?;
                emit_progress_gil(&store_progress_cb, "write-prepare", 0, batch_count_u64);

                let mut chunks_written = 0u64;
                let mut embeddings_written = 0u64;
                let mut batch_no = 0usize;

                let write_result: Result<(), String> = (|| {
                    while let Ok(batch) = store_rx.recv() {
                        let t_batch = Instant::now();
                        backend.prepare_write(&batch).map_err(|e| e.to_string())?;
                        let result = backend
                            .write_batch_incremental(&batch)
                            .map_err(|e| e.to_string())?;
                        chunks_written += result.chunks_written;
                        embeddings_written += result.embeddings_written;
                        batch_no += 1;
                        emit_progress_gil(
                            &store_progress_cb,
                            "write-data",
                            batch_no as u64,
                            batch_count_u64,
                        );
                        log::debug!(
                            "[store] batch {batch_no} done in {:.3}s",
                            t_batch.elapsed().as_secs_f64()
                        );
                    }
                    Ok(())
                })();

                if let Err(e) = write_result {
                    // Best-effort: restore HNSW indexes over whatever was
                    // already committed before surfacing the error
                    // (mirrors the single-shot path's Invariant 14 restore).
                    let _ = backend.close();
                    return Err(e);
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
                let needs_compaction = backend.needs_compaction().map_err(|e| e.to_string())?;
                if needs_compaction {
                    emit_progress_gil(&store_progress_cb, "write-compact", 0, 1);
                    backend.run_compaction().map_err(|e| e.to_string())?;
                } else {
                    emit_progress_gil(&store_progress_cb, "write-index", 0, 1);
                    backend
                        .ensure_all_hnsw_indexes()
                        .map_err(|e| e.to_string())?;
                }

                backend.close().map_err(|e| e.to_string())?;
                emit_progress_gil(&store_progress_cb, "write-done", 1, 1);

                Ok(StoreOutcome {
                    chunks_written,
                    embeddings_written,
                    // Filled in by the caller after all three threads join —
                    // the store thread has no access to the parse thread's
                    // shared `parse_errors` accumulator.
                    parse_errors: Vec::new(),
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

                while let Ok((batch_idx, mut parsed_files)) = parse_rx.recv() {
                    if error.lock().unwrap().is_some() {
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
                                if let Err(e) = Self::embed_batch_parallel(
                                    &pool,
                                    embed_batch_size,
                                    &mut parsed_files,
                                    &embed_targets,
                                    batch_cb,
                                    &provider,
                                    &model,
                                    None,
                                ) {
                                    let mut err = error.lock().unwrap();
                                    if err.is_none() {
                                        *err = Some(e);
                                    }
                                    break;
                                }
                                embedded_chunks += embed_targets.len() as u64;
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
                        Self::build_db_batch(&parsed_files, &project_root, batch_delete_paths);
                    if store_tx.send(db_batch).is_err() {
                        // Store thread exited early (DB error) — stop feeding
                        // it; the real error surfaces via store_handle.join().
                        break;
                    }
                }

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
        let _ = parse_join;
        if let Some(e) = Arc::try_unwrap(error).unwrap().into_inner().unwrap() {
            return Err(e);
        }
        // Per-file parse errors don't abort the run (unlike `error` above) —
        // collected here so they can be merged into the final report below.
        let file_parse_errors = Arc::try_unwrap(parse_errors).unwrap().into_inner().unwrap();

        let embedded_chunks = match embed_join {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("pipeline embed thread panicked".to_string()),
        };

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
                result.parse_errors = file_parse_errors;
                Ok(result)
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err("pipeline store thread panicked".to_string()),
        }
    }

    /// Convert parsed files into a `DbWriterBatch`, resolving each file's
    /// relative path against `project_root` (mirrors Python's
    /// `_get_relative_path`). Shared by the single-shot write path and the
    /// 3-stage streaming path, which calls this once per batch.
    fn build_db_batch(
        parsed: &[super::types::ParsedFile],
        project_root: &Path,
        delete_paths: Vec<String>,
    ) -> DbWriterBatch {
        let mut file_records = Vec::with_capacity(parsed.len());

        for pf in parsed {
            if pf.error.is_some() {
                continue;
            }
            if pf.chunks.is_empty() && pf.language.is_none() {
                continue;
            }

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

            let path_str = pf.path.to_string_lossy().into_owned();

            // Store relative path (like Python _get_relative_path).
            let rel_path = if project_root.as_os_str().is_empty() {
                path_str
            } else if let Ok(rel) = pf.path.strip_prefix(project_root) {
                rel.to_string_lossy().into_owned()
            } else {
                pf.path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path_str.clone())
            };

            file_records.push(FileRecord {
                existing_file_id: None,
                path: rel_path,
                mtime: Some(pf.mtime),
                size_bytes: Some(pf.file_size as i64),
                content_hash: if pf.content_hash.is_empty() {
                    None
                } else {
                    Some(pf.content_hash.clone())
                },
                language: pf.language.clone(),
                chunks,
            });
        }

        DbWriterBatch {
            files: file_records,
            delete_paths,
        }
    }

    /// Call the Python batch callback and extract ParsedFile results.
    fn parse_one_batch(
        py: Python<'_>,
        cb: &Py<PyAny>,
        paths: &[String],
        batch: &[PathBuf],
        detect_embedded_sql: bool,
    ) -> Result<Vec<super::types::ParsedFile>, String> {
        let cb = cb.bind(py);
        let py_paths = PyList::new_bound(py, paths);
        let ret = cb
            .call1((py_paths, detect_embedded_sql))
            .map_err(|e| e.to_string())?;

        let tuple_list: &Bound<'_, PyList> = ret.downcast::<PyList>().map_err(|e| e.to_string())?;

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
                parsed.push(super::types::ParsedFile {
                    path: path.clone(),
                    language: None,
                    file_size: 0,
                    mtime: 0.0,
                    content_hash: String::new(),
                    chunks: Vec::new(),
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
                    parsed.push(super::types::ParsedFile {
                        path: path.clone(),
                        language: None,
                        file_size: 0,
                        mtime: 0.0,
                        content_hash: String::new(),
                        chunks: Vec::new(),
                        error: Some("invalid chunk list".into()),
                    });
                    continue;
                }
            };

            let chunks = Self::extract_chunks(py, &chunks_py);
            let meta = std::fs::metadata(path).map(|m| (m.len(), m.modified()));

            let (file_size, mtime) = match meta {
                Ok((s, mt)) => {
                    let mtime_secs = mt
                        .unwrap_or(std::time::UNIX_EPOCH)
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0);
                    (s, mtime_secs)
                }
                Err(_) => (0, 0.0),
            };

            parsed.push(super::types::ParsedFile {
                path: path.clone(),
                language: if lang.is_empty() { None } else { Some(lang) },
                file_size,
                mtime,
                content_hash: String::new(),
                chunks,
                error: None,
            });
        }

        Ok(parsed)
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
    ) -> Result<(), String> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Mutex;

        let batch_size = embed_batch_size.max(1);
        let total = targets.len() as u64;
        let completed = AtomicU64::new(0);

        let all_results: Mutex<Vec<(usize, usize, Vec<f32>)>> = Mutex::new(Vec::new());

        pool.install(|| {
            targets.par_chunks(batch_size).for_each(|batch| {
                let texts: Vec<String> = batch.iter().map(|(_, _, t)| t.clone()).collect();
                let indices: Vec<(usize, usize)> =
                    batch.iter().map(|(fi, ci, _)| (*fi, *ci)).collect();
                let batch_len = batch.len() as u64;

                let _success = match Python::with_gil(|gil_py| {
                    let cb = callback.bind(gil_py);
                    let ret = cb.call1((texts,))?;
                    let vectors: Vec<Vec<f64>> = ret.extract()?;
                    Ok::<_, pyo3::PyErr>(vectors)
                }) {
                    Ok(vectors) => {
                        let mut batch_results = Vec::with_capacity(batch.len());
                        for (i, (fi, ci)) in indices.iter().enumerate() {
                            if let Some(vec) = vectors.get(i) {
                                batch_results.push((
                                    *fi,
                                    *ci,
                                    vec.iter().map(|x| *x as f32).collect(),
                                ));
                            }
                        }
                        all_results.lock().unwrap().extend(batch_results);
                        true
                    }
                    Err(e) => {
                        log::warn!("embed batch failed ({} chunks), continuing: {e}", batch_len);
                        false
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
        for (fi, ci, vec) in all_results.into_inner().unwrap() {
            let chunk = &mut parsed[fi].chunks[ci];
            chunk.embedding = Some(vec);
            chunk.provider = Some(provider.to_string());
            chunk.model = Some(model.to_string());
        }

        Ok(())
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
