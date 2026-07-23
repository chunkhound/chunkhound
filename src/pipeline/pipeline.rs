//! Unified Rust indexing pipeline — main orchestration class.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;
use std::time::Instant;

use super::config::PipelineConfig;
use super::differ::{DbFileEntry, DiffResult};
use super::report::PipelineReport;

use crate::db::{create_backend, DbBackend, DbConfig};
use crate::types::{BatchResult, ChunkRecord, DbWriterBatch, FileRecord};

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
    /// Pipeline order: parse → embed → write (single transaction).
    ///
    /// progress_callback receives ``(phase: str, current: int, total: int)``
    /// at phase transitions.  Phases: ``"parse"``, ``"embed"``,
    /// ``"write-prepare"``, ``"write-data"``, ``"write-index"``,
    /// ``"write-compact"``, ``"write-done"``, ``"done"``.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (files, parse_callback, embed_callback=None, progress_callback=None, incremental=false, parse_batch_callback=None, embed_batch_callback=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        files: Vec<String>,
        parse_callback: Py<PyAny>,
        embed_callback: Option<Py<PyAny>>,
        progress_callback: Option<Py<PyAny>>,
        incremental: bool,
        parse_batch_callback: Option<Py<PyAny>>,
        embed_batch_callback: Option<Py<PyAny>>,
    ) -> PyResult<PipelineReport> {
        let started = Instant::now();

        if files.is_empty() {
            return Ok(PipelineReport::empty());
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

        // ── Parse + Embed (with optional pipeline parallelism) ──
        let provider = self.config.embedding_provider.clone();
        let model = self.config.embedding_model.clone();
        let total_files = batch_paths.len() as u64;

        emit_progress(py, &progress_callback, "parse", 0, total_files);

        let mut total_embeds: u64 = 0;
        let parsed: Vec<super::types::ParsedFile> = if self.config.pipeline_parallel
            && self.config.parse_thread_pool_size > 1
        {
            if let Some(parse_cb) = &parse_batch_callback {
                // Pipeline-parallel: parse and embed overlap via channels.
                // Clone Python references before releasing the GIL.
                let parse_cb = parse_cb.clone_ref(py);
                let embed_cb = embed_batch_callback.as_ref().map(|cb| cb.clone_ref(py));
                let seq_embed_cb = embed_callback.as_ref().map(|cb| cb.clone_ref(py));
                let progress_cb = progress_callback.as_ref().map(|cb| cb.clone_ref(py));
                let t_parse_embed = Instant::now();
                let result = py
                    .allow_threads(|| {
                        self.pipeline_parse_and_embed(
                            &batch_paths,
                            parse_cb,
                            embed_cb,
                            seq_embed_cb,
                            &provider,
                            &model,
                            progress_cb,
                        )
                    })
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                log::info!(
                    "pipeline parse+embed: {:.2}s",
                    t_parse_embed.elapsed().as_secs_f64()
                );
                result
            } else {
                // Should not reach here — pipeline_parallel requires batch callback.
                self.parse_batch(py, &batch_paths, &parse_callback)?
            }
        } else {
            // ── Sequential parse ──────────────────────────
            let t_parse = Instant::now();
            let mut parsed = if self.config.parse_thread_pool_size > 1 {
                if let Some(ref batch_cb) = parse_batch_callback {
                    let cb = batch_cb.clone_ref(py);
                    let progress_cb = progress_callback.as_ref().map(|cb| cb.clone_ref(py));
                    let parsed_result = py.allow_threads(|| {
                        self.parse_batch_parallel(&batch_paths, &cb, progress_cb, total_files)
                    });
                    parsed_result.map_err(pyo3::exceptions::PyRuntimeError::new_err)?
                } else {
                    let result = self.parse_batch(py, &batch_paths, &parse_callback)?;
                    let n = result.len() as u64;
                    emit_progress(py, &progress_callback, "parse", n, total_files);
                    result
                }
            } else {
                let result = self.parse_batch(py, &batch_paths, &parse_callback)?;
                let n = result.len() as u64;
                emit_progress(py, &progress_callback, "parse", n, total_files);
                result
            };
            log::info!("pipeline parse: {:.2}s", t_parse.elapsed().as_secs_f64());

            // ── Sequential embed ──────────────────────────
            let t_embed = Instant::now();
            if !self.config.skip_embeddings {
                // Collect all chunk texts with their positions for back-mapping.
                let mut embed_targets: Vec<(usize, usize, String)> = Vec::new();
                for (fi, pf) in parsed.iter().enumerate() {
                    if pf.error.is_some() {
                        continue;
                    }
                    for (ci, ch) in pf.chunks.iter().enumerate() {
                        let text = ch.embed_text.as_deref().unwrap_or(&ch.code).to_string();
                        embed_targets.push((fi, ci, text));
                    }
                }

                total_embeds = embed_targets.len() as u64;
                emit_progress(py, &progress_callback, "embed", 0, total_embeds);

                if !embed_targets.is_empty() {
                    // ── Parallel embed (embed_batch_callback + rayon) ──
                    if let Some(ref batch_cb) = embed_batch_callback {
                        if self.config.parse_thread_pool_size > 1 {
                            let cb = batch_cb.clone_ref(py);
                            let prog_ref = progress_callback.as_ref().map(|p| p.clone_ref(py));
                            py.allow_threads(|| {
                                self.embed_batch_parallel(
                                    &mut parsed,
                                    &embed_targets,
                                    &cb,
                                    &provider,
                                    &model,
                                    prog_ref,
                                )
                            })
                            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                        }
                    }

                    // ── Sequential embed (fallback) ───────
                    if embed_batch_callback.is_none() || self.config.parse_thread_pool_size <= 1 {
                        if let Some(ref embed_py) = embed_callback {
                            let batch_size = self.config.embed_batch_size.max(1);
                            let total_chunks = embed_targets.len() as u64;
                            let mut embedded = 0u64;

                            for batch in embed_targets.chunks(batch_size) {
                                let texts: Vec<String> =
                                    batch.iter().map(|(_, _, t)| t.clone()).collect();

                                let embed_cb = embed_py.bind(py);
                                let vectors: Vec<Vec<f64>> = embed_cb
                                    .call1((texts, provider.as_str(), model.as_str()))?
                                    .extract()?;

                                if vectors.is_empty() {
                                    continue;
                                }

                                for (i, (fi, ci, _)) in batch.iter().enumerate() {
                                    if let Some(vec) = vectors.get(i) {
                                        let chunk = &mut parsed[*fi].chunks[*ci];
                                        chunk.embedding =
                                            Some(vec.iter().map(|x| *x as f32).collect());
                                        chunk.provider = Some(provider.clone());
                                        chunk.model = Some(model.clone());
                                    }
                                }

                                embedded += batch.len() as u64;
                                emit_progress(
                                    py,
                                    &progress_callback,
                                    "embed",
                                    embedded,
                                    total_chunks,
                                );
                            }
                        }
                    }
                }
            }
            log::info!("pipeline embed: {:.2}s", t_embed.elapsed().as_secs_f64());
            parsed
        };

        emit_progress(py, &progress_callback, "embed", total_embeds, total_embeds);

        // Build DbWriterBatch from parsed results.
        let mut file_records = Vec::with_capacity(parsed.len());

        for pf in &parsed {
            if pf.error.is_some() {
                continue;
            }
            if pf.chunks.is_empty() && pf.language.is_none() {
                continue;
            }

            // Build FileRecord with its chunks

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
            let rel_path = if self.config.project_root.as_os_str().is_empty() {
                path_str
            } else if let Ok(rel) = pf.path.strip_prefix(&self.config.project_root) {
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

        let batch = DbWriterBatch {
            files: file_records,
            delete_paths,
        };

        // Resolve directory→db file path.
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

        emit_progress(py, &progress_callback, "write-prepare", 0, 1);
        let t_write = Instant::now();

        // Phase 1: Open + prepare (delete paths, ensure tables, drop HNSW).
        let batch_ref = &batch;
        let mut backend: Box<dyn DbBackend> = py
            .allow_threads(move || {
                let mut backend = create_backend(db_config);
                backend.open()?;
                backend.prepare_write(batch_ref)?;
                Ok::<_, crate::error::DbError>(backend)
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Phase 2: Write data (files, chunks, embeddings inside txn).
        emit_progress(py, &progress_callback, "write-data", 0, 1);
        let (mut backend, write_result) = py.allow_threads(move || {
            let result = backend.write_batch_incremental(batch_ref);
            (backend, result)
        });
        let result: BatchResult =
            write_result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Phase 3: Rebuild HNSW indexes (the expensive CPU-intensive step).
        // Called unconditionally — write_batch_incremental errors are propagated
        // above, and finish_write restores HNSW indexes (Invariant 14).
        emit_progress(py, &progress_callback, "write-index", 0, 1);
        let (backend, finish_result) = py.allow_threads(move || {
            let result = backend.finish_write();
            (backend, result)
        });
        finish_result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Phase 4: Compaction (if needed).
        let (mut backend, needs_result) = py.allow_threads(move || {
            let needs = backend.needs_compaction();
            (backend, needs)
        });
        let needs_compaction =
            needs_result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        if needs_compaction {
            emit_progress(py, &progress_callback, "write-compact", 0, 1);
            let (mut backend, compact_result) = py.allow_threads(move || {
                let result = backend.run_compaction();
                (backend, result)
            });
            compact_result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            backend
                .close()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        } else {
            backend
                .close()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        }

        emit_progress(py, &progress_callback, "write-done", 1, 1);
        log::info!("pipeline write: {:.2}s", t_write.elapsed().as_secs_f64());

        let chunks_written = result.chunks_written;
        let embeddings_generated = result.embeddings_written;
        let total_secs = started.elapsed().as_secs_f64();

        emit_progress(py, &progress_callback, "done", file_count, file_count);
        log::info!(
            "pipeline total: {total_secs:.2}s (files={file_count}, chunks={chunks_written}, embeds={embeddings_generated})"
        );

        Ok(PipelineReport {
            files_processed: file_count,
            files_skipped: 0,
            chunks_written,
            embeddings_generated,
            elapsed_secs: total_secs,
            errors: Vec::new(),
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

    // ── Pipeline-parallel parse+embed (Phase 8) ───────────────────

    /// Overlap parse and embed via a producer-consumer channel.
    ///
    /// A parse thread batches files, parses them via the batch callback,
    /// and sends results through a channel.  The calling thread receives
    /// parsed batches and embeds them while the parse thread works on the
    /// next batch.  Returns all ParsedFiles with embeddings populated.
    ///
    /// **Caller must release the GIL** before entering this method.
    /// Each thread re-acquires the GIL independently via ``Python::with_gil()``.
    #[allow(clippy::too_many_arguments)]
    fn pipeline_parse_and_embed(
        &self,
        files: &[PathBuf],
        parse_cb: Py<PyAny>,
        embed_cb: Option<Py<PyAny>>,
        seq_embed_cb: Option<Py<PyAny>>,
        provider: &str,
        model: &str,
        progress_cb: Option<Py<PyAny>>,
    ) -> Result<Vec<super::types::ParsedFile>, String> {
        use std::sync::mpsc;
        use std::sync::{Arc, Mutex};

        let batch_size = self.config.parse_batch_size.max(1);
        let detect_sql = self.config.detect_embedded_sql;
        let parse_thread_pool_size = self.config.parse_thread_pool_size;
        let embed_batch_size = self.config.embed_batch_size.max(1);
        let skip_embeddings = self.config.skip_embeddings;
        let provider = provider.to_string();
        let model = model.to_string();

        // Build file batches (indices into `files` slice).
        let batches: Vec<(usize, Vec<PathBuf>)> = files
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.to_vec()))
            .collect();
        let batch_count = batches.len();

        let (parse_tx, parse_rx) = mpsc::channel::<(usize, Vec<super::types::ParsedFile>)>();
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // ── Parse thread ──────────────────────────────────────
        let parse_handle = {
            let error = Arc::clone(&error);
            std::thread::spawn(move || {
                for (batch_idx, batch) in batches {
                    if error.lock().unwrap().is_some() {
                        break;
                    }

                    let paths: Vec<String> = batch
                        .iter()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();

                    let parsed = match Python::with_gil(|gil_py| {
                        // Use parallel parse dispatch within the batch.
                        if parse_thread_pool_size > 1 {
                            // Build a mini-pipeline config for this batch.
                            Self::parse_one_batch(gil_py, &parse_cb, &paths, &batch, detect_sql)
                        } else {
                            // Single-file fallback — not recommended for pipeline mode.
                            let cb = parse_cb.bind(gil_py);
                            let mut results = Vec::with_capacity(batch.len());
                            for p in batch.iter() {
                                let path_str = p.to_string_lossy();
                                let ret = cb.call1((path_str.as_ref(), detect_sql));
                                match ret {
                                    Ok(ret) => {
                                        let lang: String = ret
                                            .get_item(0)
                                            .ok()
                                            .and_then(|v| v.extract::<String>().ok())
                                            .unwrap_or_default();
                                        let chunks_py = ret
                                            .get_item(1)
                                            .ok()
                                            .and_then(|v| v.downcast_into::<PyList>().ok());
                                        let chunks = match chunks_py {
                                            Some(l) => Self::extract_chunks(gil_py, &l),
                                            None => vec![],
                                        };
                                        let meta =
                                            std::fs::metadata(p).map(|m| (m.len(), m.modified()));
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
                                        results.push(super::types::ParsedFile {
                                            path: p.clone(),
                                            language: if lang.is_empty() {
                                                None
                                            } else {
                                                Some(lang)
                                            },
                                            file_size,
                                            mtime,
                                            content_hash: String::new(),
                                            chunks,
                                            error: None,
                                        });
                                    }
                                    Err(e) => {
                                        results.push(super::types::ParsedFile {
                                            path: p.clone(),
                                            language: None,
                                            file_size: 0,
                                            mtime: 0.0,
                                            content_hash: String::new(),
                                            chunks: vec![],
                                            error: Some(e.to_string()),
                                        });
                                    }
                                }
                            }
                            Ok(results)
                        }
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

                    if parse_tx.send((batch_idx, parsed)).is_err() {
                        // Receiver dropped (main thread error) — stop.
                        break;
                    }
                }
            })
        };

        // ── Main thread: receive + embed ─────────────────────
        let mut all_parsed: Vec<super::types::ParsedFile> = Vec::new();
        let mut received = 0usize;

        while received < batch_count {
            match parse_rx.recv() {
                Ok((_batch_idx, mut parsed_files)) => {
                    received += 1;

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

                        if !embed_targets.is_empty() {
                            if let Some(ref batch_cb) = embed_cb {
                                // Parallel embed dispatch.
                                self.embed_batch_parallel(
                                    &mut parsed_files,
                                    &embed_targets,
                                    batch_cb,
                                    &provider,
                                    &model,
                                    None,
                                )?;
                            } else if let Some(ref seq_cb) = seq_embed_cb {
                                // Sequential fallback: call per-batch embed callback.
                                Python::with_gil(|gil_py| -> Result<(), String> {
                                    let cb = seq_cb.bind(gil_py);
                                    for chunk_batch in embed_targets.chunks(embed_batch_size) {
                                        let texts: Vec<String> =
                                            chunk_batch.iter().map(|(_, _, t)| t.clone()).collect();
                                        let vectors: Vec<Vec<f64>> = cb
                                            .call1((texts,))
                                            .map_err(|e| e.to_string())?
                                            .extract()
                                            .map_err(|e: pyo3::PyErr| e.to_string())?;
                                        for (i, (fi, ci, _)) in chunk_batch.iter().enumerate() {
                                            if let Some(vec) = vectors.get(i) {
                                                let chunk = &mut parsed_files[*fi].chunks[*ci];
                                                chunk.embedding =
                                                    Some(vec.iter().map(|x| *x as f32).collect());
                                                chunk.provider = Some(provider.clone());
                                                chunk.model = Some(model.clone());
                                            }
                                        }
                                    }
                                    Ok(())
                                })?;
                            }
                        }
                    }

                    all_parsed.extend(parsed_files);
                }
                Err(_) => {
                    // Channel closed — parse thread panicked or errored.
                    break;
                }
            }
        }

        // Wait for parse thread to exit.
        let _ = parse_handle.join();

        // Check for errors.
        if let Some(e) = Arc::try_unwrap(error).unwrap().into_inner().unwrap() {
            return Err(e);
        }

        // Emit final progress.
        let total = files.len() as u64;
        if let Some(ref cb) = progress_cb {
            let _ =
                Python::with_gil(|gil_py| cb.bind(gil_py).call1(("parse", total, total)).is_ok());
            let _ =
                Python::with_gil(|gil_py| cb.bind(gil_py).call1(("embed", total, total)).is_ok());
        }

        Ok(all_parsed)
    }

    /// Parse files using the batch callback with a rayon thread pool.
    ///
    /// Each batch calls Python's ``parse_batch_callback`` which uses
    /// ``ProcessPoolExecutor`` internally.  While a batch is waiting for
    /// subprocess results, CPython releases the GIL, allowing other rayon
    /// threads to dispatch their own batches — true CPU parallelism.
    ///
    /// **Caller must release the GIL** before entering this method (via
    /// ``py.allow_threads()``).  Each rayon thread re-acquires the GIL
    /// independently via ``Python::with_gil()``.
    fn parse_batch_parallel(
        &self,
        files: &[PathBuf],
        callback: &Py<PyAny>,
        progress_cb: Option<Py<PyAny>>,
        total_files: u64,
    ) -> Result<Vec<super::types::ParsedFile>, String> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        let batch_size = self.config.parse_batch_size.max(1);
        let detect_sql = self.config.detect_embedded_sql;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.parse_thread_pool_size.max(1))
            .build()
            .map_err(|e| e.to_string())?;

        let error: Mutex<Option<String>> = Mutex::new(None);

        let batch_results: Vec<Vec<super::types::ParsedFile>> = pool.install(|| {
            files
                .par_chunks(batch_size)
                .filter_map(|batch| {
                    if error.lock().unwrap().is_some() {
                        return None;
                    }

                    let paths: Vec<String> = batch
                        .iter()
                        .map(|p| p.to_string_lossy().into_owned())
                        .collect();

                    match Python::with_gil(|gil_py| {
                        Self::parse_one_batch(gil_py, callback, &paths, batch, detect_sql)
                    }) {
                        Ok(parsed) => Some(parsed),
                        Err(e) => {
                            let mut err = error.lock().unwrap();
                            if err.is_none() {
                                *err = Some(e);
                            }
                            None
                        }
                    }
                })
                .collect()
        });

        if let Some(e) = error.into_inner().unwrap() {
            return Err(e);
        }

        let result: Vec<super::types::ParsedFile> = batch_results.into_iter().flatten().collect();
        let parsed_count = result.len() as u64;
        if let Some(ref cb) = progress_cb {
            if Python::with_gil(|gil_py| {
                cb.bind(gil_py)
                    .call1(("parse", parsed_count, total_files))
                    .is_ok()
            }) {
                // progress emitted
            }
        }
        Ok(result)
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

    /// Dispatch embed batches via rayon to a Python batch callback.
    ///
    /// Each batch calls Python's ``embed_batch_callback(texts) → List[List[float]]``.
    /// Results are collected and applied to ``parsed`` after all batches complete.
    ///
    /// **Caller must release the GIL** before entering this method (via
    /// ``py.allow_threads()``).  Each rayon thread re-acquires the GIL
    /// independently via ``Python::with_gil()``.
    fn embed_batch_parallel(
        &self,
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

        let batch_size = self.config.embed_batch_size.max(1);
        let total = targets.len() as u64;
        let completed = AtomicU64::new(0);

        // Respect embed_thread_pool_size if set, else cap to half of CPUs
        // to avoid overwhelming the embedding API with concurrent requests.
        let embed_threads = if self.config.embed_thread_pool_size > 0 {
            self.config.embed_thread_pool_size
        } else {
            let cpu = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (cpu / 2).clamp(1, 4)
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(embed_threads)
            .build()
            .map_err(|e| e.to_string())?;

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

    /// Parse files using the single-file callback (serial).
    fn parse_batch(
        &self,
        py: Python<'_>,
        files: &[PathBuf],
        callback: &Py<PyAny>,
    ) -> PyResult<Vec<super::types::ParsedFile>> {
        let mut results = Vec::with_capacity(files.len());

        for path in files {
            let cb = callback.bind(py);
            let path_str = path.to_string_lossy();

            let result = cb.call1((path_str.as_ref(), self.config.detect_embedded_sql));

            match result {
                Ok(ret) => {
                    // Callback returns (language: str, chunks: list[dict])
                    let tuple: Bound<'_, PyAny> = ret;
                    let lang: String = tuple.get_item(0)?.extract::<String>().unwrap_or_default();
                    let chunks_py: Bound<'_, PyList> =
                        tuple.get_item(1)?.downcast_into::<PyList>().map_err(|_| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "parse callback must return (str, list[dict])",
                            )
                        })?;

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

                    results.push(super::types::ParsedFile {
                        path: path.clone(),
                        language: if lang.is_empty() { None } else { Some(lang) },
                        file_size,
                        mtime,
                        content_hash: String::new(),
                        chunks,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(super::types::ParsedFile {
                        path: path.clone(),
                        language: None,
                        file_size: 0,
                        mtime: 0.0,
                        content_hash: String::new(),
                        chunks: Vec::new(),
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(results)
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
