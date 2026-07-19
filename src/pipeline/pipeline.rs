//! Unified Rust indexing pipeline — main orchestration class.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::path::PathBuf;
use std::time::Instant;

use super::config::PipelineConfig;
use super::report::PipelineReport;

use crate::db::{create_backend, DbBackend, DbConfig};
use crate::types::{BatchResult, ChunkRecord, DbWriterBatch, FileRecord};

/// The main PyO3 class — Python calls `.run()` from `asyncio.to_thread`.
#[pyclass]
#[derive(Debug)]
pub(crate) struct IndexingPipeline {
    config: PipelineConfig,
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
    /// Phase 1 (current): single-threaded parse → store, no embeddings, no diffing.
    #[pyo3(signature = (files, parse_callback, embed_callback=None, progress_callback=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        files: Vec<String>,
        parse_callback: Py<PyAny>,
        embed_callback: Option<Py<PyAny>>,
        progress_callback: Option<Py<PyAny>>,
    ) -> PyResult<PipelineReport> {
        let _ = (&embed_callback, &progress_callback); // unused in Phase 1
        let started = Instant::now();

        if files.is_empty() {
            return Ok(PipelineReport::empty());
        }

        let file_count = files.len() as u64;
        let batch_paths: Vec<PathBuf> = files.into_iter().map(PathBuf::from).collect();

        // Phase 1: single-threaded parse.
        let parsed = self.parse_batch(py, &batch_paths, &parse_callback)?;

        // Build DbWriterBatch from parsed results.
        let mut file_records = Vec::with_capacity(parsed.len());
        let delete_paths: Vec<String> = Vec::new();

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
                    embedding: None,
                    provider: None,
                    model: None,
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
            || self.config.db_path == PathBuf::from(":memory:")
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
                        parent.display(), e
                    ))
                })?;
            }
        }

        let result: BatchResult = py.allow_threads(|| {
            let mut backend: Box<dyn DbBackend> = create_backend(db_config);
            backend.open()?;
            let res = backend.write_batch(&batch)?;
            backend.close()?;
            Ok::<_, crate::error::DbError>(res)
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let chunks_written = result.chunks_written;
        let mut embeddings_generated: u64 = result.embeddings_written;

        // Phase 2: Generate embeddings if callback provided and not skipped.
        if let Some(ref embed_py) = embed_callback {
            if !self.config.skip_embeddings && chunks_written > 0 {
                let db_path = self.config.db_path.join("chunks.db");
                let db_path_str = db_path.to_string_lossy().into_owned();

                // Read chunks from the DB (re-open connection).
                let db_cfg = DbConfig {
                    db_path: db_path_str.clone(),
                    compaction_batch_threshold: self.config.compaction_batch_threshold,
                    compaction_threshold: self.config.compaction_threshold,
                    compaction_min_size_bytes: self.config.compaction_min_size_mb * 1024
                        * 1024,
                };

                let chunks = py.allow_threads(|| {
                    let mut backend: Box<dyn DbBackend> = create_backend(db_cfg);
                    backend.open()?;
                    let chunks = backend.read_chunks()?;
                    backend.close()?;
                    Ok::<_, crate::error::DbError>(chunks)
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                if !chunks.is_empty() {
                    // Batch the chunks
                    let batch_size = self.config.embed_batch_size as usize;
                    for batch_chunks in chunks.chunks(batch_size) {
                        let texts: Vec<String> =
                            batch_chunks.iter().map(|c| c.code.clone()).collect();

                        // Call Python embed_callback
                        let embed_cb = embed_py.bind(py);
                        let vectors: Vec<Vec<f64>> = embed_cb
                            .call1((texts,))?
                            .extract()?;

                        if vectors.is_empty() {
                            continue;
                        }
                        let dims = vectors[0].len();

                        // Build Python list of dicts for the store callback
                        let py_list = PyList::empty_bound(py);
                        for (chunk, vec) in batch_chunks.iter().zip(vectors.iter()) {
                            let py_dict = PyDict::new_bound(py);
                            py_dict.set_item("chunk_id", chunk.id)?;
                            py_dict.set_item("provider", self.config.embedding_provider.as_str())?;
                            py_dict.set_item("model", self.config.embedding_model.as_str())?;
                            py_dict.set_item("embedding", vec.clone())?;
                            py_dict.set_item("dims", dims)?;
                            py_list.append(py_dict)?;
                        }

                        // Call Python store_embeddings_callback via the registry
                        let store_fn = py
                            .import_bound("chunkhound.pipeline_bridge")?
                            .getattr("store_embeddings_callback")?;
                        let stored: u64 = store_fn
                            .call1((db_path_str.as_str(), py_list))?
                            .extract()?;
                        embeddings_generated += stored;
                    }
                }
            }
        }

        Ok(PipelineReport {
            files_processed: file_count,
            files_skipped: 0,
            chunks_written,
            embeddings_generated,
            elapsed_secs: started.elapsed().as_secs_f64(),
            errors: Vec::new(),
            peak_rss_mb: None,
        })
    }
}

// ── Internal helpers ────────────────────────────────────────────

impl IndexingPipeline {
    /// Parse a batch of files using the Python callback.
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
                    let lang: String = tuple
                        .get_item(0)?
                        .extract::<String>()
                        .unwrap_or_default();
                    let chunks_py: Bound<'_, PyList> = tuple
                        .get_item(1)?
                        .downcast_into::<PyList>()
                        .map_err(|_| {
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
    fn extract_chunks(py: Python<'_>, chunks_py: &Bound<'_, PyList>) -> Vec<super::types::NewChunk> {
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