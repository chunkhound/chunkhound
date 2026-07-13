use std::sync::Mutex;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::db::{create_backend, DbBackend, DbConfig};
use crate::types::{ChunkRecord, DbWriterBatch, FileRecord};

struct WriterInner {
    backend: Box<dyn DbBackend>,
}

#[pyclass]
pub struct RustDbWriter {
    inner: Mutex<WriterInner>,
}

/// Extract an optional field from a PyDict — returns None if key is absent or value is Python None.
fn extract_opt<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        None => Ok(None),
        Some(v) if v.is_none() => Ok(None),
        Some(v) => Ok(Some(v.extract()?)),
    }
}

/// Extract a required field from a PyDict — errors if key is absent.
fn extract_req<'py, T: FromPyObject<'py>>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(key.to_string()))?
        .extract()
}

/// Extract a DbWriterBatch directly from a Python dict — no json.dumps / serde_json round-trip.
fn extract_batch(batch: &Bound<'_, PyAny>) -> PyResult<DbWriterBatch> {
    let batch_dict = batch.downcast::<PyDict>()?;

    let delete_paths: Vec<String> = batch_dict
        .get_item("delete_paths")?
        .ok_or_else(|| PyKeyError::new_err("delete_paths"))?
        .extract()?;

    let files_obj = batch_dict
        .get_item("files")?
        .ok_or_else(|| PyKeyError::new_err("files"))?;
    let files_list = files_obj.downcast::<PyList>()?;

    let mut files = Vec::with_capacity(files_list.len());
    for file_obj in files_list.iter() {
        let fd = file_obj.downcast::<PyDict>()?;

        let chunks_obj = fd
            .get_item("chunks")?
            .ok_or_else(|| PyKeyError::new_err("chunks"))?;
        let chunks_list = chunks_obj.downcast::<PyList>()?;

        let mut chunks = Vec::with_capacity(chunks_list.len());
        for chunk_obj in chunks_list.iter() {
            let cd = chunk_obj.downcast::<PyDict>()?;
            chunks.push(ChunkRecord {
                chunk_type: extract_req(cd, "chunk_type")?,
                symbol: extract_opt(cd, "symbol")?,
                code: extract_req(cd, "code")?,
                start_line: extract_opt(cd, "start_line")?,
                end_line: extract_opt(cd, "end_line")?,
                start_byte: extract_opt(cd, "start_byte")?,
                end_byte: extract_opt(cd, "end_byte")?,
                language: extract_opt(cd, "language")?,
                metadata: extract_opt(cd, "metadata")?,
                embedding: extract_opt(cd, "embedding")?,
                provider: extract_opt(cd, "provider")?,
                model: extract_opt(cd, "model")?,
            });
        }

        files.push(FileRecord {
            existing_file_id: extract_opt(fd, "existing_file_id")?,
            path: extract_req(fd, "path")?,
            mtime: extract_opt(fd, "mtime")?,
            size_bytes: extract_opt(fd, "size_bytes")?,
            content_hash: extract_opt(fd, "content_hash")?,
            language: extract_opt(fd, "language")?,
            chunks,
        });
    }

    Ok(DbWriterBatch {
        files,
        delete_paths,
    })
}

#[pymethods]
impl RustDbWriter {
    #[new]
    fn new(db_config: &Bound<'_, PyDict>) -> PyResult<Self> {
        let db_path: String = extract_req(db_config, "db_path")?;
        let compaction_batch_threshold: u32 = extract_opt::<u64>(db_config, "compaction_batch_threshold")?
            .map(|v| v as u32)
            .unwrap_or(50);
        let config = DbConfig { db_path, compaction_batch_threshold };
        let backend = create_backend(config);
        Ok(RustDbWriter {
            inner: Mutex::new(WriterInner { backend }),
        })
    }

    fn open(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.open().map_err(PyErr::from)
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.close().map_err(PyErr::from)
        })
    }

    fn write_batch(&self, py: Python<'_>, batch: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Extract batch fields directly from Python objects while holding the GIL.
        // This replaces json.dumps(batch) → String → serde_json::from_str, saving
        // ~4MB of JSON serialization per batch at 20 files × 17 chunks × 1536-dim embeddings.
        let batch = extract_batch(batch)?;

        // Release GIL for the heavy DB work.
        let result = py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.write_batch(&batch)
        });

        let batch_result = result.map_err(PyErr::from)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("file_ids", batch_result.file_ids)?;
        dict.set_item("chunks_written", batch_result.chunks_written)?;
        dict.set_item("embeddings_written", batch_result.embeddings_written)?;
        Ok(dict.into())
    }

    fn needs_compaction(&self, py: Python<'_>) -> PyResult<bool> {
        py.allow_threads(|| {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.needs_compaction().map_err(PyErr::from)
        })
    }

    fn run_compaction(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.run_compaction()
        })
        .map_err(PyErr::from)
    }

    fn drop_all_hnsw_indexes(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.drop_all_hnsw_indexes().map_err(PyErr::from)
        })
    }

    fn ensure_all_hnsw_indexes(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.backend.ensure_all_hnsw_indexes()
        })
        .map_err(PyErr::from)
    }
}
