use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::db::{create_backend, DbBackend, DbConfig};
use crate::error::DbError;
use crate::types::DbWriterBatch;

struct WriterInner {
    backend: Box<dyn DbBackend>,
}

// Safety: WriterInner is accessed only under the Mutex, and DuckDbHnswBackend is Send.
unsafe impl Send for WriterInner {}

#[pyclass]
pub struct RustDbWriter {
    inner: Mutex<WriterInner>,
}

#[pymethods]
impl RustDbWriter {
    #[new]
    fn new(db_config: &Bound<'_, PyDict>) -> PyResult<Self> {
        // Convert PyDict → JSON → DbConfig
        let json_module = db_config.py().import_bound("json")?;
        let json_str: String = json_module
            .call_method1("dumps", (db_config,))?
            .extract()?;
        let json_val: serde_json::Value =
            serde_json::from_str(&json_str).map_err(|e| DbError::Json(e))?;
        let config = DbConfig::from_json_value(&json_val).map_err(PyErr::from)?;
        let backend = create_backend(config);
        Ok(RustDbWriter {
            inner: Mutex::new(WriterInner { backend }),
        })
    }

    fn open(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.backend.open().map_err(PyErr::from)
    }

    fn close(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.backend.close().map_err(PyErr::from)
    }

    fn write_batch(&self, py: Python<'_>, batch: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // Serialize batch to JSON inside the GIL
        let json_module = py.import_bound("json")?;
        let json_str: String = json_module.call_method1("dumps", (batch,))?.extract()?;

        // Release GIL for the heavy DB work
        let result = py.allow_threads(|| {
            let batch: DbWriterBatch = serde_json::from_str(&json_str).map_err(DbError::Json)?;
            let mut inner = self.inner.lock().unwrap();
            inner.backend.write_batch(&batch)
        });

        let batch_result = result.map_err(PyErr::from)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("file_ids", batch_result.file_ids)?;
        dict.set_item("chunks_written", batch_result.chunks_written)?;
        dict.set_item("embeddings_written", batch_result.embeddings_written)?;
        Ok(dict.into())
    }

    fn needs_compaction(&self) -> PyResult<bool> {
        let mut inner = self.inner.lock().unwrap();
        inner.backend.needs_compaction().map_err(PyErr::from)
    }

    fn run_compaction(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap();
            inner.backend.run_compaction()
        })
        .map_err(PyErr::from)
    }

    fn drop_all_hnsw_indexes(&self) -> PyResult<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.backend.drop_all_hnsw_indexes().map_err(PyErr::from)
    }

    fn ensure_all_hnsw_indexes(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            let mut inner = self.inner.lock().unwrap();
            inner.backend.ensure_all_hnsw_indexes()
        })
        .map_err(PyErr::from)
    }
}
