// PyO3 0.22's create_exception! emits cfg(feature = "gil-refs") in this crate,
// although that feature belongs to PyO3. Keep the workaround local to its macros.
#![allow(unexpected_cfgs)]

use pyo3::{create_exception, exceptions::PyRuntimeError, PyErr};

create_exception!(chunkhound_native, DatabaseBusyError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseInterruptedError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseLayoutError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseFilesystemError, PyRuntimeError);

#[derive(Debug, thiserror::Error)]
pub enum LeaseError {
    #[error("database is busy; close other ChunkHound users and retry")]
    Busy,
    #[error("database installation was interrupted; run chunkhound pull again; local work may need re-indexing")]
    Interrupted,
    #[error("unsupported database coordination layout: {0}")]
    Layout(&'static str),
    #[error("database coordination filesystem operation failed")]
    Filesystem(#[from] std::io::Error),
}

impl From<LeaseError> for PyErr {
    fn from(error: LeaseError) -> Self {
        let message = error.to_string();
        match error {
            LeaseError::Busy => DatabaseBusyError::new_err(message),
            LeaseError::Interrupted => DatabaseInterruptedError::new_err(message),
            LeaseError::Layout(_) => DatabaseLayoutError::new_err(message),
            LeaseError::Filesystem(_) => DatabaseFilesystemError::new_err(message),
        }
    }
}
