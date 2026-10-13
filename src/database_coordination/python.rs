use std::path::PathBuf;

use pyo3::prelude::*;

use super::error::*;
use super::lease::{self, Backend, Lease};

#[pyclass(module = "chunkhound_native")]
pub(crate) struct DatabaseLease {
    held: Option<Lease>,
}

#[pymethods]
impl DatabaseLease {
    fn close(&mut self, py: Python<'_>) {
        let held = self.held.take();
        py.allow_threads(move || drop(held));
    }

    fn assert_covers(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let path = PathBuf::from(path);
        py.allow_threads(move || {
            self.held
                .as_ref()
                .ok_or(LeaseError::Layout("lease is closed"))?
                .assert_covers(&path)
        })
        .map_err(Into::into)
    }
}

#[pyclass(module = "chunkhound_native")]
pub(crate) struct InstallLease {
    held: Option<Lease>,
    #[allow(dead_code)] // T6 consumes this when its install method lands; remove then.
    backend: Backend,
}

#[pymethods]
impl InstallLease {
    fn close(&mut self, py: Python<'_>) {
        let held = self.held.take();
        py.allow_threads(move || drop(held));
    }

    fn assert_covers(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let path = PathBuf::from(path);
        py.allow_threads(move || {
            self.held
                .as_ref()
                .ok_or(LeaseError::Layout("lease is closed"))?
                .assert_covers(&path)
        })
        .map_err(Into::into)
    }

    #[getter]
    fn interrupted(&self, py: Python<'_>) -> PyResult<bool> {
        py.allow_threads(|| {
            self.held
                .as_ref()
                .ok_or(LeaseError::Layout("lease is closed"))?
                .interrupted()
        })
        .map_err(Into::into)
    }
}

#[pyfunction]
fn acquire_database_lease(py: Python<'_>, path: String) -> PyResult<DatabaseLease> {
    let path = PathBuf::from(path);
    py.allow_threads(move || lease::acquire_shared(&path))
        .map(|held| DatabaseLease { held: Some(held) })
        .map_err(Into::into)
}

#[pyfunction]
fn acquire_install_lease(
    py: Python<'_>,
    target: String,
    backend: String,
) -> PyResult<InstallLease> {
    let path = PathBuf::from(target);
    py.allow_threads(move || {
        let backend = Backend::parse(&backend)?;
        let held = lease::acquire_exclusive(&path, backend)?;
        Ok(InstallLease {
            held: Some(held),
            backend,
        })
    })
    .map_err(|error: LeaseError| error.into())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatabaseLease>()?;
    m.add_class::<InstallLease>()?;
    m.add_function(wrap_pyfunction!(acquire_database_lease, m)?)?;
    m.add_function(wrap_pyfunction!(acquire_install_lease, m)?)?;
    m.add(
        "DatabaseBusyError",
        m.py().get_type_bound::<DatabaseBusyError>(),
    )?;
    m.add(
        "DatabaseInterruptedError",
        m.py().get_type_bound::<DatabaseInterruptedError>(),
    )?;
    m.add(
        "DatabaseLayoutError",
        m.py().get_type_bound::<DatabaseLayoutError>(),
    )?;
    m.add(
        "DatabaseFilesystemError",
        m.py().get_type_bound::<DatabaseFilesystemError>(),
    )?;
    Ok(())
}
