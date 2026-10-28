// PyO3 0.22's create_exception! emits cfg(feature = "gil-refs") in this crate,
// although that feature belongs to PyO3. Keep the workaround local to its macros.
#![allow(unexpected_cfgs)]

use crate::error::PipelineError;
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pyo3::create_exception!(
    chunkhound_native,
    PipelineCancelledError,
    pyo3::exceptions::PyException
);
const POLL: Duration = Duration::from_millis(25);

#[derive(Debug, Default)]
struct State {
    stopped: AtomicBool,
    started: AtomicBool,
    cleanup_complete: AtomicBool,
    wake_lock: Mutex<()>,
    wake: Condvar,
}
#[derive(Debug, Clone, Default)]
pub(crate) struct StopToken(Arc<State>);
impl StopToken {
    pub(crate) fn begin(&self) -> PyResult<()> {
        if self.0.started.swap(true, Ordering::AcqRel) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cancellation token is single-use",
            ));
        }
        Ok(())
    }
    pub(crate) fn cancel(&self) {
        let _guard = self.0.wake_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.0.stopped.store(true, Ordering::Release);
        self.0.wake.notify_all();
    }
    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.stopped.load(Ordering::Acquire)
    }
    pub(crate) fn check(&self) -> Result<(), PipelineError> {
        if self.is_cancelled() {
            Err(PipelineError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub(crate) fn wait(&self, duration: Duration) -> Result<(), PipelineError> {
        let guard = self.0.wake_lock.lock().unwrap_or_else(|e| e.into_inner());
        let _wait = self
            .0
            .wake
            .wait_timeout_while(guard, duration, |_| !self.is_cancelled())
            .unwrap_or_else(|e| e.into_inner());
        self.check()
    }
    pub(crate) fn finish(&self, success: bool) {
        self.0.cleanup_complete.store(success, Ordering::Release);
    }
}
#[pyclass(frozen)]
#[derive(Debug, Clone, Default)]
pub(crate) struct PipelineCancellationToken {
    pub(crate) stop: StopToken,
}
#[pymethods]
impl PipelineCancellationToken {
    #[new]
    fn new() -> Self {
        Self::default()
    }
    fn cancel(&self) {
        self.stop.cancel();
    }
    fn is_cancelled(&self) -> bool {
        self.stop.is_cancelled()
    }
    #[getter]
    fn cleanup_complete(&self) -> bool {
        self.stop.0.cleanup_complete.load(Ordering::Acquire)
    }
}
pub(crate) fn receive<T>(rx: &Receiver<T>, stop: &StopToken) -> Option<T> {
    loop {
        if stop.is_cancelled() {
            return None;
        }
        match rx.recv_timeout(POLL) {
            Ok(value) => return (!stop.is_cancelled()).then_some(value),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
}
pub(crate) fn send<T>(tx: &SyncSender<T>, mut value: T, stop: &StopToken) -> bool {
    loop {
        if stop.is_cancelled() {
            return false;
        }
        match tx.try_send(value) {
            Ok(()) => return true,
            Err(TrySendError::Disconnected(_)) => return false,
            Err(TrySendError::Full(returned)) => value = returned,
        }
        if stop.wait(POLL).is_err() {
            return false;
        }
    }
}
