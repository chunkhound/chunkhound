use super::{test_support, DuckDbHnswBackend};
use crate::database_coordination::{
    error::LeaseError,
    lease::{acquire_exclusive, sibling, Backend},
};
use crate::db::DbBackend;
use crate::error::DbError;

#[test]
fn direct_reader_checks_marker_before_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    std::fs::write(sibling(&path, ".chunkhound-install.json"), b"{").unwrap();
    std::fs::write(sibling(&path, ".old"), b"old recovery input").unwrap();
    std::fs::write(sibling(&path, ".swap_intent"), b"phase1").unwrap();
    let backend = DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    assert!(matches!(
        backend.read_file_states(),
        Err(DbError::Coordination(LeaseError::Interrupted))
    ));
    assert!(!path.exists());
    assert_eq!(
        std::fs::read(sibling(&path, ".old")).unwrap(),
        b"old recovery input"
    );
}

#[test]
fn open_close_and_implicit_drop_release_only_after_owned_connection() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    let mut backend =
        DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    backend.open().unwrap();
    backend.open().unwrap();
    assert!(matches!(
        acquire_exclusive(&path, Backend::DuckDb),
        Err(LeaseError::Busy)
    ));
    backend.close().unwrap();
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
    backend.open().unwrap();
    drop(backend);
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
}

#[test]
fn compaction_keeps_ownership_while_live_connection_is_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    let mut backend =
        DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    backend.open().unwrap();
    backend
        .write_batch(&test_support::single_file_batch("source.py"))
        .unwrap();
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed_in_hook = observed.clone();
    backend.compaction_closed_hook = Some(Box::new(move |path| {
        assert!(matches!(
            acquire_exclusive(path, Backend::DuckDb),
            Err(LeaseError::Busy)
        ));
        observed_in_hook.store(true, std::sync::atomic::Ordering::SeqCst);
    }));
    backend.run_compaction().unwrap();
    assert!(observed.load(std::sync::atomic::Ordering::SeqCst));
    backend.close().unwrap();
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
}
