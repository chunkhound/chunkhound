use super::{test_support, DuckDbHnswBackend};
use crate::database_coordination::{
    error::LeaseError,
    lease::{acquire_exclusive, sibling, Backend},
};
use crate::db::DbBackend;
use crate::error::DbError;

#[cfg(unix)]
mod non_utf8_targets {
    use super::*;
    use std::collections::BTreeMap;
    use std::ffi::OsStr;
    use std::os::unix::{ffi::OsStrExt, fs::symlink};
    use std::path::Path;

    fn assert_rejected_without_mutation(read_only: bool) {
        for db_present in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let original = tmp.path().join(OsStr::from_bytes(b"index-\xff"));
            let neighbor = tmp.path().join("index-\u{fffd}");
            std::fs::create_dir(&original).unwrap();
            std::fs::create_dir(&neighbor).unwrap();
            let alias = tmp.path().join("alias");
            symlink(&original, &alias).unwrap();
            for directory in [&original, &neighbor] {
                let target = directory.join("chunks.db");
                for suffix in ["", ".old", ".compact", ".wal", ".swap_intent"] {
                    if suffix.is_empty() && !db_present {
                        continue;
                    }
                    std::fs::write(sibling(&target, suffix), b"untouched input").unwrap();
                }
                // Include the permitted stable file in the baseline; never filter it out.
                drop(crate::database_coordination::lease::acquire_shared(&target).unwrap());
            }
            let snapshot = |directory: &Path| {
                std::fs::read_dir(directory)
                    .unwrap()
                    .map(|entry| {
                        let path = entry.unwrap().path();
                        let bytes = std::fs::read(&path).unwrap();
                        let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
                        (path, (bytes, modified))
                    })
                    .collect::<BTreeMap<_, _>>()
            };
            let before = (snapshot(&original), snapshot(&neighbor));
            let path = alias.join("chunks.db");
            let mut backend =
                DuckDbHnswBackend::new(test_support::config(path.to_str().unwrap().to_owned()));
            let result = if read_only {
                backend.read_file_states().map(|_| ())
            } else {
                backend.open()
            };
            assert!(
                matches!(result, Err(DbError::Coordination(LeaseError::Layout(_)))),
                "expected typed layout refusal before preflight/recovery: {result:?}"
            );
            assert_eq!((snapshot(&original), snapshot(&neighbor)), before);
            // Keep the failed backend alive: ownership must already have been released.
            drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
            drop(acquire_exclusive(&neighbor.join("chunks.db"), Backend::DuckDb).unwrap());
            assert_eq!((snapshot(&original), snapshot(&neighbor)), before);
        }
    }

    #[test]
    fn direct_open_rejects_non_utf8_target_before_preflight_and_recovery() {
        assert_rejected_without_mutation(false);
    }

    #[test]
    fn direct_read_rejects_non_utf8_target_before_preflight_and_recovery() {
        assert_rejected_without_mutation(true);
    }
}

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
