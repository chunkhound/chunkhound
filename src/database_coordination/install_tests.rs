use super::*;
use crate::database_coordination::lease::{acquire_exclusive, acquire_shared};
use std::time::{Duration, Instant};

#[test]
fn filesystem_failures_preserve_old_or_refuse_open_and_cleanup_is_only_warning() {
    let cases = [
        (InstallPoint::CandidatePreflightRename, false, false),
        (InstallPoint::MarkerCreate, false, false),
        (InstallPoint::Isolate(0), true, false),
        (InstallPoint::Isolate(1), true, false),
        (InstallPoint::CandidateRename, true, false),
        (InstallPoint::MarkerRemove, true, false),
        (InstallPoint::QuarantineCleanup, false, true),
    ];
    for (point, interrupted, committed) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("chunks.db");
        let stage = tmp.path().join("stage");
        fs::create_dir(&stage).unwrap();
        let candidate = stage.join("chunks.db");
        fs::write(&target, b"old").unwrap();
        fs::write(sibling(&target, ".wal"), b"old wal").unwrap();
        fs::write(&candidate, b"new").unwrap();
        let lease = acquire_exclusive(&target, Backend::DuckDb).unwrap();
        let result = install_with(&lease, Backend::DuckDb, &candidate, &mut |at| {
            if at == point {
                Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected"))
            } else {
                Ok(())
            }
        });
        if committed {
            assert_eq!(result.unwrap(), vec!["quarantine_cleanup_failed"]);
            assert_eq!(fs::read(&target).unwrap(), b"new");
            assert!(!sibling(&target, ".wal").exists());
        } else if interrupted {
            assert!(matches!(result, Err(LeaseError::Interrupted)));
        } else {
            assert!(matches!(result, Err(LeaseError::Filesystem(_))));
            assert_eq!(fs::read(&target).unwrap(), b"old");
            assert_eq!(fs::read(sibling(&target, ".wal")).unwrap(), b"old wal");
        }
        assert_eq!(lease.interrupted().unwrap(), interrupted);
        drop(lease);
        assert_eq!(
            matches!(acquire_shared(&target), Err(LeaseError::Interrupted)),
            interrupted
        );
    }
}

#[test]
#[ignore = "subprocess entry point, invoked by kill tests"]
fn installer_child() {
    let target = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_TARGET").unwrap());
    let candidate = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_CANDIDATE").unwrap());
    let ready = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_READY").unwrap());
    let phase = std::env::var("CH_TEST_INSTALL_PHASE").unwrap();
    let backend = Backend::parse(&std::env::var("CH_TEST_INSTALL_BACKEND").unwrap()).unwrap();
    let point = match phase.as_str() {
        "marker" => InstallPoint::MarkerPublished,
        "isolate" => InstallPoint::Isolated(0),
        "rename" => InstallPoint::CandidateRenamed,
        "commit" => InstallPoint::Committed,
        _ => panic!("unknown test phase"),
    };
    let lease = acquire_exclusive(&target, backend).unwrap();
    install_with(&lease, backend, &candidate, &mut |at| {
        if at == point {
            fs::write(&ready, b"ready")?;
            loop {
                std::thread::park();
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn killed_installer_refuses_partial_state_and_fresh_repull_recovers() {
    for backend in [Backend::DuckDb, Backend::LanceDb] {
        for phase in ["marker", "isolate", "rename", "commit"] {
            let tmp = tempfile::tempdir().unwrap();
            let target = tmp.path().join(backend.filename());
            let stage = tmp.path().join("stage");
            fs::create_dir(&stage).unwrap();
            let candidate = stage.join(backend.filename());
            for (path, bytes) in [
                (&target, b"old".as_slice()),
                (&candidate, b"new".as_slice()),
            ] {
                match backend {
                    Backend::DuckDb => fs::write(path, bytes).unwrap(),
                    Backend::LanceDb => {
                        fs::create_dir(path).unwrap();
                        fs::write(path.join("payload"), bytes).unwrap();
                    }
                }
            }
            let neighbor = tmp.path().join("notes.txt");
            fs::write(&neighbor, b"keep").unwrap();
            let ready = tmp.path().join("ready");
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "database_coordination::install::tests::installer_child",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                ])
                .env("CH_TEST_INSTALL_TARGET", &target)
                .env("CH_TEST_INSTALL_CANDIDATE", &candidate)
                .env("CH_TEST_INSTALL_READY", &ready)
                .env("CH_TEST_INSTALL_PHASE", phase)
                .env(
                    "CH_TEST_INSTALL_BACKEND",
                    if backend == Backend::DuckDb {
                        "duckdb"
                    } else {
                        "lancedb"
                    },
                )
                .spawn()
                .unwrap();
            let deadline = Instant::now() + Duration::from_secs(20);
            while !ready.exists() && Instant::now() < deadline {
                if child.try_wait().unwrap().is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let reached = ready.exists();
            // Collect the observations before killing, but assert only after reaping
            // so a failed exclusion assertion cannot leave a parked child behind.
            let alive = child.try_wait().unwrap().is_none();
            let exclusive_busy =
                reached && matches!(acquire_exclusive(&target, backend), Err(LeaseError::Busy));
            let shared_busy = reached && matches!(acquire_shared(&target), Err(LeaseError::Busy));
            let _ = child.kill();
            child.wait().unwrap();
            assert!(
                reached && alive,
                "installer did not remain paused at {phase}"
            );
            assert!(
                exclusive_busy,
                "second installer was not excluded at {phase}"
            );
            assert!(shared_busy, "cold opener was not excluded at {phase}");
            assert_eq!(fs::read(&neighbor).unwrap(), b"keep");
            if phase != "commit" {
                assert!(matches!(
                    acquire_shared(&target),
                    Err(LeaseError::Interrupted)
                ));
            } else {
                drop(acquire_shared(&target).unwrap());
                let payload = if backend == Backend::DuckDb {
                    target.clone()
                } else {
                    target.join("payload")
                };
                assert_eq!(fs::read(payload).unwrap(), b"new");
            }
            // Fresh candidate; never restore/reuse data from abandoned quarantine.
            let fresh_parent = tmp.path().join("fresh");
            fs::create_dir(&fresh_parent).unwrap();
            let fresh = fresh_parent.join(backend.filename());
            if backend == Backend::DuckDb {
                fs::write(&fresh, b"recovered").unwrap();
            } else {
                fs::create_dir(&fresh).unwrap();
                fs::write(fresh.join("payload"), b"recovered").unwrap();
            }
            let recovery = acquire_exclusive(&target, backend).unwrap();
            assert!(install(&recovery, backend, &fresh).unwrap().is_empty());
            drop(recovery);
            drop(acquire_shared(&target).unwrap());
            let payload = if backend == Backend::DuckDb {
                target.clone()
            } else {
                target.join("payload")
            };
            assert_eq!(fs::read(payload).unwrap(), b"recovered");
            assert_eq!(fs::read(&neighbor).unwrap(), b"keep");
        }
    }
}
