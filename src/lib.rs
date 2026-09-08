#![forbid(unsafe_code)]
// PyO3 0.22's #[pyfunction] macro emits a PyErr→PyErr .into() in its generated wrapper code,
// which clippy's useless_conversion lint flags. The allow must be crate-level because the lint
// fires in the proc-macro expansion, not in the function's textual body. Fixed upstream in PyO3 0.23+.
#![allow(clippy::useless_conversion)]
mod db;
mod error;
mod types;

mod pipeline;

use crate::error::ScanError;
use ignore::gitignore::GitignoreBuilder;
use ignore::{WalkBuilder, WalkState};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[pyfunction]
#[pyo3(signature = (root, extensions, skip_dirs=None, exclude_patterns=None, exact_names=None, include_all=false))]
fn scan_files(
    py: Python<'_>,
    root: String,
    extensions: Vec<String>,
    skip_dirs: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    exact_names: Option<Vec<String>>,
    include_all: bool,
) -> PyResult<Vec<String>> {
    py.allow_threads(|| {
        scan_files_impl(
            root,
            extensions,
            skip_dirs,
            exclude_patterns,
            exact_names,
            include_all,
        )
    })
    .map_err(Into::into)
}

/// Core file-discovery logic, decoupled from the PyO3/GIL boundary so it can be
/// unit-tested directly -- this crate's `extension-module` PyO3 feature means a
/// standalone `cargo test` binary can't construct a real `Python<'_>` token.
fn scan_files_impl(
    root: String,
    extensions: Vec<String>,
    skip_dirs: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    exact_names: Option<Vec<String>>,
    include_all: bool,
) -> Result<Vec<String>, ScanError> {
    match std::fs::metadata(&root) {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            return Err(ScanError::RootUnreadable {
                root: root.clone(),
                source: std::io::Error::other("path exists but is not a directory"),
            });
        }
        Err(source) => {
            return Err(ScanError::RootUnreadable {
                root: root.clone(),
                source,
            });
        }
    }

    let ext_set = Arc::new(
        extensions
            .into_iter()
            .map(|e| e.to_lowercase())
            .collect::<HashSet<String>>(),
    );
    let name_set = Arc::new(
        exact_names
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<String>>(),
    );
    let skip_set = Arc::new(
        skip_dirs
            .unwrap_or_default()
            .into_iter()
            .collect::<HashSet<String>>(),
    );

    let custom_gi = Arc::new({
        let pats = exclude_patterns.unwrap_or_default();
        if pats.is_empty() {
            None
        } else {
            let mut b = GitignoreBuilder::new(&root);
            for p in &pats {
                let _ = b.add_line(None, p);
            }
            b.build().ok()
        }
    });

    let results: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let error_count = Arc::new(AtomicUsize::new(0));
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    WalkBuilder::new(&root)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .hidden(false)
        .build_parallel()
        .run(|| {
            let ext_set = Arc::clone(&ext_set);
            let name_set = Arc::clone(&name_set);
            let skip_set = Arc::clone(&skip_set);
            let custom_gi = Arc::clone(&custom_gi);
            let results = Arc::clone(&results);
            let error_count = Arc::clone(&error_count);
            let first_error = Arc::clone(&first_error);
            Box::new(move |result| {
                let entry = match result {
                    Ok(e) => e,
                    Err(e) => {
                        // Never let a swallowed walk error (permission denied,
                        // a briefly-unmounted path, a raced-out entry) look
                        // identical to "this subtree is genuinely empty" --
                        // the pipeline treats an empty scan as license to
                        // delete every DB row for files it didn't see.
                        error_count.fetch_add(1, Ordering::Relaxed);
                        let mut fe = first_error.lock().expect("first_error mutex poisoned");
                        if fe.is_none() {
                            *fe = Some(e.to_string());
                        }
                        return WalkState::Continue;
                    }
                };
                let ft = match entry.file_type() {
                    Some(t) => t,
                    None => return WalkState::Continue,
                };
                if ft.is_dir() {
                    let name = entry.file_name().to_string_lossy();
                    if skip_set.contains(name.as_ref()) {
                        return WalkState::Skip;
                    }
                    return WalkState::Continue;
                }
                if !ft.is_file() {
                    return WalkState::Continue;
                }
                let path = entry.path();
                if let Some(ref gi) = *custom_gi {
                    if gi.matched(path, false).is_ignore() {
                        return WalkState::Continue;
                    }
                }
                let file_name = entry.file_name().to_string_lossy();
                let matched = include_all
                    || if let Some(ext) = path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        ext_set.contains(ext_lower.as_str())
                    } else {
                        false
                    }
                    || (!name_set.is_empty() && name_set.contains(file_name.as_ref()));
                if matched {
                    if let Some(s) = path.to_str() {
                        results
                            .lock()
                            .expect("results mutex poisoned")
                            .push(s.to_owned());
                    }
                }
                WalkState::Continue
            })
        });

    let results = Arc::try_unwrap(results)
        .expect("Arc still has live references after walk completed")
        .into_inner()
        .expect("results mutex poisoned");

    let n_errors = error_count.load(Ordering::Relaxed);
    if n_errors > 0 && results.is_empty() {
        // Only fail closed when the walk errors leave *nothing* to show for
        // it. A permission-denied subdirectory alongside otherwise-readable
        // content is an intentional, tested tolerance (see
        // test_permission_error_fallback in tests/test_parallel_discovery.py
        // -- the project treats an inaccessible subtree like an excluded
        // one, not a fatal error, as long as something else was still
        // found). But errors + zero files is indistinguishable from "this
        // project has no files at all", which callers treat as license to
        // delete every DB row -- that specific combination must never be
        // silently reported as an ordinary empty scan.
        let example = first_error
            .lock()
            .expect("first_error mutex poisoned")
            .clone()
            .unwrap_or_default();
        return Err(ScanError::Incomplete {
            root,
            count: n_errors,
            example,
        });
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn create_file(dir: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        fs::write(&path, b"contents").unwrap();
        path
    }

    fn file_names(results: &[String]) -> HashSet<String> {
        results
            .iter()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect()
    }

    #[test]
    fn test_include_all_matches_unknown_and_extensionless_files() {
        let tmp = tempfile::tempdir().unwrap();
        create_file(&tmp, "known.py");
        create_file(&tmp, "unknown.xyz");
        create_file(&tmp, "README");

        let results = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec!["py".to_string()],
            None,
            None,
            None,
            true,
        )
        .unwrap();

        assert_eq!(
            file_names(&results),
            ["known.py", "unknown.xyz", "README"]
                .into_iter()
                .map(String::from)
                .collect::<HashSet<String>>(),
            "include_all=true must match every file regardless of the extensions passed in"
        );
    }

    #[test]
    fn test_include_all_still_respects_skip_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        create_file(&tmp, "top_level.py");
        let heavy_dir = tmp.path().join("node_modules");
        fs::create_dir(&heavy_dir).unwrap();
        fs::write(heavy_dir.join("inside.js"), b"contents").unwrap();

        let results = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec![],
            Some(vec!["node_modules".to_string()]),
            None,
            None,
            true,
        )
        .unwrap();

        assert_eq!(
            file_names(&results),
            ["top_level.py".to_string()]
                .into_iter()
                .collect::<HashSet<String>>(),
            "include_all=true must not bypass skip_dirs pruning"
        );
    }

    #[test]
    fn test_include_all_still_respects_exclude_patterns() {
        let tmp = tempfile::tempdir().unwrap();
        create_file(&tmp, "keep.dat");
        create_file(&tmp, "excluded.dat");

        let results = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec![],
            None,
            Some(vec!["excluded.dat".to_string()]),
            None,
            true,
        )
        .unwrap();

        assert_eq!(
            file_names(&results),
            ["keep.dat".to_string()]
                .into_iter()
                .collect::<HashSet<String>>(),
            "include_all=true must not bypass custom exclude_patterns"
        );
    }

    #[test]
    fn test_include_all_false_preserves_existing_extension_filter() {
        let tmp = tempfile::tempdir().unwrap();
        create_file(&tmp, "known.py");
        create_file(&tmp, "unknown.xyz");

        let results = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec!["py".to_string()],
            None,
            None,
            None,
            false,
        )
        .unwrap();

        assert_eq!(
            file_names(&results),
            ["known.py".to_string()]
                .into_iter()
                .collect::<HashSet<String>>(),
            "default include_all=false must keep the existing extension allow-list behavior"
        );
    }

    #[cfg(unix)]
    #[test]
    fn walk_error_with_zero_files_found_fails_closed() {
        use std::os::unix::fs::PermissionsExt;

        let tmp = tempfile::tempdir().unwrap();
        let locked_dir = tmp.path().join("locked");
        fs::create_dir(&locked_dir).unwrap();
        fs::write(locked_dir.join("secret.py"), b"contents").unwrap();
        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o000)).unwrap();

        if fs::read_dir(&locked_dir).is_ok() {
            // Running as root (or another environment where permission bits
            // don't apply) -- the walk error under test can't be produced here.
            fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }

        let result = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec!["py".to_string()],
            None,
            None,
            None,
            false,
        );

        // Restore permissions before asserting so tempdir cleanup always succeeds,
        // regardless of whether the assertion below panics.
        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o755)).unwrap();

        assert!(
            result.is_err(),
            "a walk error that leaves zero files found must fail closed instead of \
             silently reporting an empty scan -- an empty scan is treated downstream \
             as license to delete every existing DB row"
        );
    }

    #[cfg(unix)]
    #[test]
    fn partial_scan_with_some_files_found_tolerates_the_error() {
        use std::os::unix::fs::PermissionsExt;

        // Mirrors tests/test_parallel_discovery.py::test_permission_error_fallback --
        // an inaccessible subdirectory alongside otherwise-readable content is an
        // intentional, tested tolerance (equivalent to an excluded subtree), not a
        // fatal error, as long as something else was still found.
        let tmp = tempfile::tempdir().unwrap();
        create_file(&tmp, "visible.py");
        let locked_dir = tmp.path().join("locked");
        fs::create_dir(&locked_dir).unwrap();
        fs::write(locked_dir.join("secret.py"), b"contents").unwrap();
        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o000)).unwrap();

        if fs::read_dir(&locked_dir).is_ok() {
            fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }

        let result = scan_files_impl(
            tmp.path().to_string_lossy().into_owned(),
            vec!["py".to_string()],
            None,
            None,
            None,
            false,
        );

        fs::set_permissions(&locked_dir, fs::Permissions::from_mode(0o755)).unwrap();

        let files = result.expect(
            "a scan that errors on part of the tree but still finds files elsewhere \
             must still succeed with the partial list, matching the project's existing \
             tolerance for inaccessible subtrees",
        );
        assert_eq!(
            file_names(&files),
            ["visible.py".to_string()]
                .into_iter()
                .collect::<HashSet<String>>()
        );
    }

    #[test]
    fn nonexistent_root_fails_closed_with_root_unreadable() {
        let result = scan_files_impl(
            "/definitely/does/not/exist/chunkhound-test".to_string(),
            vec!["py".to_string()],
            None,
            None,
            None,
            false,
        );

        assert!(
            matches!(result, Err(ScanError::RootUnreadable { .. })),
            "a nonexistent root must fail closed with a RootUnreadable error, got: {result:?}"
        );
    }

    #[test]
    fn root_path_that_is_a_file_fails_closed_with_root_unreadable() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = create_file(&tmp, "not_a_dir.txt");

        let result = scan_files_impl(
            file_path.to_string_lossy().into_owned(),
            vec!["py".to_string()],
            None,
            None,
            None,
            false,
        );

        assert!(
            matches!(result, Err(ScanError::RootUnreadable { .. })),
            "a root path that is a file (not a directory) must fail closed with a \
             RootUnreadable error, got: {result:?}"
        );
    }
}

#[pymodule]
fn chunkhound_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize pyo3-log (Rust logs → Python logging)
    pyo3_log::init();

    m.add_function(wrap_pyfunction!(scan_files, m)?)?;

    m.add_class::<pipeline::IndexingPipeline>()?;
    m.add_class::<pipeline::PipelineReport>()?;
    m.add_class::<pipeline::ParseCallConfig>()?;

    Ok(())
}
