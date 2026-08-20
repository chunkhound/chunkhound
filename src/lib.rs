#![forbid(unsafe_code)]
// PyO3 0.22's #[pyfunction] macro emits a PyErr→PyErr .into() in its generated wrapper code,
// which clippy's useless_conversion lint flags. The allow must be crate-level because the lint
// fires in the proc-macro expansion, not in the function's textual body. Fixed upstream in PyO3 0.23+.
#![allow(clippy::useless_conversion)]
mod db;
mod error;
mod types;

mod pipeline;

use ignore::gitignore::GitignoreBuilder;
use ignore::{WalkBuilder, WalkState};
use pyo3::prelude::*;
use std::collections::HashSet;
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
    Ok(py.allow_threads(|| {
        scan_files_impl(
            root,
            extensions,
            skip_dirs,
            exclude_patterns,
            exact_names,
            include_all,
        )
    }))
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
) -> Vec<String> {
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
            Box::new(move |result| {
                let entry = match result {
                    Ok(e) => e,
                    Err(_) => return WalkState::Continue,
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

    Arc::try_unwrap(results)
        .expect("Arc still has live references after walk completed")
        .into_inner()
        .expect("results mutex poisoned")
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
        );

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
        );

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
        );

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
        );

        assert_eq!(
            file_names(&results),
            ["known.py".to_string()]
                .into_iter()
                .collect::<HashSet<String>>(),
            "default include_all=false must keep the existing extension allow-list behavior"
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
