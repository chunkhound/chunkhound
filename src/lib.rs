#![forbid(unsafe_code)]
// PyO3 0.22's #[pyfunction] macro emits a PyErr→PyErr .into() in its generated wrapper code,
// which clippy's useless_conversion lint flags. The allow must be crate-level because the lint
// fires in the proc-macro expansion, not in the function's textual body. Fixed upstream in PyO3 0.23+.
#![allow(clippy::useless_conversion)]

use ignore::gitignore::GitignoreBuilder;
use ignore::{WalkBuilder, WalkState};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[pyfunction]
#[pyo3(signature = (root, extensions, skip_dirs=None, exclude_patterns=None, exact_names=None))]
fn scan_files(
    py: Python<'_>,
    root: String,
    extensions: Vec<String>,
    skip_dirs: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
    exact_names: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
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
                // Patterns are fully normalized to gitignore syntax by Python's
                // _fnmatch_to_gitignore before being passed here. Directory subtree
                // patterns keep their "/**" suffix; bare extension/name patterns
                // (e.g. "*.pyc") have "**/" stripped — gitignore bare patterns
                // without a "/" already match at any depth, so no re-addition needed.
                let _ = b.add_line(None, p);
            }
            b.build().ok()
        }
    });

    let results: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    py.allow_threads(|| {
        WalkBuilder::new(&root)
            .git_ignore(true)
            .git_global(false)
            .git_exclude(false) // .git/info/exclude not modeled by Python path
            .ignore(false) // .ignore files not modeled by Python path
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
                    let matched = if let Some(ext) = path.extension() {
                        let ext_lower = ext.to_string_lossy().to_lowercase();
                        ext_set.contains(ext_lower.as_str())
                    } else {
                        false
                    } || (!name_set.is_empty()
                        && name_set.contains(file_name.as_ref()));
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
    });

    Ok(Arc::try_unwrap(results)
        .expect("Arc still has live references after walk completed")
        .into_inner()
        .expect("results mutex poisoned"))
}

#[pymodule]
fn chunkhound_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_files, m)?)?;
    Ok(())
}
