use ignore::gitignore::GitignoreBuilder;
use ignore::{WalkBuilder, WalkState};
use pyo3::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[pyfunction]
#[pyo3(signature = (root, extensions, skip_dirs=None, exclude_patterns=None))]
fn scan_files(
    root: &str,
    extensions: Vec<String>,
    skip_dirs: Option<Vec<String>>,
    exclude_patterns: Option<Vec<String>>,
) -> PyResult<Vec<String>> {
    let ext_set = Arc::new(
        extensions
            .into_iter()
            .map(|e| e.to_lowercase())
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
            let mut b = GitignoreBuilder::new(root);
            for p in &pats {
                let _ = b.add_line(None, p);
            }
            b.build().ok()
        }
    });

    let results: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    WalkBuilder::new(root)
        .git_ignore(true)
        .git_global(false)
        .hidden(false)
        .build_parallel()
        .run(|| {
            let ext_set = Arc::clone(&ext_set);
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
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if ext_set.contains(ext_lower.as_str()) {
                        if let Some(s) = path.to_str() {
                            results.lock().unwrap().push(s.to_owned());
                        }
                    }
                }
                WalkState::Continue
            })
        });

    Ok(Arc::try_unwrap(results).unwrap().into_inner().unwrap())
}

#[pymodule]
fn chunkhound_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scan_files, m)?)?;
    Ok(())
}
