//! File diff engine — compares filesystem state against the DB snapshot
//! to produce a minimal set of files that need re-processing.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Result of diffing the filesystem against the DB.
#[derive(Debug, Default)]
pub(crate) struct DiffResult {
    /// Files that are new or whose mtime has changed — must be re-parsed.
    pub changed: Vec<PathBuf>,
    /// DB file paths that no longer exist on disk — need to be deleted.
    pub removed: Vec<String>,
    /// Total files scanned on disk (for reporting).
    pub files_scanned: usize,
}

/// Snapshot of a single file from the DB.
#[derive(Debug, Clone)]
pub(crate) struct DbFileEntry {
    pub(crate) path: String,
    pub(crate) mtime: f64, // Unix timestamp
}

/// Compute the diff between the files provided and the DB state.
///
/// Returns the set of files that need re-processing, plus the set
/// of DB paths that should be deleted.
///
/// `db_file_entries` is the result of querying `SELECT path, modified_time FROM files`.
/// `files_on_disk` are the absolute paths provided by the caller (scanner).
/// `mtime_epsilon` controls how close two timestamps must be to be considered equal.
pub(crate) fn compute_diff(
    db_file_entries: &[DbFileEntry],
    files_on_disk: &[PathBuf],
    project_root: &Path,
    mtime_epsilon: f64,
) -> DiffResult {
    // Build a lookup: DB path → mtime
    let db_map: std::collections::HashMap<&str, f64> = db_file_entries
        .iter()
        .map(|e| (e.path.as_str(), e.mtime))
        .collect();

    // Build a set of DB paths for removal detection
    let db_paths: HashSet<String> = db_file_entries.iter().map(|e| e.path.clone()).collect();

    let mut changed = Vec::new();
    let mut disk_paths = HashSet::new();
    let mut files_scanned = 0;

    for abs_path in files_on_disk {
        files_scanned += 1;

        // Compute relative path (matching Python's _get_relative_path)
        let rel = match abs_path.strip_prefix(project_root) {
            Ok(p) => p.to_string_lossy().replace('\\', "/"),
            Err(_) => {
                // Can't relativize — process it anyway
                changed.push(abs_path.clone());
                continue;
            }
        };

        disk_paths.insert(rel.clone());

        // Check mtime
        let current_mtime = file_mtime(abs_path);

        if let Some(&db_mtime) = db_map.get(rel.as_str()) {
            // File exists in DB — check if mtime changed
            if let Some(cur) = current_mtime {
                if (cur - db_mtime).abs() > mtime_epsilon {
                    // mtime changed — needs re-processing
                    changed.push(abs_path.clone());
                }
                // else: mtime matches → skip
            } else {
                // Can't stat the file → process it anyway (safety)
                changed.push(abs_path.clone());
            }
        } else {
            // New file not in DB
            changed.push(abs_path.clone());
        }
    }

    // Files in DB but NOT on disk → need removal
    let removed: Vec<String> = db_paths
        .difference(&disk_paths)
        .map(|s| (*s).to_string())
        .collect();

    DiffResult {
        changed,
        removed,
        files_scanned,
    }
}

/// Read the mtime of a file as a Unix timestamp (seconds).
fn file_mtime(path: &Path) -> Option<f64> {
    std::fs::metadata(path).ok()?.modified().ok().map(|t| {
        t.duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    })
}

impl DiffResult {
    /// Number of files that need processing (changed).
    pub fn changed_count(&self) -> usize {
        self.changed.len()
    }

    /// Number of files to be removed.
    pub fn removed_count(&self) -> usize {
        self.removed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_empty_db_all_new() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let f2 = create_file(&tmp, "b.rs");

        let files = vec![f1.clone(), f2.clone()];
        let diff = compute_diff(&[], &files, tmp.path(), 0.01);

        assert_eq!(diff.changed_count(), 2);
        assert_eq!(diff.removed_count(), 0);
    }

    #[test]
    fn test_unchanged_files_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let mtime = file_mtime(&f1).unwrap();

        let db = vec![DbFileEntry {
            path: "a.py".into(),
            mtime,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01);
        assert!(diff.changed.is_empty(), "unchanged file should be skipped");
    }

    #[test]
    fn test_mtime_change_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let old_mtime = 0.0; // epoch — clearly different

        let db = vec![DbFileEntry {
            path: "a.py".into(),
            mtime: old_mtime,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01);
        assert_eq!(
            diff.changed_count(),
            1,
            "changed mtime should trigger re-process"
        );
    }

    #[test]
    fn test_removed_files_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        let db = vec![
            DbFileEntry {
                path: "a.py".into(),
                mtime: file_mtime(&f1).unwrap(),
            },
            DbFileEntry {
                path: "gone.py".into(), // this file doesn't exist on disk
                mtime: 1.0,
            },
        ];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01);
        assert_eq!(diff.changed_count(), 0); // a.py unchanged
        assert_eq!(diff.removed_count(), 1);
        assert!(diff.removed.contains(&"gone.py".to_string()));
    }

    fn create_file(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# test file").unwrap();
        path
    }
}
