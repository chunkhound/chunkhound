//! File diff engine — compares filesystem state against the DB snapshot
//! to produce a minimal set of files that need re-processing.

use std::collections::{HashMap, HashSet};
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
    /// Files whose mtime differed from the DB but whose content hash matched
    /// the DB's stored hash — confirmed unchanged, never entered `changed`.
    pub skipped_by_hash: u64,
    /// Content hash freshly computed for each file remaining in `changed`
    /// (keyed by the same absolute path used there), so the parse stage
    /// doesn't need to re-hash a file the diff phase already read.
    pub new_hashes: HashMap<PathBuf, String>,
}

/// Snapshot of a single file from the DB.
#[derive(Debug, Clone)]
pub(crate) struct DbFileEntry {
    pub(crate) path: String,
    pub(crate) mtime: f64, // Unix timestamp
    pub(crate) content_hash: Option<String>,
}

/// How often (in files scanned) to invoke `on_tick` — bounds the number of
/// PyO3 call-into-Python round-trips for large repos; Rich's own terminal
/// refresh is already throttled separately (10Hz), this throttle is purely
/// about callback overhead.
const DIFF_TICK_INTERVAL: usize = 200;

/// Compute the diff between the files provided and the DB state.
///
/// Returns the set of files that need re-processing, plus the set
/// of DB paths that should be deleted.
///
/// `db_file_entries` is the result of querying
/// `SELECT path, modified_time, content_hash FROM files`.
/// `files_on_disk` are the absolute paths provided by the caller (scanner).
/// `mtime_epsilon` controls how close two timestamps must be to be considered equal.
/// `on_tick`, if provided, is called periodically with `(files_scanned, total)`.
pub(crate) fn compute_diff(
    db_file_entries: &[DbFileEntry],
    files_on_disk: &[PathBuf],
    project_root: &Path,
    mtime_epsilon: f64,
    mut on_tick: Option<&mut dyn FnMut(usize, usize)>,
) -> DiffResult {
    // Build a lookup: DB path → mtime
    let db_map: HashMap<&str, f64> = db_file_entries
        .iter()
        .map(|e| (e.path.as_str(), e.mtime))
        .collect();

    // Build a lookup: DB path → stored content hash (only entries that have one).
    let db_hashes: HashMap<&str, &str> = db_file_entries
        .iter()
        .filter_map(|e| e.content_hash.as_deref().map(|h| (e.path.as_str(), h)))
        .collect();

    // Build a set of DB paths for removal detection
    let db_paths: HashSet<String> = db_file_entries.iter().map(|e| e.path.clone()).collect();

    let mut changed = Vec::new();
    let mut disk_paths = HashSet::new();
    let mut files_scanned = 0;
    let mut skipped_by_hash = 0u64;
    let mut new_hashes: HashMap<PathBuf, String> = HashMap::new();

    for abs_path in files_on_disk {
        files_scanned += 1;

        if let Some(tick) = on_tick.as_deref_mut() {
            if files_scanned % DIFF_TICK_INTERVAL == 0 {
                tick(files_scanned, files_on_disk.len());
            }
        }

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
                    // mtime changed — before assuming the content changed,
                    // check the content hash if the DB has one stored for
                    // this path. A touch/checkout/CI-restore can bump mtime
                    // without changing bytes; confirming via hash avoids a
                    // wasted re-embed + chunk rewrite for files that are
                    // actually unchanged.
                    match db_hashes.get(rel.as_str()) {
                        Some(&stored_hash) => match hash_file_contents(abs_path) {
                            Some(new_hash) if new_hash == stored_hash => {
                                // Content confirmed unchanged despite the
                                // mtime bump — skip entirely, never enters
                                // the parse/embed/store pipeline.
                                skipped_by_hash += 1;
                            }
                            Some(new_hash) => {
                                new_hashes.insert(abs_path.clone(), new_hash);
                                changed.push(abs_path.clone());
                            }
                            None => {
                                // Couldn't read/hash — fall back to the safe
                                // default of reprocessing.
                                changed.push(abs_path.clone());
                            }
                        },
                        None => {
                            // No prior hash stored for this path — can't
                            // confirm unchanged. Reprocess, but stash a hash
                            // now so future runs can compare.
                            if let Some(new_hash) = hash_file_contents(abs_path) {
                                new_hashes.insert(abs_path.clone(), new_hash);
                            }
                            changed.push(abs_path.clone());
                        }
                    }
                }
                // else: mtime matches → skip
            } else {
                // Can't stat the file → process it anyway (safety)
                changed.push(abs_path.clone());
            }
        } else {
            // New file not in DB — always changed; stash a hash now so
            // future runs have something to compare against.
            if let Some(new_hash) = hash_file_contents(abs_path) {
                new_hashes.insert(abs_path.clone(), new_hash);
            }
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
        skipped_by_hash,
        new_hashes,
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

/// Compute a content hash for `path`, formatted to match Python's
/// `xxhash.xxh3_64(...).hexdigest()` exactly (16 lowercase hex chars) — see
/// `test_hash_matches_python_xxh3_64` below for the cross-language golden
/// check. Returns `None` if the file can't be read; callers treat that as
/// "no hash available" and fall back to reprocessing.
fn hash_file_contents(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let digest = xxhash_rust::xxh3::xxh3_64(&bytes);
    Some(format!("{digest:016x}"))
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
        let diff = compute_diff(&[], &files, tmp.path(), 0.01, None);

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
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None);
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
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None);
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
                content_hash: None,
            },
            DbFileEntry {
                path: "gone.py".into(), // this file doesn't exist on disk
                mtime: 1.0,
                content_hash: None,
            },
        ];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None);
        assert_eq!(diff.changed_count(), 0); // a.py unchanged
        assert_eq!(diff.removed_count(), 1);
        assert!(diff.removed.contains(&"gone.py".to_string()));
    }

    #[test]
    fn test_hash_matches_python_xxh3_64() {
        // Golden values from:
        //   python -c "import xxhash; print(xxhash.xxh3_64(b'hello world').hexdigest())"
        //   python -c "import xxhash; print(xxhash.xxh3_64(b'').hexdigest())"
        let tmp = tempfile::tempdir().unwrap();

        let path = tmp.path().join("golden.txt");
        std::fs::write(&path, b"hello world").unwrap();
        assert_eq!(
            hash_file_contents(&path).unwrap(),
            "d447b1ea40e6988b",
            "Rust xxh3_64 digest must match Python's xxhash.xxh3_64().hexdigest() bit-for-bit"
        );

        let empty_path = tmp.path().join("empty.txt");
        std::fs::write(&empty_path, b"").unwrap();
        assert_eq!(
            hash_file_contents(&empty_path).unwrap(),
            "2d06800538d394c2",
            "empty-file hash must also match Python's xxhash output"
        );
    }

    #[test]
    fn test_hash_confirms_unchanged_despite_mtime_bump() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let hash = hash_file_contents(&f1).unwrap();

        let db = vec![DbFileEntry {
            path: "a.py".into(),
            mtime: 0.0, // clearly different from the file's real mtime
            content_hash: Some(hash),
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None);
        assert_eq!(
            diff.changed_count(),
            0,
            "hash match should skip despite mtime differing"
        );
        assert_eq!(diff.skipped_by_hash, 1);
    }

    #[test]
    fn test_hash_mismatch_still_reprocesses() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        let db = vec![DbFileEntry {
            path: "a.py".into(),
            mtime: 0.0,
            content_hash: Some("deadbeefdeadbeef".into()),
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None);
        assert_eq!(
            diff.changed_count(),
            1,
            "hash mismatch should still reprocess"
        );
        assert_eq!(diff.skipped_by_hash, 0);
        assert!(diff.new_hashes.contains_key(&f1));
    }

    fn create_file(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# test file").unwrap();
        path
    }
}
