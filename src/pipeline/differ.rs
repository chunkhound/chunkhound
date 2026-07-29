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
    /// DB row id for each file in `changed` that already existed in the DB
    /// (keyed by absolute path). New files (first index) are absent here.
    /// Lets the write phase skip the per-file SELECT in upsert_file.
    pub existing_ids: HashMap<PathBuf, i64>,
    /// (size_bytes, mtime) for every file in `changed`, keyed by absolute path.
    /// Populated by the diff phase so the parse stage can skip re-stat-ing
    /// files whose metadata was already read here.
    pub disk_stats: HashMap<PathBuf, (u64, f64)>,
}

/// Snapshot of a single file from the DB.
#[derive(Debug, Clone)]
pub(crate) struct DbFileEntry {
    pub(crate) id: i64,
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
/// `SELECT id, path, modified_time, content_hash FROM files`.
/// `files_on_disk` are the absolute paths provided by the caller (scanner).
/// `mtime_epsilon` controls how close two timestamps must be to be considered equal.
/// `precomputed_stats`, if provided, is a map of absolute path → (size_bytes, mtime)
/// produced by a prior stat pass (e.g. the TZ-offset pass in `compute_diff_blocking`).
/// When present, `compute_diff` reuses those values instead of calling `stat` again.
/// `on_tick`, if provided, is called periodically with `(files_scanned, total)`.
pub(crate) fn compute_diff(
    db_file_entries: &[DbFileEntry],
    files_on_disk: &[PathBuf],
    project_root: &Path,
    mtime_epsilon: f64,
    precomputed_stats: Option<&HashMap<PathBuf, (u64, f64)>>,
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

    // Build a lookup: DB path → DB row id, so changed files that already existed
    // in the DB carry their id into the write phase and skip the per-file SELECT.
    let db_ids: HashMap<&str, i64> = db_file_entries
        .iter()
        .map(|e| (e.path.as_str(), e.id))
        .collect();

    // Build a set of DB paths for removal detection
    let db_paths: HashSet<String> = db_file_entries.iter().map(|e| e.path.clone()).collect();

    let mut changed = Vec::new();
    let mut disk_paths = HashSet::new();
    let mut files_scanned = 0;
    let mut skipped_by_hash = 0u64;
    let mut new_hashes: HashMap<PathBuf, String> = HashMap::new();
    let mut existing_ids: HashMap<PathBuf, i64> = HashMap::new();
    let mut disk_stats: HashMap<PathBuf, (u64, f64)> = HashMap::new();

    for abs_path in files_on_disk {
        files_scanned += 1;

        if let Some(tick) = on_tick.as_deref_mut() {
            if files_scanned % DIFF_TICK_INTERVAL == 0 {
                tick(files_scanned, files_on_disk.len());
            }
        }

        // Resolve disk stats once per file — reuse precomputed_stats when available
        // (eliminates a second stat pass over the repo), falling back to metadata()
        // when called from tests or non-incremental paths that have no precomputed map.
        let (current_size, current_mtime_raw) = precomputed_stats
            .and_then(|pre| pre.get(abs_path).copied())
            .unwrap_or_else(|| {
                let meta = std::fs::metadata(abs_path).ok();
                let mtime = meta
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);
                (meta.map_or(0, |m| m.len()), mtime)
            });
        // Mirror the old file_mtime() contract: None means stat failed (size == 0 and
        // mtime == 0.0 from the fallback above); Some wraps the actual timestamp.
        let current_mtime: Option<f64> = if current_mtime_raw > 0.0 || current_size > 0 {
            Some(current_mtime_raw)
        } else {
            // Both are zero → could be a genuine epoch file, but more likely stat failed.
            // Preserve the original safety behaviour: try a fresh metadata call to confirm.
            std::fs::metadata(abs_path)
                .ok()
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    t.duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_secs_f64())
                        .unwrap_or(0.0)
                })
        };

        // Compute relative path (matching Python's _get_relative_path)
        let rel = match abs_path.strip_prefix(project_root) {
            Ok(p) => p.to_string_lossy().replace('\\', "/"),
            Err(_) => {
                // Can't relativize — process it anyway
                disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
                changed.push(abs_path.clone());
                continue;
            }
        };

        disk_paths.insert(rel.clone());

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
                                if let Some(&id) = db_ids.get(rel.as_str()) {
                                    existing_ids.insert(abs_path.clone(), id);
                                }
                                disk_stats
                                    .insert(abs_path.clone(), (current_size, current_mtime_raw));
                                changed.push(abs_path.clone());
                            }
                            None => {
                                // Couldn't read/hash — fall back to the safe
                                // default of reprocessing.
                                if let Some(&id) = db_ids.get(rel.as_str()) {
                                    existing_ids.insert(abs_path.clone(), id);
                                }
                                disk_stats
                                    .insert(abs_path.clone(), (current_size, current_mtime_raw));
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
                            if let Some(&id) = db_ids.get(rel.as_str()) {
                                existing_ids.insert(abs_path.clone(), id);
                            }
                            disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
                            changed.push(abs_path.clone());
                        }
                    }
                }
                // else: mtime matches → file unchanged, skip (not in disk_stats)
            } else {
                // Can't stat the file → process it anyway (safety)
                if let Some(&id) = db_ids.get(rel.as_str()) {
                    existing_ids.insert(abs_path.clone(), id);
                }
                // disk_stats entry has (0, 0.0) — parse_one_batch fallback handles it
                disk_stats.insert(abs_path.clone(), (0, 0.0));
                changed.push(abs_path.clone());
            }
        } else {
            // New file not in DB — always changed; stash a hash now so
            // future runs have something to compare against.
            if let Some(new_hash) = hash_file_contents(abs_path) {
                new_hashes.insert(abs_path.clone(), new_hash);
            }
            disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
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
        existing_ids,
        disk_stats,
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
        let diff = compute_diff(&[], &files, tmp.path(), 0.01, None, None);

        assert_eq!(diff.changed_count(), 2);
        assert_eq!(diff.removed_count(), 0);
    }

    #[test]
    fn test_unchanged_files_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let mtime = file_mtime(&f1).unwrap();

        let db = vec![DbFileEntry {
            id: 1,
            path: "a.py".into(),
            mtime,
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
        assert!(diff.changed.is_empty(), "unchanged file should be skipped");
    }

    #[test]
    fn test_mtime_change_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let old_mtime = 0.0; // epoch — clearly different

        let db = vec![DbFileEntry {
            id: 42,
            path: "a.py".into(),
            mtime: old_mtime,
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
        assert_eq!(
            diff.changed_count(),
            1,
            "changed mtime should trigger re-process"
        );
        assert_eq!(
            diff.existing_ids.get(&f1).copied(),
            Some(42),
            "existing_ids must carry the DB id for a changed file that was already in the DB"
        );
        let (size, _mtime) = diff
            .disk_stats
            .get(&f1)
            .copied()
            .expect("disk_stats must contain a changed file");
        assert!(
            size > 0,
            "file size in disk_stats must be non-zero for a real file"
        );
    }

    #[test]
    fn test_disk_stats_populated_for_new_file() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "new.py");

        // Empty DB — file has never been indexed.
        let diff = compute_diff(&[], std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
        assert_eq!(diff.changed_count(), 1);
        let (size, mtime) = diff
            .disk_stats
            .get(&f1)
            .copied()
            .expect("new file must appear in disk_stats so parse_one_batch can skip re-stat");
        assert!(size > 0, "size must be non-zero for a real file");
        assert!(mtime > 0.0, "mtime must be non-zero for a real file");
    }

    #[test]
    fn test_disk_stats_absent_for_unchanged_file() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let mtime = file_mtime(&f1).unwrap();

        let db = vec![DbFileEntry {
            id: 1,
            path: "a.py".into(),
            mtime,
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
        assert!(diff.changed.is_empty(), "unchanged file should be skipped");
        assert!(
            diff.disk_stats.is_empty(),
            "unchanged files must not appear in disk_stats — they never reach parse_one_batch"
        );
    }

    #[test]
    fn test_precomputed_stats_values_appear_in_disk_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        // DB has epoch mtime → file appears changed regardless of disk mtime.
        let db = vec![DbFileEntry {
            id: 1,
            path: "a.py".into(),
            mtime: 0.0,
            content_hash: None,
        }];

        // Pass sentinel values via precomputed_stats — these must flow through
        // into disk_stats unchanged, proving that no second stat call is made.
        let mut pre = HashMap::new();
        pre.insert(f1.clone(), (99_999u64, 42.0f64));
        let diff = compute_diff(
            &db,
            std::slice::from_ref(&f1),
            tmp.path(),
            0.01,
            Some(&pre),
            None,
        );
        assert_eq!(diff.changed_count(), 1);
        let (size, mtime) = diff
            .disk_stats
            .get(&f1)
            .copied()
            .expect("changed file must appear in disk_stats");
        assert_eq!(
            size, 99_999,
            "size must be the sentinel from precomputed_stats"
        );
        assert_eq!(
            mtime, 42.0,
            "mtime must be the sentinel from precomputed_stats"
        );
    }

    #[test]
    fn test_removed_files_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        let db = vec![
            DbFileEntry {
                id: 1,
                path: "a.py".into(),
                mtime: file_mtime(&f1).unwrap(),
                content_hash: None,
            },
            DbFileEntry {
                id: 2,
                path: "gone.py".into(), // this file doesn't exist on disk
                mtime: 1.0,
                content_hash: None,
            },
        ];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
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
            id: 1,
            path: "a.py".into(),
            mtime: 0.0, // clearly different from the file's real mtime
            content_hash: Some(hash),
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
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
            id: 1,
            path: "a.py".into(),
            mtime: 0.0,
            content_hash: Some("deadbeefdeadbeef".into()),
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);
        assert_eq!(
            diff.changed_count(),
            1,
            "hash mismatch should still reprocess"
        );
        assert_eq!(diff.skipped_by_hash, 0);
        assert!(diff.new_hashes.contains_key(&f1));
    }

    #[test]
    fn test_compute_diff_no_id_for_new_files() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "new.py");

        // Empty DB — the file has never been indexed.
        let diff = compute_diff(&[], std::slice::from_ref(&f1), tmp.path(), 0.01, None, None);

        assert_eq!(diff.changed_count(), 1);
        assert!(
            diff.existing_ids.is_empty(),
            "new files (not in DB) must not appear in existing_ids"
        );
    }

    fn create_file(dir: &tempfile::TempDir, name: &str) -> PathBuf {
        let path = dir.path().join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# test file").unwrap();
        path
    }
}
