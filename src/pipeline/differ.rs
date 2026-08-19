//! File diff engine — compares filesystem state against the DB snapshot
//! to produce a minimal set of files that need re-processing.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::types::DbFileEntry;

/// Result of diffing the filesystem against the DB.
#[derive(Debug, Default)]
pub(crate) struct DiffResult {
    /// Files that are new or whose mtime has changed — must be re-parsed.
    pub changed: Vec<PathBuf>,
    /// DB file paths that no longer exist on disk — need to be deleted.
    pub removed: Vec<String>,
    /// Files whose mtime differed from the DB but whose content hash matched
    /// the DB's stored hash — confirmed unchanged, never entered `changed`.
    pub skipped_by_hash: u64,
    /// Content hash for every file scanned that already had a DB row (keyed
    /// by absolute path) — a freshly-computed hash for a file in `changed`,
    /// or the DB's already-stored hash reused as-is for a file left
    /// unchanged (mtime matched, or hash-confirmed identical despite a
    /// differing mtime). Covers every scanned file, not just `changed`, so a
    /// force-reindex caller — which reprocesses every file regardless of
    /// this diff's outcome — can still write back the correct hash for a
    /// file it didn't need to change, instead of nulling it out.
    pub new_hashes: HashMap<PathBuf, String>,
    /// DB row id for every file scanned that already existed in the DB
    /// (keyed by absolute path), whether or not it ended up in `changed`.
    /// New files (first index) are absent here. Lets the write phase skip
    /// the per-file SELECT in upsert_file.
    pub existing_ids: HashMap<PathBuf, i64>,
    /// (size_bytes, mtime) for every file scanned, keyed by absolute path —
    /// covers `changed` files and unchanged/hash-confirmed ones alike.
    /// Populated by the diff phase so the parse stage can skip re-stat-ing
    /// files whose metadata was already read here.
    pub disk_stats: HashMap<PathBuf, (u64, f64)>,
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
/// `rel_keys` maps each absolute path to its canonical project-relative
/// DB/lookup key, computed once by Python's `get_relative_path_safe()` — the
/// single symlink-aware source of truth (git-worktree support: a symlink's
/// logical path is preserved even when its target resolves outside
/// project_root). This function used to re-derive the key itself via a naive
/// `strip_prefix(project_root)`, which diverged from Python's DB-write path
/// for symlinked files (a different or missing key meant the file's DB row
/// could be misclassified as `removed` and deleted while the file itself was
/// simultaneously processed as `changed`) — see the fix that replaced that
/// re-derivation with this caller-supplied map.
/// `mtime_epsilon` controls how close two timestamps must be to be considered equal.
/// `precomputed_stats`, if provided, is a map of absolute path → (size_bytes, mtime)
/// produced by a prior stat pass (e.g. the TZ-offset pass in `compute_diff_blocking`).
/// When present, `compute_diff` reuses those values instead of calling `stat` again.
/// `on_tick`, if provided, is called periodically with `(files_scanned, total)`.
pub(crate) fn compute_diff(
    db_file_entries: &[DbFileEntry],
    files_on_disk: &[PathBuf],
    rel_keys: &HashMap<PathBuf, String>,
    mtime_epsilon: f64,
    precomputed_stats: Option<&HashMap<PathBuf, (u64, f64)>>,
    mut on_tick: Option<&mut dyn FnMut(usize, usize)>,
) -> DiffResult {
    // Build a lookup: DB path → stored mtime (None if the column is NULL).
    let db_map: HashMap<&str, Option<f64>> = db_file_entries
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

        // Relative key supplied by the caller (see `rel_keys` doc above).
        let rel = match rel_keys.get(abs_path) {
            Some(r) => r.clone(),
            None => {
                // Should be unreachable: Python guarantees a relative key
                // for every path it sends. Treat defensively — an internal
                // contract violation, not a normal runtime condition — and
                // log so it's visible if it ever fires.
                log::warn!("no relative key provided for {abs_path:?}; processing anyway");
                disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
                changed.push(abs_path.clone());
                continue;
            }
        };

        disk_paths.insert(rel.clone());

        if let Some(stored_mtime) = db_map.get(rel.as_str()).copied() {
            let db_mtime = match stored_mtime {
                Some(m) => m,
                None => {
                    // NULL modified_time is not "unchanged" — reprocess, but
                    // keep the row id so upsert updates in place. The path
                    // stays in db_paths for orphan detection if the file is
                    // gone from disk.
                    if let Some(new_hash) = hash_file_contents(abs_path) {
                        new_hashes.insert(abs_path.clone(), new_hash);
                    }
                    if let Some(&id) = db_ids.get(rel.as_str()) {
                        existing_ids.insert(abs_path.clone(), id);
                    }
                    disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
                    changed.push(abs_path.clone());
                    continue;
                }
            };
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
                                // the parse/embed/store pipeline. Still
                                // stash the (already-computed) hash/stat/id
                                // so a force-reindex caller — which
                                // reprocesses this file regardless — writes
                                // back the same hash instead of nulling it.
                                skipped_by_hash += 1;
                                new_hashes.insert(abs_path.clone(), new_hash);
                                if let Some(&id) = db_ids.get(rel.as_str()) {
                                    existing_ids.insert(abs_path.clone(), id);
                                }
                                disk_stats
                                    .insert(abs_path.clone(), (current_size, current_mtime_raw));
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
                } else {
                    // mtime matches → skip reprocessing, same trust the
                    // mtime-only fast path above already relies on (not a
                    // new assumption: a false match from clock skew or an
                    // mtime-preserving restore tool is already possible
                    // here regardless of this branch, and self-corrects the
                    // next time the mtime genuinely differs). Given that
                    // trust, no extra read/hash is needed — mirror the DB's
                    // already-stored hash/id/stat verbatim, so a
                    // force-reindex caller (which reprocesses this file
                    // regardless) writes back the same values instead of
                    // nulling them out.
                    if let Some(&stored_hash) = db_hashes.get(rel.as_str()) {
                        new_hashes.insert(abs_path.clone(), stored_hash.to_string());
                    }
                    if let Some(&id) = db_ids.get(rel.as_str()) {
                        existing_ids.insert(abs_path.clone(), id);
                    }
                    disk_stats.insert(abs_path.clone(), (current_size, current_mtime_raw));
                }
            } else {
                // Can't stat the file → process it anyway (safety). No
                // new_hashes entry, so a force-reindex would write NULL for
                // this file's content_hash — accepted, not fixed: a stat()
                // failure moments after the scanner found the file almost
                // always means a concurrent delete, which will also fail
                // the read-for-parse and drop the row entirely rather than
                // reach the write path with a null hash.
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
        skipped_by_hash,
        new_hashes,
        existing_ids,
        disk_stats,
    }
}

/// Read the mtime of a file as a Unix timestamp (seconds).
///
/// Only used by tests today (production code gets mtime from `precomputed_stats`
/// or the `std::fs::metadata()` fallback inlined in `compute_diff`).
#[cfg(test)]
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

/// Test-only convenience accessors — production code reads `changed`/`removed` directly.
#[cfg(test)]
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

    /// Build a `rel_keys` map for a single (path, key) pair — the common case
    /// in these tests, where the on-disk path's DB key is just its filename.
    fn rel_key_for(path: &Path, key: &str) -> HashMap<PathBuf, String> {
        [(path.to_path_buf(), key.to_string())]
            .into_iter()
            .collect()
    }

    #[test]
    fn test_empty_db_all_new() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let f2 = create_file(&tmp, "b.rs");

        let files = vec![f1.clone(), f2.clone()];
        let rel_keys: HashMap<PathBuf, String> = [
            (f1.clone(), "a.py".to_string()),
            (f2.clone(), "b.rs".to_string()),
        ]
        .into_iter()
        .collect();
        let diff = compute_diff(&[], &files, &rel_keys, 0.01, None, None);

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
            mtime: Some(mtime),
            content_hash: None,
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
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
            mtime: Some(old_mtime),
            content_hash: None,
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
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
        let rel_keys = rel_key_for(&f1, "new.py");
        let diff = compute_diff(&[], std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
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
    fn test_unchanged_file_still_populates_side_maps() {
        // Skipped from `changed` (nothing to reprocess), but a force-reindex
        // caller reprocesses every file regardless of `changed` — it needs
        // disk_stats/new_hashes/existing_ids for this file too, or it writes
        // back a null content_hash for a file that never actually changed.
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let mtime = file_mtime(&f1).unwrap();
        let stored_hash = hash_file_contents(&f1).unwrap();

        let db = vec![DbFileEntry {
            id: 7,
            path: "a.py".into(),
            mtime: Some(mtime),
            content_hash: Some(stored_hash.clone()),
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
        assert!(diff.changed.is_empty(), "unchanged file should be skipped");
        assert_eq!(
            diff.new_hashes.get(&f1),
            Some(&stored_hash),
            "unchanged file must carry its existing DB hash forward, not a null/empty one"
        );
        assert_eq!(
            diff.existing_ids.get(&f1).copied(),
            Some(7),
            "unchanged file must still carry its DB row id forward"
        );
        assert!(
            diff.disk_stats.contains_key(&f1),
            "unchanged file must still appear in disk_stats so a force-reindex \
             caller (which reprocesses it regardless) can skip re-stat-ing it"
        );
    }

    #[test]
    fn test_hash_confirmed_unchanged_still_populates_side_maps() {
        // Mirrors the branch above but for the "mtime bumped, hash still
        // matches" case — the hash was already computed to make that call,
        // so stashing it is free, not an extra read.
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let hash = hash_file_contents(&f1).unwrap();

        let db = vec![DbFileEntry {
            id: 9,
            path: "a.py".into(),
            mtime: Some(0.0), // clearly different from the file's real mtime
            content_hash: Some(hash.clone()),
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
        assert_eq!(diff.skipped_by_hash, 1);
        assert!(diff.changed.is_empty());
        assert_eq!(diff.new_hashes.get(&f1), Some(&hash));
        assert_eq!(diff.existing_ids.get(&f1).copied(), Some(9));
        assert!(diff.disk_stats.contains_key(&f1));
    }

    #[test]
    fn test_precomputed_stats_values_appear_in_disk_stats() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        // DB has epoch mtime → file appears changed regardless of disk mtime.
        let db = vec![DbFileEntry {
            id: 1,
            path: "a.py".into(),
            mtime: Some(0.0),
            content_hash: None,
        }];

        // Pass sentinel values via precomputed_stats — these must flow through
        // into disk_stats unchanged, proving that no second stat call is made.
        let mut pre = HashMap::new();
        pre.insert(f1.clone(), (99_999u64, 42.0f64));
        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(
            &db,
            std::slice::from_ref(&f1),
            &rel_keys,
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
                mtime: file_mtime(&f1),
                content_hash: None,
            },
            DbFileEntry {
                id: 2,
                path: "gone.py".into(), // this file doesn't exist on disk
                mtime: Some(1.0),
                content_hash: None,
            },
        ];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
        assert_eq!(diff.changed_count(), 0); // a.py unchanged
        assert_eq!(diff.removed_count(), 1);
        assert!(diff.removed.contains(&"gone.py".to_string()));
    }

    #[test]
    fn test_relative_key_from_caller_map_used_verbatim() {
        // Regression test for the symlink path-key divergence bug: the
        // relative key used to be re-derived here via
        // `abs_path.strip_prefix(project_root)`, which silently failed (or
        // produced the wrong key) whenever the caller's absolute path wasn't
        // textually under a single "project root" — exactly what happens for
        // a symlink whose target resolves outside project_root (Python's
        // get_relative_path_safe() still gives it a valid *logical* relative
        // key in that case). Prove `compute_diff` now trusts the caller's
        // `rel_keys` map verbatim, independent of the absolute path's own
        // structure: an absolute path with no plausible "project root" at
        // all still matches its DB row correctly via the supplied key.
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");
        let mtime = file_mtime(&f1).unwrap();

        // A path a naive strip_prefix could never relativize sensibly (it
        // shares no meaningful root with the file's own directory), mapped
        // to the same logical key the DB row uses.
        let elsewhere_abs = PathBuf::from("/completely/unrelated/tree/a.py");
        let rel_keys: HashMap<PathBuf, String> = [
            (f1.clone(), "a.py".to_string()),
            (elsewhere_abs, "a.py".to_string()),
        ]
        .into_iter()
        .collect();

        let db = vec![DbFileEntry {
            id: 1,
            path: "a.py".into(),
            mtime: Some(mtime),
            content_hash: None,
        }];

        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
        assert!(
            diff.changed.is_empty(),
            "file must match its DB row via the supplied key, not be reprocessed"
        );
        assert_eq!(
            diff.removed_count(),
            0,
            "file present via rel_keys must not be misclassified as removed"
        );
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
            mtime: Some(0.0), // clearly different from the file's real mtime
            content_hash: Some(hash),
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
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
            mtime: Some(0.0),
            content_hash: Some("deadbeefdeadbeef".into()),
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
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
        let rel_keys = rel_key_for(&f1, "new.py");
        let diff = compute_diff(&[], std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);

        assert_eq!(diff.changed_count(), 1);
        assert!(
            diff.existing_ids.is_empty(),
            "new files (not in DB) must not appear in existing_ids"
        );
    }

    #[test]
    fn test_null_mtime_on_disk_is_reprocessed_not_treated_as_new() {
        let tmp = tempfile::tempdir().unwrap();
        let f1 = create_file(&tmp, "a.py");

        let db = vec![DbFileEntry {
            id: 11,
            path: "a.py".into(),
            mtime: None,
            content_hash: None,
        }];

        let rel_keys = rel_key_for(&f1, "a.py");
        let diff = compute_diff(&db, std::slice::from_ref(&f1), &rel_keys, 0.01, None, None);
        assert_eq!(
            diff.changed_count(),
            1,
            "NULL DB mtime is not proof of unchanged content"
        );
        assert_eq!(
            diff.existing_ids.get(&f1).copied(),
            Some(11),
            "must keep the existing row id so upsert updates in place"
        );
        assert_eq!(diff.removed_count(), 0);
    }

    #[test]
    fn test_null_mtime_missing_from_disk_is_removed() {
        let db = vec![DbFileEntry {
            id: 12,
            path: "gone.py".into(),
            mtime: None,
            content_hash: None,
        }];

        let diff = compute_diff(&db, &[], &HashMap::new(), 0.01, None, None);
        assert_eq!(diff.changed_count(), 0);
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
