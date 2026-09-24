//! Identity computation for `privacy_mode`. Deliberately a single function
//! used for BOTH the event payload's `user` field and the S3 object key's
//! identity segment (`AnalyticsRecorder::object_key`) — the wiki's rev-5 fix:
//! computing this in two places let them drift, since the key segment used
//! to always be the raw OS username regardless of `privacy_mode`.

use rand::RngCore;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// `None` means "no identity in the payload" (`anonymous` mode, or an
/// unrecognized mode treated the same way — never panics on bad config).
pub(crate) fn payload_identity(
    privacy_mode: &str,
    os_username: &str,
    salt_path: &Path,
) -> Option<String> {
    match privacy_mode {
        "full" => Some(os_username.to_string()),
        "hashed" => Some(hashed_identity(os_username, salt_path)),
        _ => None,
    }
}

/// S3 keys can't have a null path component, and a per-install random ID
/// would just reintroduce pseudonymous tracking under a different name — so
/// `anonymous` mode collapses to one fixed, shared literal segment. Every
/// other mode reuses the exact same value computed for the payload.
pub(crate) fn object_key_segment(privacy_mode: &str, payload_identity: &Option<String>) -> String {
    match privacy_mode {
        "anonymous" => "anonymous".to_string(),
        _ => payload_identity
            .clone()
            .unwrap_or_else(|| "anonymous".to_string()),
    }
}

fn hashed_identity(os_username: &str, salt_path: &Path) -> String {
    let salt = load_or_create_salt(salt_path);
    let mut hasher = Sha256::new();
    hasher.update(salt.as_bytes());
    hasher.update(os_username.as_bytes());
    hex_encode(&hasher.finalize())
}

/// Per-install random salt, generated once and persisted locally — never
/// uploaded, logged, or transmitted. Deliberately not a shared org-wide
/// salt: anyone holding a shared salt could precompute a rainbow table
/// against a directory of company usernames, fully reversing every hash.
/// For that same reason the file is created (and, if found wider, tightened)
/// to `0600` on Unix — a world-readable salt lets anyone else on the box
/// read it and do exactly that.
fn load_or_create_salt(path: &Path) -> String {
    if let Ok(existing) = fs::read_to_string(path) {
        let trimmed = existing.trim();
        if !trimmed.is_empty() {
            tighten_permissions(path);
            return trimmed.to_string();
        }
    }
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    let hex = hex_encode(&bytes);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    // Best-effort persist; if this fails, the salt is still usable for the
    // current process, it just won't be stable across restarts.
    write_salt_file(path, &hex);
    hex
}

#[cfg(unix)]
fn write_salt_file(path: &Path, hex: &str) {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let result = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .and_then(|mut f| f.write_all(hex.as_bytes()));
    if let Err(e) = result {
        log::warn!("analytics: failed to persist salt file: {e}");
    }
}

#[cfg(not(unix))]
fn write_salt_file(path: &Path, hex: &str) {
    let _ = fs::write(path, hex);
}

/// Best-effort narrow-down for a salt file that predates this `0600`
/// enforcement (or was recreated by something else). Never widens
/// permissions, never errors the caller — the salt is still usable either
/// way, this is defense-in-depth, not a correctness requirement.
#[cfg(unix)]
fn tighten_permissions(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    if let Ok(metadata) = fs::metadata(path) {
        if metadata.permissions().mode() & 0o777 != 0o600 {
            let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
        }
    }
}

#[cfg(not(unix))]
fn tighten_permissions(_path: &Path) {}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn full_mode_passes_username_through() {
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        assert_eq!(
            payload_identity("full", "jsmith", &salt_path),
            Some("jsmith".to_string())
        );
    }

    #[test]
    fn anonymous_mode_has_no_payload_identity() {
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        assert_eq!(payload_identity("anonymous", "jsmith", &salt_path), None);
    }

    #[test]
    fn hashed_mode_is_stable_across_calls_with_same_salt_file() {
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        let first = payload_identity("hashed", "jsmith", &salt_path).unwrap();
        let second = payload_identity("hashed", "jsmith", &salt_path).unwrap();
        assert_eq!(first, second);
        assert_ne!(first, "jsmith");
    }

    #[test]
    fn hashed_mode_changes_if_salt_file_is_removed() {
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        let first = payload_identity("hashed", "jsmith", &salt_path).unwrap();
        fs::remove_file(&salt_path).unwrap();
        let second = payload_identity("hashed", "jsmith", &salt_path).unwrap();
        assert_ne!(first, second);
    }

    #[cfg(unix)]
    #[test]
    fn salt_file_is_created_with_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        payload_identity("hashed", "jsmith", &salt_path).unwrap();

        let mode = fs::metadata(&salt_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "salt file must not be group/world readable");
    }

    #[cfg(unix)]
    #[test]
    fn salt_file_permissions_are_tightened_if_found_wider() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        fs::write(&salt_path, "0123456789abcdef0123456789abcdef").unwrap();
        fs::set_permissions(&salt_path, fs::Permissions::from_mode(0o644)).unwrap();

        payload_identity("hashed", "jsmith", &salt_path).unwrap();

        let mode = fs::metadata(&salt_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(
            mode, 0o600,
            "a pre-existing wide-open salt must be tightened"
        );
    }

    #[test]
    fn object_key_segment_matches_payload_identity_for_full_and_hashed() {
        // This is the direct rev-5 regression guard: the key segment and the
        // payload's user field must always be computed from the same value.
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        for mode in ["full", "hashed"] {
            let identity = payload_identity(mode, "jsmith", &salt_path);
            let key_segment = object_key_segment(mode, &identity);
            assert_eq!(Some(key_segment), identity);
        }
    }

    #[test]
    fn object_key_segment_is_fixed_literal_for_anonymous() {
        let dir = tempdir().unwrap();
        let salt_path = dir.path().join("salt");
        let identity = payload_identity("anonymous", "jsmith", &salt_path);
        assert_eq!(identity, None);
        assert_eq!(object_key_segment("anonymous", &identity), "anonymous");
    }
}
