//! Process-interruption-safe replacement under the caller's exclusive lease.
//!
//! The marker gates ordinary opens, not power-loss durability. Abandoned
//! quarantines are deliberately never restored or swept by later installers.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use super::{
    error::LeaseError,
    lease::{canonical_target, metadata_optional, sibling, Backend, Lease},
};

const DUCKDB_SUFFIXES: &[&str] = &[
    "",
    ".wal",
    ".wal.corrupt",
    ".compact_backup",
    ".compact_backup.wal",
    ".compact_new",
    ".compact_new.wal",
    ".old",
    ".old.wal",
    ".compact",
    ".compact.wal",
    ".swap_intent",
    ".export_tmp",
    ".root.json",
    ".embedding.json",
];
const LANCEDB_SUFFIXES: &[&str] = &["", ".embedding.json"];
const MARKER: &[u8] = b"{\"version\":1}\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum InstallPoint {
    CandidatePreflightRename,
    MarkerCreate,
    MarkerPublished,
    Isolate(usize),
    Isolated(usize),
    CandidateRename,
    CandidateRenamed,
    MarkerRemove,
    Committed,
    QuarantineCleanup,
}

fn new_quarantine(target: &Path) -> Result<PathBuf, LeaseError> {
    for _ in 0..16 {
        let path = sibling(
            target,
            &format!(".chunkhound-quarantine-{:032x}", rand::random::<u128>()),
        );
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(LeaseError::Filesystem(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "quarantine name exhausted",
    )))
}

pub(crate) fn install(
    lease: &Lease,
    backend: Backend,
    candidate: &Path,
) -> Result<Vec<String>, LeaseError> {
    install_with(lease, backend, candidate, &mut |_| Ok(()))
}

pub(super) fn install_with(
    lease: &Lease,
    backend: Backend,
    candidate: &Path,
    boundary: &mut impl FnMut(InstallPoint) -> io::Result<()>,
) -> Result<Vec<String>, LeaseError> {
    let target = lease.target();
    backend.validate_shape(target)?;
    // Validate the unresolved leaf before canonicalization can follow a symlink.
    backend.validate_shape(candidate)?;
    let candidate = canonical_target(candidate)?;
    if metadata_optional(&candidate)?.is_none() {
        return Err(LeaseError::Layout("candidate is absent"));
    }
    if candidate.file_name() != Some(std::ffi::OsStr::new(backend.filename()))
        || candidate == target
        || candidate.starts_with(target)
    {
        return Err(LeaseError::Layout(
            "candidate must be a separate private database",
        ));
    }
    let marker = sibling(target, ".chunkhound-install.json");
    if let Some(metadata) = metadata_optional(&marker)? {
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(LeaseError::Layout(
                "installation marker is not a regular file",
            ));
        }
    }
    let quarantine = new_quarantine(target)?;
    let incoming = quarantine.join("candidate");
    // Actual rename proves same-filesystem placement before touching old inputs.
    if let Err(error) = boundary(InstallPoint::CandidatePreflightRename)
        .and_then(|()| fs::rename(&candidate, &incoming))
    {
        let _ = fs::remove_dir_all(&quarantine);
        return Err(error.into());
    }
    let transaction: Result<(), LeaseError> = (|| {
        if metadata_optional(&marker)?.is_none() {
            boundary(InstallPoint::MarkerCreate)?;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&marker)?;
            file.write_all(MARKER)?;
            file.flush()?;
            drop(file);
        }
        // Existing incomplete marker stays until commit; its contents are not paths.
        boundary(InstallPoint::MarkerPublished)?;
        let suffixes = match backend {
            Backend::DuckDb => DUCKDB_SUFFIXES,
            Backend::LanceDb => LANCEDB_SUFFIXES,
        };
        for (index, suffix) in suffixes.iter().enumerate() {
            let old = sibling(target, suffix);
            if metadata_optional(&old)?.is_some() {
                let filename = old
                    .file_name()
                    .ok_or(LeaseError::Layout("associated path has no filename"))?;
                boundary(InstallPoint::Isolate(index))?;
                fs::rename(&old, quarantine.join(filename))?;
                boundary(InstallPoint::Isolated(index))?;
            }
        }
        boundary(InstallPoint::CandidateRename)?;
        fs::rename(&incoming, target)?;
        boundary(InstallPoint::CandidateRenamed)?;
        boundary(InstallPoint::MarkerRemove)?;
        fs::remove_file(&marker)?; // Process-level commit point.
        Ok(())
    })();
    if let Err(error) = transaction {
        // An unreadable/ambiguous marker is refusal, not permission to index locally.
        match metadata_optional(&marker) {
            Ok(None) => {
                let _ = fs::remove_dir_all(&quarantine);
                return Err(error);
            }
            Ok(Some(_)) | Err(_) => return Err(LeaseError::Interrupted),
        }
    }
    // Post-commit cleanup is limited to this installer-owned quarantine and warns only.
    let cleanup = boundary(InstallPoint::Committed)
        .and_then(|()| boundary(InstallPoint::QuarantineCleanup))
        .and_then(|()| fs::remove_dir_all(&quarantine));
    if cleanup.is_err() {
        return Ok(vec!["quarantine_cleanup_failed".into()]);
    }
    Ok(Vec::new())
}

#[cfg(test)]
#[path = "install_tests.rs"]
mod tests;
