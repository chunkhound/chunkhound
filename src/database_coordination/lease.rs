use std::fs::{self, File, Metadata, OpenOptions};
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use super::error::LeaseError;

pub(crate) fn sibling(target: &Path, suffix: &str) -> PathBuf {
    let mut path = target.as_os_str().to_os_string();
    path.push(suffix);
    PathBuf::from(path)
}

pub(crate) fn metadata_optional(path: &Path) -> Result<Option<Metadata>, LeaseError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

/// Resolve existing symlinks before parent traversal, including for absent targets.
pub(crate) fn canonical_target(path: &Path) -> Result<PathBuf, LeaseError> {
    if path == Path::new(":memory:") {
        return Ok(path.to_path_buf());
    }
    if path.as_os_str().is_empty() {
        return Err(LeaseError::Layout("empty target"));
    }
    let resolved = resolve_path(path)?;
    if resolved.file_name().is_none() {
        return Err(LeaseError::Layout("target must have a filename"));
    }
    Ok(resolved)
}

/// Share absent-parent and symlink-before-`..` semantics for identities and shape checks.
fn resolve_path(path: &Path) -> Result<PathBuf, LeaseError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut resolved = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir => resolved.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                if metadata_optional(&resolved)?.is_some() {
                    resolved = fs::canonicalize(&resolved)?;
                }
            }
        }
    }
    Ok(resolved)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Backend {
    DuckDb,
    LanceDb,
}

impl Backend {
    pub(crate) fn parse(value: &str) -> Result<Self, LeaseError> {
        match value {
            "duckdb" => Ok(Self::DuckDb),
            "lancedb" => Ok(Self::LanceDb),
            _ => Err(LeaseError::Layout("unknown backend")),
        }
    }

    pub(crate) fn filename(self) -> &'static str {
        match self {
            Self::DuckDb => "chunks.db",
            Self::LanceDb => "lancedb.lancedb",
        }
    }

    pub(crate) fn validate_shape(self, path: &Path) -> Result<(), LeaseError> {
        // Strip terminal directory syntax, then resolve only the parent as for identity.
        // The final entry must stay unresolved so metadata can reject a leaf symlink.
        let path: PathBuf = path.components().collect();
        let leaf = path
            .file_name()
            .ok_or(LeaseError::Layout("target must have a filename"))?;
        let parent = path
            .parent()
            .ok_or(LeaseError::Layout("target has no parent"))?;
        let unresolved_leaf = resolve_path(parent)?.join(leaf);
        if let Some(metadata) = metadata_optional(&unresolved_leaf)? {
            if metadata.file_type().is_symlink()
                || match self {
                    Self::DuckDb => !metadata.is_file(),
                    Self::LanceDb => !metadata.is_dir(),
                }
            {
                return Err(LeaseError::Layout(
                    "target or candidate has unsupported shape",
                ));
            }
        }
        Ok(())
    }
}

/// Each owner holds an independently opened descriptor; dropping one never unlocks another.
pub(crate) struct Lease {
    target: PathBuf,
    _file: Option<File>,
}

impl Lease {
    pub(crate) fn target(&self) -> &Path {
        &self.target
    }

    pub(crate) fn assert_covers(&self, path: &Path) -> Result<(), LeaseError> {
        if canonical_target(path)?.as_path() != self.target() {
            return Err(LeaseError::Layout("lease covers a different target"));
        }
        Ok(())
    }

    pub(crate) fn interrupted(&self) -> Result<bool, LeaseError> {
        if self.target == Path::new(":memory:") {
            return Ok(false);
        }
        Ok(metadata_optional(&sibling(&self.target, ".chunkhound-install.json"))?.is_some())
    }
}

fn open_coordination_file(path: &Path) -> Result<File, LeaseError> {
    match File::open(path) {
        Ok(file) => return Ok(file),
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(File::open(path)?),
        Err(error) => Err(error.into()),
    }
}

fn acquire(path: &Path, exclusive: bool) -> Result<Lease, LeaseError> {
    let mut target = canonical_target(path)?;
    if target == Path::new(":memory:") {
        if exclusive {
            return Err(LeaseError::Layout("in-memory installation"));
        }
        return Ok(Lease {
            target,
            _file: None,
        });
    }
    let parent = target
        .parent()
        .ok_or(LeaseError::Layout("target has no parent"))?;
    fs::create_dir_all(parent)?;
    target = canonical_target(&target)?;
    let file = open_coordination_file(&sibling(&target, ".chunkhound.lock"))?;
    let locked = if exclusive {
        fs4::FileExt::try_lock(&file)
    } else {
        fs4::FileExt::try_lock_shared(&file)
    };
    match locked {
        Ok(()) => {}
        Err(fs4::TryLockError::WouldBlock) => return Err(LeaseError::Busy),
        Err(fs4::TryLockError::Error(error)) => return Err(error.into()),
    }
    let lease = Lease {
        target,
        _file: Some(file),
    };
    if !exclusive && lease.interrupted()? {
        return Err(LeaseError::Interrupted);
    }
    Ok(lease)
}

pub(crate) fn acquire_shared(path: &Path) -> Result<Lease, LeaseError> {
    acquire(path, false)
}

pub(crate) fn acquire_exclusive(path: &Path, backend: Backend) -> Result<Lease, LeaseError> {
    if path.file_name() != Some(std::ffi::OsStr::new(backend.filename())) {
        return Err(LeaseError::Layout(
            "installation requires directory-layout target",
        ));
    }
    // Reject the unresolved leaf symlink while allowing symlinked parents.
    backend.validate_shape(path)?;
    let lease = acquire(path, true)?;
    // An ordinary cooperative opener could have raced acquisition.
    backend.validate_shape(path)?;
    lease.assert_covers(path)?;
    // A surviving marker intentionally permits explicit recovery ownership.
    Ok(lease)
}
