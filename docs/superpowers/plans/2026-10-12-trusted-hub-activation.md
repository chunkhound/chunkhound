# Trusted Hub Activation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver trusted-hub pull, searchable installed bases and local overlay while excluding installation during current-version database use and refusing interrupted mixed state.

**Architecture:** Retain the completed Python manifest/HTTP/archive/inspection foundation. One Rust advisory-lease implementation protects Python and native database lifetimes; a scoped exclusive native handle owns marker/quarantine installation. Python retains selection, backend knowledge, CLI policy, automatic recheck and cancellation-safe orchestration.

**Tech Stack:** Python >=3.10,<3.15, Rust 2021/MSRV 1.82, PyO3 0.22, `fs4 = "=1.1.0"` (sync only), existing DuckDB/LanceDB, asyncio, pytest and uv. No new Python dependency or native LanceDB backend.

## Global Constraints

Source of truth: `docs/superpowers/specs/2026-10-12-trusted-hub-activation-design.md`, approved by the user. This successor replaces refusal-only Tasks 8–9 of `2026-10-11-central-indexing-hub-client.md`; completed foundation Tasks 1–7 are not repeated.

- “Hubs are private, organization-operated and trusted.” Preserve checksum, bounded archive, format/embedding, TLS and sanitized diagnostic checks; do not restart adversarial-publisher audit closure.
- “No publisher, permanent artifact cache, authentication service, new diff engine, generation-pointer storage layout, or native LanceDB backend is introduced.”
- “No custom unsafe platform layer.” Preserve `#![forbid(unsafe_code)]`; use owned paths across PyO3 `allow_threads`; no boundary unwraps.
- “Shared here means 'installation exclusion', not permission for concurrent backend writers.”
- “Do not roll back or discard another live provider's transaction to acquire a lease.”
- “Read-only holders participate as well.” No unlocked fallback; `CHUNKHOUND_USE_RUST=0` still acquires native leases.
- “Download/staging happens without exclusive ownership of the target.” Candidate handles close before installation; same-filesystem rename only.
- “Inspection under that lease must not try to acquire a nested shared lease.” Borrow explicit ownership for the automatic recheck; no lock upgrade.
- “Async cancellation must not drop leases or staging directories while native filesystem/database work is still running.” Shield and drain actual workers, including repeated cancellation.
- “A downloaded database contradicting its manifest is fatal for both modes; the old target and its sidecars remain untouched.” Surviving/new interruption markers are fatal for ordinary opens and automatic indexing.
- “Quarantine cleanup is best-effort and limited to installer-owned paths. Cleanup failure after commit is a warning, not a failed pull or a reason to restore stale sidecars.” Apply the success rule to postcommit staging cleanup too.
- “Simulation, ignore checking and no-embeddings indexing do not pull. MCP and search never initiate a pull.” Aligned local state makes zero hub requests.
- `chunkhound pull [path] --version latest|-1|-2`; explicit inability exits 1; no version walking. Hook automatic pull before registry initialization/drift pinning.
- “Power-loss durability across platforms/filesystems is not guaranteed by a marker or rename.” Never claim an atomic multi-file swap or durable rollback.
- Directory-layout installation only: `chunks.db` / `lancedb.lancedb`. Ordinary legacy/memory use stays supported. Reject symlink install leaves; canonical relative/symlink-parent aliases coordinate. The native String-backed database interface requires UTF-8 canonical targets: reject aliases resolving to non-UTF-8 paths with typed `DatabaseLayoutError` before any database preflight/read/recovery/write, including empty pipeline runs. Never convert database operation targets lossily. Preserve exact `PathBuf` lease-only identity; stable coordination-file creation is permitted, but original DB/recovery files and replacement-character neighbors remain untouched and acquired leases are released. UTF-8 Unicode paths and aliases to valid UTF-8 targets remain supported; arbitrary filename-byte database support is not added. Hard-link aliases, old/direct clients and unreliable network locks are outside cooperative support.
- Never sweep neighbors or read cleanup paths from a marker; never unlink/recreate the stable coordination file. Abandoned quarantines are manual removal only.
- All Python operations use `UV_NO_SYNC=1 uv`. Smoke before every commit; Rust fmt/clippy/test for Rust changes; mypy baseline comparison; full suite before any future PR push; platform CI before cross-platform support claims.
- AGENTS files are immutable without explicit request and approval. No version changes, push, merge or release are authorized.

---

## Execution order and delegation

This is one coupled lifecycle deliverable, not three independently activatable features. Tasks expose testable internal contracts before activation.

| Task | Deliverable | Dependencies / review checkpoint |
|---|---|---|
| T1 | Native lease identity, errors and PyO3 handles | Foundation only; sole native writer |
| T2 | Python ownership adapter and complete inspection scopes | T1; sole Python inspection writer |
| T3 | Complete provider/executor handle lifetime and compatible DuckDB preflight | T2; sole lifecycle writer |
| T4 | Native direct backend and run-wide ownership | T1; execute after T3 for serial review; sole native writer |
| T5 | Cancellation-safe drainage, Python/native handoff and spawned pools | T3 + T4; sole lifecycle writer |
| Gate A | Lease/lifetime correctness review | T1–T5 must pass before installer activation |
| T6 | Native marker/quarantine installation and process interruption | Gate A; sole native writer |
| T7 | Python verified-candidate installation and auto recheck policy | T2 + T5 + T6; sole hub writer |
| T8 | Explicit pull CLI and pre-registry index hook | T7; sole CLI writer |
| T9 | Actual search/overlay/recovery acceptance and operator docs | T8; sole acceptance writer |
| Gate B | Regression/type/platform evidence and human integration checkpoint | No push/merge authorization |

For subagent execution, dispatch one fresh implementer per task with this plan, the approved spec and only its predecessor interfaces. Keep one writer at a time (T3/T5 and T7/T8 share files). After each task, review spec compliance then code quality before the next writer. Independent read-only reviewers can work in parallel after writes stop; neither spawns further agents. Delegate noisy backend/platform verification separately if needed. The parent performs the final plan self-review itself. Do not activate T7/T8 while Gate A has unresolved lifetime defects.

Tasks are commit-sized review units. Within a long numbered implementation step, apply one displayed function/block at a time, then run its nearest RED/GREEN command; do not implement a whole subsystem before checking tests. Existing body-preserving edits intentionally reference exact symbols rather than copying unrelated code.

## File responsibility map

| Files | Responsibility / owning task |
|---|---|
| `src/database_coordination/{mod,error,lease,python}.rs` | Stable identity, advisory ownership, typed errors, PyO3 handles (T1) |
| `src/database_coordination/{install,install_tests}.rs` | Target-derived marker/quarantine protocol and filesystem/kill contracts (T6) |
| `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, `chunkhound_native/__init__.py` | Dependency and export wiring (T1) |
| `chunkhound/providers/database/ownership.py`, `index_format.py`, `chunkhound/hub/inspect.py` | Owned/borrowed complete inspection (T2) |
| `chunkhound/providers/database/duckdb/live_connections.py` | Same-process committed-cursor preflight, not another lock protocol (T3) |
| `serial_database_provider.py`, `serial_executor.py`, `duckdb_provider.py`, `lancedb_provider.py` under `chunkhound/providers/database/` | Provider acquisition, work admission, teardown and temporary handoff (T3/T5) |
| `chunkhound/providers/database/duckdb/connection_manager.py`, `chunkhound/database.py` | Additional/partial handle cleanup (T3) |
| `src/error.rs`, `src/db/mod.rs`, `src/db/duckdb_backend/{mod,compaction}.rs`, `src/pipeline/pipeline.rs` | Direct and run-wide native ownership/error propagation; checked canonical database path conversion (T4) |
| `chunkhound/utils/async_worker.py`, `chunkhound/pipeline_bridge.py`, `chunkhound/services/{indexing_coordinator,rust_pipeline_runner}.py` | Drain worker completion and avoid forked lease inheritance (T5) |
| `chunkhound/hub/{install,client}.py`, `errors.py`, `artifact.py` | Thin native adapter, policy and postcommit staging state (T7) |
| `chunkhound/api/cli/parsers/pull_parser.py`, `commands/pull.py`, `main.py`, `commands/run.py`, `utils/rich_output.py` | CLI registration, exits, one pre-registry hook and safe logs (T8) |
| `tests/hub/activation_support.py`, `tests/hub/test_*ownership.py`, `test_native_{leases,install}.py`, `test_activation_{client,cli,acceptance}.py` | Real backend/subprocess contracts; exact per-task lists below |
| `docs/hub-client.md`, `README.md` | Operator lifecycle, support boundaries and recovery (T9) |

## Shared interface contract

```python
# Supplied by T1 (T6 adds install): chunkhound_native
acquire_database_lease(path: str) -> DatabaseLease
acquire_install_lease(target: str, backend: str) -> InstallLease
# Both handles: close() -> None; assert_covers(path: str) -> None
# assert_covers validates identity/open handle, does not acquire a nested lease.
# InstallLease: interrupted: bool; install(candidate: str) -> list[str]
# Only install warning code: quarantine_cleanup_failed
# Typed errors: DatabaseBusyError, DatabaseInterruptedError,
#              DatabaseLayoutError, DatabaseFilesystemError

# Supplied by T2
DatabaseOwnership(handle: DatabaseLease | InstallLease)
ownership_scope(path: Path | str, ownership: DatabaseOwnership | None = None)
inspect_database(path: Path, backend: str, *,
                 ownership: DatabaseOwnership | None = None) -> Inventory

# Supplied by T5
async def await_worker_completion(future: asyncio.Future[T]) -> T
```

Shared acquisition checks `.chunkhound-install.json` under the stable `.chunkhound.lock`; exclusive permits explicit recovery. Close is idempotent. Each owner has its own descriptor; one provider closing cannot unlock others. Installer precommit failures with a marker map to interruption; before-marker filesystem failures leave old files untouched; after-commit cleanup only warns. Candidate may be consumed on pre-marker failure, so never reuse it. T7 validates configured layout, which the native actual-target basename cannot infer.

## Environment and validation preamble

```bash
cd /home/e119870/views/g/my_chunkhound/.worktrees/hub-client
export UV_NO_SYNC=1
mkdir -p .superpowers/sdd
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/activation-mypy-before.log 2>&1
export DUCKDB_LIB_DIR=/tmp/duckdb-1.5.4
export DUCKDB_INCLUDE_DIR=/tmp/duckdb-1.5.4
export LD_LIBRARY_PATH=/tmp/duckdb-1.5.4${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

Record the existing mypy baseline; do not treat known baseline errors as introduced errors. This host needs its prepared DuckDB library (see notebook `dev-env-native`); verify files exist rather than silently syncing a different wheel/runtime. Rebuild after every native task:

```bash
RUSTFLAGS='-C link-arg=-Wl,-rpath,$ORIGIN' UV_NO_SYNC=1 uv run maturin build --out target/wheels/
UV_NO_SYNC=1 uv run python scripts/install_native.py
```

Those linker flags are host-specific; CI uses its normal native build action. A missing prerequisite export/runtime is not a valid RED for a dependent task. Do not skip backend assertions to hide runtime failures.

The planning probe verified `fs4 = "=1.1.0"` with sync-only API `fs4::FileExt::try_lock_shared` / `try_lock` returning `Result<(), fs4::TryLockError>`. Published crate MSRV is 1.75; an isolated library check with Rust 1.82 passed. Use UFCS to avoid newer std File lock APIs. This is not proof of whole-app MSRV or Windows runtime correctness; platform gates remain.

---


---

### Task T1: Native leases and Python handles

**Files:** Create `src/database_coordination/{mod,error,lease,python}.rs`, `tests/hub/test_native_leases.py`; modify `Cargo.toml`, `Cargo.lock`, `src/lib.rs`, `chunkhound_native/__init__.py`.

**Consumes:** std/fs4. **Produces:** Python API except T6 install; Rust types/functions below.

- [ ] **T1.1 Write public Python ownership tests before implementing exports.** Start `tests/hub/test_native_leases.py` with these real native calls (no fake locks):

```python
import os
import subprocess
import sys
from pathlib import Path

import pytest
import chunkhound_native as native


def test_each_shared_owner_excludes_install_until_its_own_close(tmp_path):
    target = tmp_path / "chunks.db"
    first = native.acquire_database_lease(str(target))
    second = native.acquire_database_lease(str(target))
    try:
        first.close()
        first.close()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
        second.assert_covers(str(target))
    finally:
        first.close()
        second.close()
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    try:
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
        exclusive.assert_covers(str(target))
        assert not exclusive.interrupted
    finally:
        exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").is_file()


@pytest.mark.parametrize("contents", [b"", b"{", b'{"version":1}\n'])
def test_surviving_marker_refuses_shared_but_allows_exclusive(tmp_path, contents):
    target = tmp_path / "chunks.db"
    Path(str(target) + ".chunkhound-install.json").write_bytes(contents)
    with pytest.raises(native.DatabaseInterruptedError, match="chunkhound pull"):
        native.acquire_database_lease(str(target))
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    try:
        assert exclusive.interrupted
        exclusive.assert_covers(str(target))  # identity-only; no nested shared lock
        with pytest.raises(native.DatabaseLayoutError):
            exclusive.assert_covers(str(tmp_path / "other.db"))
    finally:
        exclusive.close()
    with pytest.raises(native.DatabaseLayoutError):
        exclusive.assert_covers(str(target))


def test_relative_and_symlink_parent_aliases_share_absent_identity(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    monkeypatch.chdir(tmp_path)
    lease = native.acquire_database_lease("alias/new/deeper/chunks.db")
    try:
        lease.assert_covers("real/new/deeper/../deeper/chunks.db")
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease("real/new/deeper/chunks.db", "duckdb")
    finally:
        lease.close()


def test_process_exit_releases_ownership_without_deleting_lock(tmp_path):
    target = tmp_path / "chunks.db"
    ready = tmp_path / "ready"
    code = (
        "import pathlib,sys,time,chunkhound_native as n; "
        "lease=n.acquire_database_lease(sys.argv[1]); "
        "pathlib.Path(sys.argv[2]).write_text('ready'); time.sleep(120)"
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(target), str(ready)])
    try:
        import time
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert ready.exists()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        process.kill()
        process.wait(timeout=20)
    exclusive = native.acquire_install_lease(str(target), "duckdb")
    exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission-bit contract")
def test_read_only_existing_coordination_is_usable_but_no_unlocked_fallback(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"owned by backend")
    seed = native.acquire_database_lease(str(target))
    seed.close()
    lock = Path(str(target) + ".chunkhound.lock")
    lock.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        lease = native.acquire_database_lease(str(target))
        lease.close()
        if os.access(tmp_path, os.W_OK):
            pytest.skip("privileged process bypasses directory permissions")
        with pytest.raises(native.DatabaseFilesystemError):
            native.acquire_database_lease(str(tmp_path / "uncoordinated.db"))
    finally:
        tmp_path.chmod(0o755)
        lock.chmod(0o644)
```

`UV_NO_SYNC=1 uv run pytest tests/hub/test_native_leases.py -q` => RED missing exports.

Before the RED run, append the installation-leaf boundary contract to `tests/hub/test_native_leases.py`:

```python
def test_install_refuses_symlink_leaf_but_normal_users_coordinate_referent(tmp_path):
    real = tmp_path / "real" / "chunks.db"
    real.parent.mkdir()
    real.write_bytes(b"existing")
    alias = tmp_path / "alias" / "chunks.db"
    alias.parent.mkdir()
    try:
        alias.symlink_to(real)
    except OSError:
        pytest.skip("file symlink creation is unavailable")
    holder = native.acquire_database_lease(str(alias))
    try:
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(real), "duckdb")
        with pytest.raises(native.DatabaseLayoutError):
            native.acquire_install_lease(str(alias), "duckdb")
    finally:
        holder.close()
    assert alias.is_symlink() and real.read_bytes() == b"existing"
```

Approved I1 correction: cover LanceDB directory leaves with bare, trailing `/` and
trailing `/.` spellings. Pass strings directly so Python `Path` does not remove
the suffix before the native boundary. All symlink-leaf variants must be rejected
before creating coordination; ordinary directory variants remain supported,
including a symlinked parent followed by `..` (resolve the symlink first).

```python
@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_refuses_lancedb_symlink_leaf_with_directory_suffix(tmp_path, suffix):
    real = tmp_path / "real" / "lancedb.lancedb"
    real.mkdir(parents=True)
    alias = tmp_path / "alias" / "lancedb.lancedb"
    alias.parent.mkdir()
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(str(alias) + suffix, "lancedb").close()
    assert alias.is_symlink() and real.is_dir()
    assert not Path(str(real) + ".chunkhound.lock").exists()


@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_accepts_directory_suffix_after_symlink_parent_traversal(
    tmp_path, suffix
):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    target = real / "lancedb.lancedb"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    spelling = str(alias / ".." / "lancedb.lancedb") + suffix
    exclusive = native.acquire_install_lease(spelling, "lancedb")
    try:
        exclusive.assert_covers(str(target))
        with pytest.raises(native.DatabaseLayoutError):
            exclusive.assert_covers(str(tmp_path / "lancedb.lancedb"))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        exclusive.close()
    assert Path(str(target) + ".chunkhound.lock").is_file()
```

I1 regression command against the original T1 implementation:
`UV_NO_SYNC=1 uv run pytest tests/hub/test_native_leases.py -q -k directory_suffix`
=> RED: two suffix symlink cases fail with missing `DatabaseLayoutError`; four
controls pass. After the normalization correction and native rebuild => six pass.

Approved I2 correction: raw-parent shape checks can report `NotFound` for
`missing/../lancedb.lancedb` while canonical identity reaches an existing symlink
leaf. Resolve the parent with the same absent-parent/symlink-before-`..` semantics
as identity, then inspect the still-unresolved final entry. Rejection must happen
before lock creation; valid absent target parents and symlink-parent aliases stay
supported. Include these real public regression/preservation cases:

```python
@pytest.mark.parametrize("suffix", ["", "/", "/."])
def test_install_refuses_symlink_leaf_after_absent_parent_traversal(tmp_path, suffix):
    real = tmp_path / "real" / "lancedb.lancedb"
    real.mkdir(parents=True)
    alias = tmp_path / "lancedb.lancedb"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    spelling = str(tmp_path / "missing" / ".." / alias.name) + suffix
    with pytest.raises(native.DatabaseLayoutError):
        native.acquire_install_lease(spelling, "lancedb").close()
    assert alias.is_symlink() and real.is_dir()
    assert not (tmp_path / "missing").exists()
    assert not list(tmp_path.rglob("*.chunkhound.lock"))


@pytest.mark.parametrize(
    "spelling",
    ["lancedb.lancedb", "alias/../missing/../new/deeper/lancedb.lancedb"],
)
def test_install_accepts_absent_target_parents(tmp_path, monkeypatch, spelling):
    real = tmp_path / "real"
    (real / "nested").mkdir(parents=True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real / "nested", target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    monkeypatch.chdir(tmp_path)
    target = (
        tmp_path / spelling
        if spelling == "lancedb.lancedb"
        else real / "new" / "deeper" / "lancedb.lancedb"
    )
    assert not target.exists()
    exclusive = native.acquire_install_lease(spelling, "lancedb")
    try:
        exclusive.assert_covers(str(target))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        exclusive.close()
    assert not target.exists()
    assert not (real / "missing").exists()
    assert Path(str(target) + ".chunkhound.lock").is_file()
```

I2 regression command against the I1-corrected T1 implementation:
`UV_NO_SYNC=1 uv run pytest tests/hub/test_native_leases.py -q -k 'absent_parent_traversal or absent_target_parents'`
=> RED: three symlink cases fail with missing `DatabaseLayoutError`; two
preservation controls pass. After shared parent resolution and native rebuild
=> five pass. The I1 suffix and symlink-parent traversal contracts remain required.

- [ ] **T1.2 Dependency/errors.** Add `fs4 = { version = "=1.1.0", default-features = false, features = ["sync"] }`; regenerate necessary Cargo.lock entries. New `error.rs`:

```rust
use pyo3::{create_exception, exceptions::PyRuntimeError, PyErr};

create_exception!(chunkhound_native, DatabaseBusyError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseInterruptedError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseLayoutError, PyRuntimeError);
create_exception!(chunkhound_native, DatabaseFilesystemError, PyRuntimeError);

#[derive(Debug, thiserror::Error)]
pub enum LeaseError {
    #[error("database is busy; close other ChunkHound users and retry")]
    Busy,
    #[error("database installation was interrupted; run chunkhound pull again; local work may need re-indexing")]
    Interrupted,
    #[error("unsupported database coordination layout: {0}")]
    Layout(&'static str),
    #[error("database coordination filesystem operation failed")]
    Filesystem(#[from] std::io::Error),
}

impl From<LeaseError> for PyErr {
    fn from(error: LeaseError) -> Self {
        let message = error.to_string();
        match error {
            LeaseError::Busy => DatabaseBusyError::new_err(message),
            LeaseError::Interrupted => DatabaseInterruptedError::new_err(message),
            LeaseError::Layout(_) => DatabaseLayoutError::new_err(message),
            LeaseError::Filesystem(_) => DatabaseFilesystemError::new_err(message),
        }
    }
}
```

- [ ] **T1.3 Implement identity in `lease.rs`.** Resolve existing symlinks before `..`, not lexical collapse first:

```rust
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

// Shared identity/shape resolution also permits a root or empty relative parent.
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
            Component::ParentDir => { resolved.pop(); }
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
pub(crate) enum Backend { DuckDb, LanceDb }

impl Backend {
    pub(crate) fn parse(value: &str) -> Result<Self, LeaseError> {
        match value {
            "duckdb" => Ok(Self::DuckDb),
            "lancedb" => Ok(Self::LanceDb),
            _ => Err(LeaseError::Layout("unknown backend")),
        }
    }
    pub(crate) fn filename(self) -> &'static str {
        match self { Self::DuckDb => "chunks.db", Self::LanceDb => "lancedb.lancedb" }
    }
    pub(crate) fn validate_shape(self, path: &Path) -> Result<(), LeaseError> {
        // Strip terminal directory syntax, then resolve only the parent as for identity.
        // The final entry must stay unresolved so metadata can reject a leaf symlink.
        let path: PathBuf = path.components().collect();
        let leaf = path.file_name().ok_or(LeaseError::Layout("target must have a filename"))?;
        let parent = path.parent().ok_or(LeaseError::Layout("target has no parent"))?;
        let unresolved_leaf = resolve_path(parent)?.join(leaf);
        if let Some(metadata) = metadata_optional(&unresolved_leaf)? {
            if metadata.file_type().is_symlink()
                || match self { Self::DuckDb => !metadata.is_file(), Self::LanceDb => !metadata.is_dir() }
            {
                return Err(LeaseError::Layout("target or candidate has unsupported shape"));
            }
        }
        Ok(())
    }
}
```

Shared retains legacy/memory use. Before each unresolved-leaf shape check,
reconstruct the path from `components()` to remove redundant separators and dot
components, including terminal `/` and `/.`. Split off the final filename, resolve
only its parent through the same `resolve_path` walk used by canonical identity,
and reattach the unresolved leaf before `symlink_metadata`. The common walk
resolves existing symlinks before `..` and supports absent parents; raw parent
spelling must not hide a real symlink leaf behind a false `NotFound`. Do not use
`fs::canonicalize(parent)` alone (absent parents are valid), lexically collapse
`..` before resolving symlinks, or canonicalize the leaf before checking its shape.

Exclusive rejects symlink install leaves before acquisition/coordination creation
and rechecks after acquisition. Ordinary directory suffix variants, relative
filenames, absent target parents, and symlink-parent aliases remain supported.
`canonical_target` retains the memory/empty-target/filename rules; the shared
resolver itself permits root and empty relative parents for shape validation.

- [ ] **T1.4 Implement one independently opened descriptor per owner.** Continue `lease.rs`:

```rust
pub(crate) struct Lease {
    target: PathBuf,
    _file: Option<File>,
}

impl Lease {
    pub(crate) fn target(&self) -> &Path { &self.target }
    pub(crate) fn assert_covers(&self, path: &Path) -> Result<(), LeaseError> {
        if canonical_target(path)?.as_path() != self.target() {
            return Err(LeaseError::Layout("lease covers a different target"));
        }
        Ok(())
    }
    pub(crate) fn interrupted(&self) -> Result<bool, LeaseError> {
        if self.target == Path::new(":memory:") { return Ok(false); }
        Ok(metadata_optional(&sibling(&self.target, ".chunkhound-install.json"))?.is_some())
    }
}

fn open_coordination_file(path: &Path) -> Result<File, LeaseError> {
    match File::open(path) {
        Ok(file) => return Ok(file),
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    match OpenOptions::new().read(true).write(true).create_new(true).open(path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Ok(File::open(path)?),
        Err(error) => Err(error.into()),
    }
}

fn acquire(path: &Path, exclusive: bool) -> Result<Lease, LeaseError> {
    let mut target = canonical_target(path)?;
    if target == Path::new(":memory:") {
        if exclusive { return Err(LeaseError::Layout("in-memory installation")); }
        return Ok(Lease { target, _file: None });
    }
    let parent = target.parent().ok_or(LeaseError::Layout("target has no parent"))?;
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
    let lease = Lease { target, _file: Some(file) };
    if !exclusive && lease.interrupted()? { return Err(LeaseError::Interrupted); }
    Ok(lease)
}

pub(crate) fn acquire_shared(path: &Path) -> Result<Lease, LeaseError> {
    acquire(path, false)
}

pub(crate) fn acquire_exclusive(path: &Path, backend: Backend) -> Result<Lease, LeaseError> {
    if path.file_name() != Some(std::ffi::OsStr::new(backend.filename())) {
        return Err(LeaseError::Layout("installation requires directory-layout target"));
    }
    backend.validate_shape(path)?; // resolve parent only; reject leaf symlinks before coordination
    let lease = acquire(path, true)?;
    backend.validate_shape(path)?; // ordinary cooperative opener could have raced acquisition
    lease.assert_covers(path)?;
    Ok(lease) // surviving marker is intentional explicit recovery authority
}
```

CLOEXEC is insufficient: parent uses spawn in `pipeline_bridge._get_parse_pool()`, passes no leases, tests owner exit with surviving parser children. No unsafe fork hooks; external raw fork unsupported.

- [ ] **T1.5 Export GIL-safe handles.** `python.rs` follows. T6 adds the install method; no temporary stub.

```rust
use std::path::PathBuf;
use pyo3::prelude::*;
use super::error::*;
use super::lease::{self, Backend, Lease};

#[pyclass(module = "chunkhound_native")]
pub(crate) struct DatabaseLease { held: Option<Lease> }

#[pymethods]
impl DatabaseLease {
    fn close(&mut self, py: Python<'_>) {
        let held = self.held.take();
        py.allow_threads(move || drop(held));
    }
    fn assert_covers(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let path = PathBuf::from(path);
        py.allow_threads(move || {
            self.held.as_ref().ok_or(LeaseError::Layout("lease is closed"))?.assert_covers(&path)
        }).map_err(Into::into)
    }
}

#[pyclass(module = "chunkhound_native")]
pub(crate) struct InstallLease {
    held: Option<Lease>,
    #[allow(dead_code)] // T6 consumes this when its install method lands; remove then.
    backend: Backend,
}

#[pymethods]
impl InstallLease {
    fn close(&mut self, py: Python<'_>) {
        let held = self.held.take();
        py.allow_threads(move || drop(held));
    }
    fn assert_covers(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let path = PathBuf::from(path);
        py.allow_threads(move || {
            self.held.as_ref().ok_or(LeaseError::Layout("lease is closed"))?.assert_covers(&path)
        }).map_err(Into::into)
    }
    #[getter]
    fn interrupted(&self, py: Python<'_>) -> PyResult<bool> {
        py.allow_threads(|| {
            self.held.as_ref().ok_or(LeaseError::Layout("lease is closed"))?.interrupted()
        }).map_err(Into::into)
    }
}

#[pyfunction]
fn acquire_database_lease(py: Python<'_>, path: String) -> PyResult<DatabaseLease> {
    let path = PathBuf::from(path);
    py.allow_threads(move || lease::acquire_shared(&path))
        .map(|held| DatabaseLease { held: Some(held) }).map_err(Into::into)
}

#[pyfunction]
fn acquire_install_lease(py: Python<'_>, target: String, backend: String) -> PyResult<InstallLease> {
    let path = PathBuf::from(target);
    py.allow_threads(move || {
        let backend = Backend::parse(&backend)?;
        let held = lease::acquire_exclusive(&path, backend)?;
        Ok(InstallLease { held: Some(held), backend })
    }).map_err(|error: LeaseError| error.into())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatabaseLease>()?;
    m.add_class::<InstallLease>()?;
    m.add_function(wrap_pyfunction!(acquire_database_lease, m)?)?;
    m.add_function(wrap_pyfunction!(acquire_install_lease, m)?)?;
    m.add("DatabaseBusyError", m.py().get_type_bound::<DatabaseBusyError>())?;
    m.add("DatabaseInterruptedError", m.py().get_type_bound::<DatabaseInterruptedError>())?;
    m.add("DatabaseLayoutError", m.py().get_type_bound::<DatabaseLayoutError>())?;
    m.add("DatabaseFilesystemError", m.py().get_type_bound::<DatabaseFilesystemError>())?;
    Ok(())
}
```

`mod.rs`: `pub(crate) mod error; pub(crate) mod lease; pub(crate) mod python;`. lib.rs: declare `mod database_coordination;`, call `database_coordination::python::register(m)?;` from pymodule. Append Python exports:

```python
DatabaseLease = _native_module.DatabaseLease
InstallLease = _native_module.InstallLease
acquire_database_lease = _native_module.acquire_database_lease
acquire_install_lease = _native_module.acquire_install_lease
DatabaseBusyError = _native_module.DatabaseBusyError
DatabaseInterruptedError = _native_module.DatabaseInterruptedError
DatabaseLayoutError = _native_module.DatabaseLayoutError
DatabaseFilesystemError = _native_module.DatabaseFilesystemError
```

- [ ] **T1.6 Verify:** `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`; rebuild; `UV_NO_SYNC=1 uv run pytest tests/hub/test_native_leases.py tests/contracts/test_native_duckdb_diagnostics.py -q` => PASS.
- [ ] **T1.7 Smoke/commit:** `UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto` => PASS; stage T1 files, `git commit -m "feat: add native database and installation leases"`. Review API/spawn dependency.

---

---

### Task T2: Explicit ownership scopes for complete inspection

**User-approved missing-parent correction:** Preserve non-creating inspection by returning the existing absence result when the parent is missing, without opening a backend. All actual database reads remain lease-protected; borrowed ownership and interruption checks remain enforced.

This is terminal absence-only handling at the inspection entry points, **not** an unlocked `ownership_scope` that can continue to backend reads. Resolve aliases before checking parent absence: `missing/../existing` and symlink-parent aliases must still coordinate with existing parents and refuse their markers. A supplied owner always proceeds through native identity/open validation, even for a missing path. Once absence is observed, return/raise immediately even if another process creates the parent/database concurrently. Do not change native acquisition, provider creation, or the ordinary owned/borrowed/memory scope contract. A live-connection preflight never uses the absence shortcut.

Absence results remain: DuckDB reader/format raises `IndexFormatError(_INSPECTION_FAILED)`; Lance format returns `LEGACY_INDEX_FORMAT_VERSION`; inventory raises `ValueError("Database is absent")`; preflight without a live connection returns `None`. No backend is opened on these terminal paths. Existing non-creation tests remain unchanged. Add external contracts in `test_inspection_ownership.py` for absent-parent aliases, wrong/closed borrowed owners, existing-parent interruption through missing-component aliases, and a deterministic filesystem race that publishes real storage after observing absence while forbidding backend opens.

**Files:**
- Create `chunkhound/providers/database/ownership.py`.
- Modify `chunkhound/providers/database/index_format.py` (`read_only_duckdb`, `read_format`, `preflight_duckdb`).
- Modify `chunkhound/hub/inspect.py` (`inspect_database` wrapper and Lance lifetime cleanup).
- Create `tests/hub/test_inspection_ownership.py`.
- Modify `tests/hub/test_index_format.py` (`snapshot` helper only in this task).

**Consumes:** Fixed T1 exports; existing `_stored_version`, `read_duckdb_format`, `owned_duckdb_catalog`, `require_supported`, `Inventory`, `tests.hub.artifacts.build_database`.

**Produces:**

```python
DatabaseOwnership(handle: DatabaseLease | InstallLease)
DatabaseOwnership.assert_covers(path: Path | str) -> None
DatabaseOwnership.close() -> None
ownership_scope(path: Path | str, ownership: DatabaseOwnership | None = None)
# contextmanager yielding DatabaseOwnership | None (:memory: yields None)
read_only_duckdb(path: Path, *, ownership: DatabaseOwnership | None = None)
read_format(path: Path, backend: str, *, ownership: DatabaseOwnership | None = None) -> int
preflight_duckdb(path: Path | str, *, live_connection: Any | None = None,
                 ownership: DatabaseOwnership | None = None) -> None
inspect_database(path: Path, backend: str, *, ownership: DatabaseOwnership | None = None) -> Inventory
```

- [ ] **T2.1 Write the external inspection/ownership contracts.** Create `tests/hub/test_inspection_ownership.py`:

```python
from pathlib import Path

import pytest
import chunkhound_native as native

from chunkhound.hub.inspect import inspect_database
from chunkhound.providers.database.index_format import read_only_duckdb
from chunkhound.providers.database.ownership import DatabaseOwnership
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import run_python


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_inventory_can_borrow_exclusive_ownership_without_nested_lock(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    with DatabaseOwnership(native.acquire_install_lease(str(path), backend)) as owner:
        inventory = inspect_database(path, backend, ownership=owner)
        assert inventory.format_version == 1
        assert sum(group.count for group in inventory.groups) == 100
        owner.assert_covers(path)
        with pytest.raises(native.DatabaseLayoutError):
            inspect_database(tmp_path / "other.db", backend, ownership=owner)
    with pytest.raises(native.DatabaseLayoutError):
        inspect_database(path, backend, ownership=owner)
    replacement = native.acquire_install_lease(str(path), backend)
    replacement.close()


def test_ordinary_inspection_owns_database_until_context_exit(tmp_path):
    path = build_database(tmp_path / "published")
    with read_only_duckdb(path) as conn:
        assert conn.execute("SELECT count(*) FROM inspected.main.files").fetchone()[0]
        run_python("""
            import sys
            import chunkhound_native as native
            try:
                native.acquire_install_lease(sys.argv[1], 'duckdb')
            except native.DatabaseBusyError:
                pass
            else:
                raise AssertionError('inspection released ownership early')
        """, path)
    exclusive = native.acquire_install_lease(str(path), "duckdb")
    exclusive.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_marker_refuses_ordinary_inspection_but_not_exclusive_identity_validation(tmp_path, backend):
    path = build_database(tmp_path / "published", backend)
    marker = Path(str(path) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    with pytest.raises(native.DatabaseInterruptedError):
        inspect_database(path, backend)
    lease = native.acquire_install_lease(str(path), backend)
    with DatabaseOwnership(lease) as owner:
        assert lease.interrupted
        # Explicit recovery may inspect a still-readable partial target. The
        # parent skips inspection if its recovery policy needs no old inventory.
        assert inspect_database(path, backend, ownership=owner).format_version == 1
    assert marker.read_bytes() == b"{"
```

- [ ] **T2.2 RED.** Run `UV_NO_SYNC=1 uv run pytest tests/hub/test_inspection_ownership.py -v`. Expected missing ownership module/keyword or failed exclusion assertion; not missing T1.

- [ ] **T2.3 Implement the focused ownership adapter.** Complete contents of `chunkhound/providers/database/ownership.py`:

```python
"""Scoped ownership of the native database coordination handles."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType

from chunkhound_native import DatabaseLease, InstallLease, acquire_database_lease


class DatabaseOwnership:
    """Own either native lease; borrowed scopes never close their caller's lease."""

    def __init__(self, handle: DatabaseLease | InstallLease) -> None:
        self._handle = handle

    def assert_covers(self, path: Path | str) -> None:
        self._handle.assert_covers(str(path))

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> DatabaseOwnership:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def inspection_parent_missing(
    path: Path | str, ownership: DatabaseOwnership | None = None
) -> bool:
    """Permit only a terminal absence result, never an unowned backend read."""
    if ownership is not None or str(path) == ":memory:":
        # Borrowed ownership must reach ownership_scope's identity/open check.
        return False
    try:
        # Resolve aliases before checking: missing/../existing still coordinates
        # with the existing target, including its interruption marker.
        Path(path).resolve().parent.stat()
    except FileNotFoundError:
        return True
    return False


@contextmanager
def ownership_scope(
    path: Path | str, ownership: DatabaseOwnership | None = None
) -> Iterator[DatabaseOwnership | None]:
    if ownership is not None:
        ownership.assert_covers(path)
        yield ownership
    elif str(path) == ":memory:":
        yield None
    else:
        with DatabaseOwnership(acquire_database_lease(str(path))) as acquired:
            yield acquired
```

- [ ] **T2.4 Scope format/preflight.** Import `traceback`, ownership types, and `inspection_parent_missing` in `index_format.py`. Rename existing `read_only_duckdb`/`read_format` to `_read_only_duckdb`/`_read_format`; private format calls private reader. Preserve bodies; add public wrappers:

```python
@contextmanager
def read_only_duckdb(
    path: Path, *, ownership: DatabaseOwnership | None = None
) -> Iterator[Any]:
    if inspection_parent_missing(path, ownership):
        raise IndexFormatError(_INSPECTION_FAILED)
    with ownership_scope(path, ownership):
        with _read_only_duckdb(path) as conn:
            yield conn


def read_format(
    path: Path, backend: str, *, ownership: DatabaseOwnership | None = None
) -> int:
    if inspection_parent_missing(path, ownership):
        if backend == "duckdb":
            raise IndexFormatError(_INSPECTION_FAILED)
        if backend == "lancedb":
            return LEGACY_INDEX_FORMAT_VERSION
        raise ValueError("Unsupported database backend")
    with ownership_scope(path, ownership):
        try:
            return _read_format(path, backend)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
```

Replace `preflight_duckdb` completely with:

```python
def preflight_duckdb(
    path: Path | str,
    *,
    live_connection: Any | None = None,
    ownership: DatabaseOwnership | None = None,
) -> None:
    if str(path) == ":memory:":
        return
    if live_connection is None and inspection_parent_missing(path, ownership):
        return
    with ownership_scope(path, ownership):
        live = Path(path)
        if live_connection is not None:
            import duckdb
            try:
                catalog = owned_duckdb_catalog(live_connection, live)
                require_supported(read_duckdb_format(live_connection, catalog=catalog))
            except duckdb.Error as exc:
                raise IndexFormatError(_INSPECTION_FAILED) from exc
        for suffix in ("", ".compact_backup", ".compact_new", ".old", ".compact"):
            if not suffix and live_connection is not None:
                continue
            candidate = Path(str(live) + suffix)
            if candidate.is_file() and candidate.stat().st_size:
                # These are recovery inputs belonging to the PRIMARY target.
                # Do not assert that a primary lease covers a different filename,
                # and do not obtain a second lease for each sidecar candidate.
                require_supported(_read_format(candidate, "duckdb"))
```

Private readers stay inside `index_format.py`; preserve error/close policy except for the approved Gate A failure-safe reader correction below. Remove outdated no-lifetime-protection docstrings.

**User-approved Gate A reader lifetime correction (supersedes only T2's preserve-error/close-policy instruction and T3 C3's audit-without-close-policy-change clause):** DuckDB inspection must explicitly establish reader shutdown even when third-party `__exit__` fails and a retained exception's cause/context graph keeps the reader alive. Retry idempotent close before ownership release. If close still fails, strongly retain the reader with its original ownership independently of exception frames and expose `DatabaseReaderCloseError.retry_close()`. This retries reader-only shutdown for borrowed ownership without releasing the enclosing lease; callers use the provider's public `close()` for complete provider cleanup. For an acquired standalone inspection scope whose exit already requested ownership closure, the retry also completes lease release. Do not prescribe `error.ownership.close()` for borrowed-provider recovery: its other handles may still be live. An abandoned failure conservatively retains exclusion until process exit. Borrowed scopes keep the caller's exact owner and never acquire nested leases or close its lease themselves; the owner's own close must also drain retained readers first. Provider temporary native handoff must drain/check retained inspection readers without releasing ownership and refuse while any remain unclosed. Final provider close publishes terminal closure only after all handles and native ownership close successfully, and public close retries failed shutdown even after executor finalization. Apply this lifetime to public reader, format, inventory and primary-owned recovery preflight. This narrow correction includes the retained-reader provider integration above but does not change WAL-safe read-only ATTACH/no-install policy, compatible transaction semantics, other exception cleanup, or otherwise redesign provider teardown.

- [ ] **T2.5 Scope complete inventory.** In `hub/inspect.py` import traceback/ownership/`inspection_parent_missing`; rename existing body `_inspect_database`, adding required keyword `ownership: DatabaseOwnership | None`. Change calls:

```python
version = read_format(path, backend, ownership=ownership)
# Within the DuckDB branch:
with read_only_duckdb(path, ownership=ownership) as conn:
```

Keep the existing SQL, schemas, grouping and `Inventory` return unchanged. Add this public wrapper before `_inspect_database`:

```python
def inspect_database(
    path: Path, backend: str, *, ownership: DatabaseOwnership | None = None
) -> Inventory:
    if inspection_parent_missing(path, ownership):
        raise ValueError("Database is absent")
    with ownership_scope(path, ownership) as held:
        try:
            return _inspect_database(path, backend, ownership=held)
        except BaseException as error:
            # The finished _inspect_database frame and SDK frames can otherwise
            # retain Lance datasets/scanners through a caught exception.
            traceback.clear_frames(error.__traceback__)
            raise
```

Lance locals disappear on helper return; failed completed frames clear before ownership exits. Inventory holds only scalars.

- [ ] **T2.6 Exclude only stable lock files from snapshots.** In `tests/hub/test_index_format.py` replace snapshot's dictionary:

```python
return {
    str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    for p in files
    if not p.name.endswith(".chunkhound.lock")
}
```

T1 tests lock stability; retain marker/quarantine/WAL/root/model checks.

- [ ] **T2.7 GREEN and smoke.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_inspection_ownership.py tests/hub/test_index_format.py tests/hub/test_inspect.py -v
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/activation-T2-correction-mypy.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Expected: targeted/smoke PASS; no new mypy errors. Compare diagnostic multiplicities against the preserved `.superpowers/sdd/activation-mypy-before.log`, ignoring only shifted line/column numbers; never overwrite the baseline.

- [ ] **T2.8 Commit only after green smoke and review.**

```bash
git add chunkhound/providers/database/ownership.py chunkhound/providers/database/index_format.py chunkhound/hub/inspect.py tests/hub/test_inspection_ownership.py tests/hub/test_index_format.py docs/superpowers/plans/2026-10-12-trusted-hub-activation.md
git commit -m "feat: scope Python inspection with native database ownership"
```

---

---

### Task T3: Provider lifetime, compatible live preflight, and deterministic final close

**User-approved Gate A preparation admission correction:** The caller's `prepare_connection` precheck is insufficient: a stale admitted public connect can submit after the native handoff release barrier. Recheck compaction/native admission on the serial worker immediately before preparation can inspect, recover or open handles. Publish the native flag before release; do not start native work on release failure. Clear only after actual native work/cleanup stops (or failed release), before reconnect. Preserve successful reconnect and the existing compatible transaction/preflight policy.

**Files:** Create `chunkhound/providers/database/duckdb/live_connections.py`, `tests/hub/test_provider_ownership.py`. Modify `chunkhound/providers/database/serial_database_provider.py`, `chunkhound/providers/database/serial_executor.py`, `chunkhound/providers/database/duckdb_provider.py`, `chunkhound/providers/database/lancedb_provider.py`, `chunkhound/providers/database/duckdb/connection_manager.py`, `chunkhound/database.py`, `tests/hub/test_provider_connect.py`, `tests/hub/test_index_format.py`.

**Consumes:** T2 ownership wrappers; native errors; existing serial executor operations, root sidecar functions, DuckDB preflight and compaction code.

**Produces:** provider `_ensure_database_ownership() -> DatabaseOwnership | None`, `_close_additional_connections() -> None`, `_additional_connections_closed() -> bool`; executor `disconnect_provider(provider: Any, skip_checkpoint: bool) -> None`, `handles_closed: bool`, `finalization_complete: bool`. Final close is terminal/idempotent; Rust release retains ownership.

Registry protects cooperative open admission through registration, cursor lifetime/close, and compaction check/swap/recovery/reconnect. It does not cover arbitrary third-party connections or simultaneous direct compatibility-handle manipulation. Engine read/write rules remain authoritative.

**User-approved T3 review corrections C1/I1/I2 (supersede the affected initial code blocks below):**

- **C1 — race-safe compaction admission:** Keep the existing registry RLock held across `compatible_duckdb_cursor`'s entire context. Both manager and lazy-provider opens must register before leaving that context and before extension initialization, so even retained failed-open handles are represented. Add `exclusive_duckdb_compaction(path, owned_connections)` using the same lock: refuse any registered target handle not owned by this provider (both read modes), before checkpoint/close/filesystem mutation. Hold the gate through canonical copy, filesystem swap, rollback/recovery and reconnection. Split the existing compaction body into `_compact_database_exclusive` only to scope this gate; admission refusal must never enter restore. Never close another provider's handles or change its transaction. This is a same-process lifecycle gate, not a new native lock protocol.
- **I1 — deadline is not completed failure:** Add `DatabaseOperationTimeoutError(TimeoutError)` exclusively for caller deadlines. Share `_wait_for_operation` between prepare and sync dispatch: use `concurrent.futures.wait` to observe whether completion occurred before the deadline, then call `future.result()` only for a completed future. This preserves a backend-raised `TimeoutError` and avoids inferring its source via a racy `future.done()` check. Both base and DuckDB public connect rethrow caller-deadline errors without disconnecting. Completed initialization failures still run normal cleanup. Do not cancel a timed-out worker; retain ownership during and after its completion until explicit final close. Temporary release/final close still drain without deadlines.
- **I2 — serialize final close:** Add a provider close RLock separate from the ownership mutex, encompassing executor finalization, extra-handle cleanup and ownership release. Never hold the ownership mutex while waiting for workers. Add an executor finalization RLock shared by provider finalization and generic shutdown, covering finalization selection, wait, shutdown/join and completion publication. Reuse successful finalization futures; submit a retry only if a completed attempt retained handles. Such failures retain both lease and usable executor with ordinary work refused; additional-handle failure retries run without resubmitting to a joined worker.
- **Required real contracts:** two-provider compaction refusal with subsequent durable committed write and fresh reopen, another provider's pending transaction preservation, in-flight manager/lazy open versus compaction and new open during filesystem swap; public connect deadlines in DuckDB prepare, DuckDB initialization and Lance paths, ownership retained both during and after worker completion, completed backend TimeoutError cleanup; concurrent public close at join and additional-handle cleanup boundaries; failed worker/manager close retains lease and permits public retry. Fakes/pauses only at backend, filesystem and executor boundaries.
- Run T3.10 covering suites, backend regressions, unchanged-baseline mypy comparison and smoke before every commit. Append exact RED/GREEN and self-review evidence to `activation-T3-report.md`; require independent re-review before advancing T3. No T4/T5 or native rewrite is authorized by these corrections.

**User-approved T3 retained-handle corrections C2/I3 (also supersede affected initial code blocks below):**

- **C2 — context-exit cleanup is part of open ownership:** The lazy-provider open cleanup must enclose the entire `compatible_duckdb_cursor` context, including its exit. Strongly retain newly opened connections before extension setup; transfer them to executor/manager ownership only after cursor exit succeeds. If closing a failed-open connection fails, leave it in the provider cleanup collection. The cursor context requires a lease-owning caller's retained-handle collection: retain and register every temporary cursor before yielding, remove it only after successful registered close, and propagate close failure without leaving traceback-only ownership. Apply this to provider lazy open, provider reconnect preflight, and manager open paths. Manager disconnect must retry retained cursors and release its own lease (or forget borrowed ownership) only when its main connection and retained collection are both closed. Final provider close includes that manager predicate and all retained handles.
- **I3 — temporary Rust release covers all retained handles:** Use the same complete additional-handle cleanup for final close and `release_for_rust_pipeline()`: manager connection, manager retained cursors, provider failed-open connections and provider retained cursors. Attempt every retained close, remove successes, preserve failures for retry, and reject handoff on any cleanup error or remaining handle. Keep the provider lease and usable executor during failed temporary release and after successful temporary release; only public final close releases ownership. No native run may proceed following a release error.
- **Required real contracts:** persistent compatible-cursor close failures through lazy open, provider preflight, provider manager open and standalone manager open; a retained public exception must not preserve any usable handle after successful final close and installation admission. Demonstrate installation exclusion with the source provider closed, public cleanup retry, lazy initialization plus failed backend close, and temporary-release failure/retry with retained failed-open and cursor handles. Use real DuckDB and native leases; inject only at DuckDB boundaries and track backend handles weakly. Preserve C1 compaction exclusion, I1 deadline distinction and I2 concurrent final-close semantics.
- Amend this T3 section and extracted brief; append exact RED/GREEN, covering suites, backend regressions, unchanged-baseline mypy/lint comparison and smoke-before-commit evidence to `activation-T3-report.md`. Independent re-review is required before advancing T3. These approved corrections do not authorize T4/T5.

**User-approved T3 chained-exception lifetime correction C3 (supersedes single-traceback cleanup in T3.7):**

- **Shared completed-frame cleanup:** Create `chunkhound/providers/database/exception_cleanup.py` with `clear_exception_frames(error: BaseException) -> None`. Traverse both `__cause__` and `__context__`, even when the latter is suppressed or differs from the cause, using identity-based cycle protection. Clear only completed traceback frames; preserve exception objects, diagnostic chains and traceback locations. Do not special-case regex or discard causes.
- **Complete worker boundary:** Apply that cleanup on the worker before publishing failures, outside the submitted operation so its completed frames can be cleared. Cover sync/async operations including lazy connection creation and state setup, preparation, provider finalization and generic executor shutdown. Preserve the existing finalization locks, timeout distinction, handle-retention predicates and cleanup retry behavior. Use the same helper in `read_format` and `inspect_database` before their ownership scopes exit; the original audit-without-close-policy-change requirement for DuckDB inspection is superseded only by the approved Gate A failure-safe reader correction in T2.4.
- **Required real contracts:** Populate real Lance file/chunk storage, retain a public invalid-regex error from sync/async search, and retain only a weak table reference. After public final close and installation admission, no usable table may remain; require the backend diagnostic cause/traceback to survive. Cover failed lazy opening before handle transfer and retained inspection errors as well. Exercise distinct cause/context branches and a cycle through public inspection with real datasets and faults only at the third-party backend boundary, not internal helper tests. Do not use garbage collection to establish final-close safety.
- Amend this plan and extracted brief; append exact RED/GREEN, covering suites (including `tests/hub/test_inspect.py` when inspection changes), backend regressions, unchanged-baseline mypy/lint comparison and smoke-before-commit evidence to `activation-T3-report.md`. Independent re-review remains required. Preserve C1/I1/I2/C2/I3; no T4/T5 work is authorized.

- [ ] **T3.1 Add real provider ownership contracts.** Initial contents of `tests/hub/test_provider_ownership.py`:

```python
from pathlib import Path

import duckdb
import pytest
import chunkhound_native as native

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.index_format import IndexFormatError
from chunkhound.database import Database
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from tests.hub.test_index_format import run_python, snapshot


@pytest.mark.parametrize("backend,lazy,read_only", [
    ("duckdb", False, False), ("duckdb", True, False),
    ("duckdb", False, True), ("lancedb", False, False), ("lancedb", True, False),
])
def test_live_provider_excludes_install_until_final_close(tmp_path, backend, lazy, read_only):
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    seed = cls(path, base_directory=tmp_path)
    seed.connect()
    seed.close()
    provider = cls(path, base_directory=tmp_path, config=DatabaseConfig(read_only=read_only))
    try:
        if lazy:
            provider.get_stats()
        else:
            provider.connect()
        run_python("""
            import sys
            import chunkhound_native as n
            try:
                n.acquire_install_lease(sys.argv[1], sys.argv[2])
            except n.DatabaseBusyError:
                pass
            else:
                raise AssertionError('live provider was not coordinated')
        """, path, backend)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), backend)
    finally:
        provider.close()
        provider.close()
    lease = native.acquire_install_lease(str(path), backend)
    lease.close()


def test_compatible_providers_preserve_each_others_transactions(tmp_path):
    path = tmp_path / "chunks.db"
    first = DuckDBProvider(path, base_directory=tmp_path)
    second = DuckDBProvider(path, base_directory=tmp_path)
    first.connect()
    try:
        first.execute_query("CREATE TABLE retained(value INTEGER)")
        first.execute_query("BEGIN")
        first.execute_query("INSERT INTO retained VALUES (42)")
        second.connect()
        assert second.execute_query("SELECT * FROM retained") == []
        assert first.execute_query("SELECT * FROM retained") == [{"value": 42}]
        first.execute_query("COMMIT")
        assert second.execute_query("SELECT * FROM retained") == [{"value": 42}]
        first.close()
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(path), "duckdb")
        second.connect()
        assert second.get_stats()["files"] == 0
    finally:
        second.close()
        first.close()


def test_high_level_close_drains_executor_when_manager_is_absent(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider._connection_manager.disconnect(skip_checkpoint=True)
    assert not provider.is_connected
    database = Database(path, provider=provider, indexing_coordinator=object(),
                        search_service=object(), embedding_service=object())
    database.close()
    database.close()
    lease = native.acquire_install_lease(str(path), "duckdb")
    lease.close()


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_interruption_refusal_never_runs_recovery_or_claims_root(tmp_path, backend):
    path = tmp_path / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    marker = Path(str(path) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    backup = Path(str(path) + ".compact_backup")
    backup.write_bytes(b"old files must not be recovered")
    before = snapshot(tmp_path)
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    provider = cls(path, base_directory=tmp_path)
    try:
        with pytest.raises(native.DatabaseInterruptedError):
            provider.connect()
    finally:
        provider.close()  # Must not lazy-open while cleaning a refused open.
    assert snapshot(tmp_path) == before
    assert not path.exists()
    assert not Path(str(path) + ".root.json").exists()


```

- [ ] **T3.2 RED.** `UV_NO_SYNC=1 uv run pytest tests/hub/test_provider_ownership.py -v` fails live exclusion, second-provider preflight and partial-close contracts.

- [ ] **T3.3 Provider state.** In `serial_database_provider.py` import threading, DatabaseOwnership, native acquire_database_lease. Initialize before CWD changes:

```python
self._db_path = db_path if str(db_path) == ":memory:" else Path(db_path).absolute()
self._ownership_lock = threading.RLock()
self._database_ownership: DatabaseOwnership | None = None
self._closed = False
self._closing = False
```

Add complete methods:

```python
def _ensure_database_ownership(self) -> DatabaseOwnership | None:
    with self._ownership_lock:
        if self._closed:
            raise RuntimeError("Database provider is closed")
        if str(self._db_path) == ":memory:":
            return None
        if self._database_ownership is None:
            self._database_ownership = DatabaseOwnership(
                acquire_database_lease(str(self._db_path))
            )
        self._database_ownership.assert_covers(self._db_path)
        return self._database_ownership


def _close_additional_connections(self) -> None:
    """Subclass-owned handles not held in the executor thread."""


def _additional_connections_closed(self) -> bool:
    return True
```

Construct `DuckDBConnectionManager(self._db_path, config)`. Prepend `provider._ensure_database_ownership()` to `get_thread_local_connection`; prepend `owned = self._ensure_database_ownership()` to both providers' `_create_connection`. Rename current root implementation `_ensure_indexed_root_identity_owned` and add wrapper:

```python
def ensure_indexed_root_identity(
    self, *, requested_root: Path | str, allow_claim_if_missing: bool
) -> None:
    with self._ownership_lock:
        self._ensure_database_ownership()
        self._ensure_indexed_root_identity_owned(
            requested_root=requested_root,
            allow_claim_if_missing=allow_claim_if_missing,
        )
```

Keep root body unchanged. Never hold ownership mutex while waiting for worker work.

- [ ] **T3.4 Create `chunkhound/providers/database/duckdb/live_connections.py`:**

```python
"""Independent committed cursors; not a replacement for native/engine locks."""

import os
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

_lock = threading.RLock()
_live: dict[tuple[str, bool], weakref.WeakSet[Any]] = {}


def _key(path: Path | str, read_only: bool) -> tuple[str, bool]:
    return os.path.normcase(str(Path(path).resolve())), read_only


def register_duckdb_connection(path: Path | str, read_only: bool, conn: Any) -> None:
    if str(path) == ":memory:":
        return
    with _lock:
        _live.setdefault(_key(path, read_only), weakref.WeakSet()).add(conn)


def close_duckdb_connection(conn: Any) -> None:
    with _lock:
        # Close before unregistering; a failed close remains represented.
        conn.close()
        for key, connections in list(_live.items()):
            connections.discard(conn)
            if not connections:
                del _live[key]


@contextmanager
def compatible_duckdb_cursor(
    path: Path | str, *, read_only: bool
) -> Iterator[Any | None]:
    cursor = None
    with _lock:
        if str(path) != ":memory:":
            for conn in list(_live.get(_key(path, read_only), ())):
                try:
                    cursor = conn.cursor()
                    break
                except duckdb.ConnectionException:
                    # Public compatibility callers may have closed their handle.
                    _live[_key(path, read_only)].discard(conn)
    try:
        yield cursor
    finally:
        if cursor is not None:
            cursor.close()
```

No exception-text matching or transaction mutation.

- [ ] **T3.5 Manager lifetime.** In `connection_manager.py` import DatabaseOwnership, native acquire_database_lease, preflight_duckdb, all three live-connection functions. Initialize:

```python
self._ownership: DatabaseOwnership | None = None
self._owns_ownership = False
self.joined_live_database = False
```

Replace connect:

```python
def connect(self, *, ownership: DatabaseOwnership | None = None) -> None:
    if self.connection is not None:
        if ownership is not None:
            ownership.assert_covers(self.db_path)
        return
    if not self.is_memory_db:
        if ownership is not None:
            ownership.assert_covers(self.db_path)
            self._ownership = ownership
            self._owns_ownership = False
        elif self._ownership is None:
            self._ownership = DatabaseOwnership(acquire_database_lease(str(self.db_path)))
            self._owns_ownership = True
    try:
        if not self.is_memory_db and not self._read_only:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with compatible_duckdb_cursor(self.db_path, read_only=self._read_only) as live:
            preflight_duckdb(self.db_path, live_connection=live, ownership=self._ownership)
            self.joined_live_database = live is not None
            if live is None and not self._read_only:
                self._recover_interrupted_compaction()
                self._preemptive_wal_cleanup()
            if live is None:
                self._connect_with_wal_validation()
            else:
                # Pin the live instance through this same-mode open. Recovery
                # and WAL cleanup are inappropriate for another live user's DB.
                self.connection = duckdb.connect(str(self.db_path), read_only=self._read_only)
            self._load_extensions()
            register_duckdb_connection(self.db_path, self._read_only, self.connection)
    except BaseException:
        self.disconnect(skip_checkpoint=True)
        raise
```

Keep current `disconnect` checkpoint policy; change its close call to `close_duckdb_connection(self.connection)` and set `self.connection = None` only after that succeeds. Place the whole existing connection block in an outer `try/finally`; in that outer finally add:

```python
if self.connection is None:
    if self._owns_ownership and self._ownership is not None:
        self._ownership.close()
    self._ownership = None
    self._owns_ownership = False
```

Manager never closes borrowed ownership. In `_handle_wal_corruption` replace temporary-open through explicit close with:

```python
with duckdb.connect(":memory:") as recovery_conn:
    recovery_conn.execute("INSTALL vss")
    recovery_conn.execute("LOAD vss")
    recovery_conn.execute("SET hnsw_enable_experimental_persistence = true")
    escaped = str(db_path).replace("'", "''")
    recovery_conn.execute(f"ATTACH '{escaped}' AS recovery_db")
    recovery_conn.execute("SELECT COUNT(*) FROM recovery_db.files").fetchone()
    recovery_conn.execute("CHECKPOINT recovery_db")
    recovery_conn.execute("DETACH recovery_db")
```

Existing logging/return/fallback follow the closed context.

- [ ] **T3.6 Guard DuckDB lazy opens and serial preparation without mutating other transactions.** Import all three `live_connections.py` public functions and `preflight_duckdb` into `duckdb_provider.py`. Replace `_create_connection` with:

```python
def _create_connection(self) -> Any:
    owned = self._ensure_database_ownership()
    self._connect_initialized = False
    read_only = self._connection_manager.is_read_only
    with compatible_duckdb_cursor(self.db_path, read_only=read_only) as live:
        preflight_duckdb(self.db_path, live_connection=live, ownership=owned)
        conn = duckdb.connect(str(self.db_path), read_only=read_only)
    try:
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        register_duckdb_connection(self.db_path, read_only, conn)
        return conn
    except BaseException:
        try:
            close_duckdb_connection(conn)
        except BaseException:
            self._failed_open_connections.append(conn)
            raise
        raise
```

Replace manager helper (resetting initialization during compaction is conservative):

```python
def _create_connection_manager_connection(self) -> Any:
    return self._create_connection()
```

Initialize `self._reconnect_preflight_passed = False` and `self._failed_open_connections: list[Any] = []` in DuckDBProvider's constructor. In `_executor_prepare_connection`, add `owned = self._ensure_database_ownership()` first and reset that flag. Keep its existing usability checks and two-transaction-ID checks on **its own** remaining handles unchanged. Replace the single preflight call with:

```python
if live_conn is not None:
    preflight_duckdb(self.db_path, live_connection=live_conn, ownership=owned)
else:
    with compatible_duckdb_cursor(
        self.db_path, read_only=self._connection_manager.is_read_only
    ) as compatible:
        preflight_duckdb(self.db_path, live_connection=compatible, ownership=owned)
self._reconnect_preflight_passed = True
```

Retain the close/reset block, using registered close helpers. At its end, before `return False`, add:

```python
self._connection_manager.connect(ownership=owned)
```

Remove the caller-thread `_connection_manager.connect()` from public `connect()`. Capture `had_ownership = self._database_ownership is not None` and set `_reconnect_preflight_passed = False` at public entry. Root guard still precedes executor preparation. In the existing generic exception block, add:

```python
if not had_ownership or self._reconnect_preflight_passed:
    self.disconnect(skip_checkpoint=True)
raise
```

Keep temporary-unavailability and existing-owned preflight refusal non-cleanup. Base connect's catch adds `self.disconnect(skip_checkpoint=True)` before rethrowing (Lance partial failure). In `_executor_connect` replace WAL-once inner block:

```python
if not self._wal_cleanup_done:
    if not self._connection_manager.joined_live_database:
        self._perform_wal_cleanup_in_executor(conn)
    self._wal_cleanup_done = True
    conn = _executor_local.connection
```

Replace each `executor_conn.close()`, `conn.close()`, `thread_conn.close()` and `manager_conn.close()` in `_executor_prepare_connection`, `_perform_wal_cleanup_in_executor`, `_executor_disconnect`, `_close_live_compaction_connections` with the matching `close_duckdb_connection(executor_conn)`, `close_duckdb_connection(conn)`, `close_duckdb_connection(thread_conn)` or `close_duckdb_connection(manager_conn)`. In `_close_live_compaction_connections`, replace its two try/finally blocks with:

```python
if thread_conn is not None:
    close_duckdb_connection(thread_conn)
    delattr(_executor_local, "connection")
manager_conn = self._connection_manager.connection
if manager_conn is not None:
    close_duckdb_connection(manager_conn)
    self._connection_manager.connection = None
```

Compaction's `.compact_new` connection is private to `_compact_copy_data` and retains its existing explicit finally-close. No sidecar leases.

- [ ] **T3.7 Make serial work/finalization track real handle closure and stop lazy cleanup opens.** Add `traceback` and `Callable` imports to `serial_executor.py`. Initialize:

```python
self._submission_lock = threading.RLock()
self._accepting = True
self._joined = False
self.handles_closed = True
self.finalization_complete = False
self._finalization: concurrent.futures.Future[Any] | None = None
```

Add helper; replace prepare submission with `future = self._submit(provider._executor_prepare_connection)`, sync submission with `future = self._submit(executor_operation)`, and async run-in-executor await with `result = await asyncio.wrap_future(self._submit(ctx.run, executor_operation))` (T5 hardens cancellation):

```python
def _submit(self, function: Callable[..., Any], *args: Any) -> concurrent.futures.Future[Any]:
    with self._submission_lock:
        if not self._accepting:
            raise RuntimeError("Database executor is closing or closed")
        return self._db_executor.submit(function, *args)
```

**Approved preflight correction:** Use one shared executor-operation implementation for the new connection selection, state/activity/base-directory setup, handle tracking and exception/finally logic below. Preserve the separate sync/async submission and waiting paths; do not duplicate the lifecycle block in their closures. The sync path forwards `_bypass_rust_pipeline_guard`; the async path uses its existing default. The shared operation begins with:

```python
self._raise_if_db_unavailable_before_submit(operation_name, _bypass_rust_pipeline_guard)
```

Then obtain cleanup connections without opening:

```python
conn = (
    getattr(_executor_local, "connection", None)
    if operation_name == "disconnect"
    else get_thread_local_connection(provider)
)
self.handles_closed = not hasattr(_executor_local, "connection")
```

Keep existing state/activity/base-directory setup in that shared operation implementation. Wrap its single `return op_func(conn, state, *args, **kwargs)` with:

```python
try:
    return op_func(conn, state, *args, **kwargs)
except BaseException as error:
    traceback.clear_frames(error.__traceback__)
    raise
finally:
    conn = None
    self.handles_closed = not hasattr(_executor_local, "connection")
```

`get_thread_local_connection` now ensures provider ownership, including lazy first operations. In `_executor_disconnect` add a no-handle exit before checkpointing:

```python
self._connect_initialized = False
if conn is None:
    state.clear()
    return
```

Lance's `_executor_disconnect` becomes:

```python
def _executor_disconnect(self, conn: Any, state: dict[str, Any], skip_checkpoint: bool) -> None:
    self.connection = None
    self._files_table = None
    self._chunks_table = None
    if hasattr(_executor_local, "connection"):
        delattr(_executor_local, "connection")
    state.clear()
```

Import `_executor_local` there. Its `_create_connection` calls `read_format(inspection_path, "lancedb", ownership=owned)` using the previously acquired guard. No invented synchronous Lance `.close()` call.

Add complete finalization method to the executor:

```python
def disconnect_provider(self, provider: Any, skip_checkpoint: bool) -> None:
    with self._submission_lock:
        if self._joined:
            return
        self._accepting = False

        def finish() -> None:
            conn = getattr(_executor_local, "connection", None)
            state = get_thread_local_state()
            try:
                provider._executor_disconnect(conn, state, skip_checkpoint)
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                raise
            finally:
                conn = None
                self.handles_closed = not hasattr(_executor_local, "connection")
                if self.handles_closed:
                    self.clear_thread_local()  # This is the WORKER thread.

        # Bypass ordinary submission/compaction/Rust guards only for final close.
        if self._finalization is None or self._finalization.done():
            self._finalization = self._db_executor.submit(finish)
        future = self._finalization
    try:
        future.result()  # No timeout: it must follow all previously queued work.
    finally:
        # An interrupted caller does not prove the close worker stopped.
        if future.done() and self.handles_closed:
            self._db_executor.shutdown(wait=True)
            self._joined = True
            self.finalization_complete = True
```

Close failure retaining a handle also retains executor/lease for retry. Replace generic executor shutdown/force-close with these methods; provider teardown uses `disconnect_provider`, not generic shutdown:

```python
def shutdown(self, wait: bool = True) -> None:
    if self._joined:
        return
    with self._submission_lock:
        self._accepting = False
    try:
        self._force_close_connections()
    finally:
        # Retained handles require a usable executor for cleanup retry.
        if self.handles_closed:
            self._db_executor.shutdown(wait=True)
            self._joined = True
            self.finalization_complete = True


def _force_close_connections(self) -> None:
    def close_connection() -> None:
        conn = getattr(_executor_local, "connection", None)
        try:
            if conn is not None and hasattr(conn, "close"):
                conn.close()
            self.clear_thread_local()
        finally:
            conn = None
            self.handles_closed = not hasattr(_executor_local, "connection")
    self._db_executor.submit(close_connection).result()
```

Generic shutdown cannot release extra provider handles. There are no production `wait=False` callers; explicitly reject it so later provider close is never stranded on a nonjoined shutdown executor:

```python
if not wait:
    raise ValueError("Database executor shutdown requires wait=True")
```

Place before the `_joined` check. This is a Python executor API narrowing, not a native-interface change. **Approved preflight correction:** A failed close with retained handles keeps both ownership and a usable executor for cleanup retry; ordinary work remains refused. Shut down/join only after actual handle closure, rather than claiming success or permanently stranding later cleanup.

No new post-shutdown submission is accepted. Sync operation timeouts still raise at the configured boundary; they never call `future.cancel()` or release the provider lease. `prepare_connection` gets the same rule. T5 protects async waits.

- [ ] **T3.8 Release ownership after executor and additional handles, including partial high-level close.** Replace base `disconnect` completely:

```python
def disconnect(self, skip_checkpoint: bool = False) -> None:
    with self._ownership_lock:
        if self._closed:
            return
        if self.is_rust_pipeline_in_progress():
            raise DatabaseRustPipelineInProgressError(
                "Wait for the Rust indexing operation to finish before closing"
            )
        self._closing = True
    try:
        self._executor.disconnect_provider(self, skip_checkpoint)
    finally:
        try:
            # Never close the manager alongside a still-running DB worker.
            if self._executor.finalization_complete:
                self._close_additional_connections()
        finally:
            with self._ownership_lock:
                if self._executor.finalization_complete and self._executor.handles_closed and self._additional_connections_closed():
                    self._closed = True
                    if self._database_ownership is not None:
                        self._database_ownership.close()
                        self._database_ownership = None
```

Remove DuckDB disconnect override; add:

```python
def _close_additional_connections(self) -> None:
    try:
        self._connection_manager.disconnect(skip_checkpoint=True)
    finally:
        while self._failed_open_connections:
            close_duckdb_connection(self._failed_open_connections[-1])
            self._failed_open_connections.pop()


def _additional_connections_closed(self) -> bool:
    return self._connection_manager.connection is None and not self._failed_open_connections
```

Import `DatabaseRustPipelineInProgressError` into the base provider for its final-close guard. Serialize guard publication against final-close admission by replacing the base setter with:

```python
def set_rust_pipeline_in_progress(self, active: bool) -> None:
    with self._ownership_lock:
        if active and (self._closing or self._closed):
            raise RuntimeError("Database provider is closing or closed")
        self._executor.set_rust_pipeline_in_progress(active)
```

The root wrapper also rejects `_closing` before sidecar access:

```python
if self._closing:
    raise RuntimeError("Database provider is closing")
```

Queued executor work may still use `_ensure_database_ownership` while closing so it can drain before final release. A concurrent close during a Rust run refuses without dropping ownership; the run's caller can close after awaiting completion. `release_for_rust_pipeline()` is not final close: add `_ensure_database_ownership()` at entry, keep its connection-close/error aggregation behavior, remove caller-thread `clear_thread_local()` calls, and **never** close/reset `_database_ownership` there. Temporary release cannot time out ahead of its close worker: in `execute_sync`, replace the timed wait with:

```python
result = future.result(
    timeout=None if operation_name == "disconnect" else timeout_s
)
```

This exception to ordinary configured deadlines is necessary before starting native work. Both handles still close before Rust starts. Worker disconnect has the new no-lazy behavior and registered close. Apply the same no-caller-thread-clear change to the base temporary release.

Replace `Database.close()`'s connectivity-gated body with:

```python
with self._connection_lock:
    self._provider.disconnect()
```

Base `is_connected` becomes `return not self._closed and self._executor is not None`; DuckDB retains its manager-derived flag. High-level close always drains regardless of that flag.

- [ ] **T3.9 Update obsolete tests and cleanup through the real public contract.** Replace every teardown pair in `tests/hub/test_provider_connect.py` and `tests/hub/test_index_format.py`:

```python
provider._executor.shutdown()
provider._connection_manager.disconnect(skip_checkpoint=True)
```

with `provider.disconnect(skip_checkpoint=True)`, and lone final executor shutdowns with the same public call. Rename/replace `test_failed_initialization_with_two_live_handles_does_not_skip_preflight` because automatic cleanup intentionally invalidates its premise. The complete replacement test is:

```python
def test_failed_initialization_closes_partial_handles_before_retry_with_new_provider(tmp_path):
    path = tmp_path / "chunks.db"
    with duckdb.connect(str(path)) as conn:
        conn.execute("CREATE TABLE files(id INTEGER)")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    with pytest.raises(duckdb.Error):
        provider.connect()
    assert not provider.is_connected
    provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()
    with duckdb.connect(str(path)) as conn:
        conn.execute("UPDATE schema_version SET version=2")
    retry = DuckDBProvider(path, base_directory=tmp_path)
    before = snapshot(tmp_path)
    try:
        with pytest.raises(IndexFormatError, match="Unsupported index format"):
            retry.connect()
        assert snapshot(tmp_path) == before
    finally:
        retry.close()
```

Import `chunkhound_native as native` in `test_provider_connect.py`. In existing `test_partial_reconnect_preserves_pending_work_for_explicit_completion`, after the expected `provider.connect()` refusal add:

```python
with pytest.raises(native.DatabaseBusyError):
    native.acquire_install_lease(str(provider.db_path), "duckdb")
```

Keep existing transaction completion assertions.

- [ ] **T3.10 GREEN, backend regressions, type comparison and smoke.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_provider_ownership.py tests/hub/test_provider_connect.py tests/hub/test_index_format.py tests/hub/test_inspection_ownership.py tests/unit/test_database_close.py -v
UV_NO_SYNC=1 uv run pytest tests/test_duckdb_compaction.py tests/unit/test_rust_pipeline_flag.py tests/unit/test_rust_pipeline_runner.py -v
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/lifetime-l2-mypy.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Expected: targeted/regressions/smoke PASS; no introduced mypy errors.

- [ ] **T3.11 Commit after smoke/review.**

```bash
git add chunkhound/providers/database/duckdb/live_connections.py chunkhound/providers/database/serial_database_provider.py chunkhound/providers/database/serial_executor.py chunkhound/providers/database/duckdb_provider.py chunkhound/providers/database/duckdb/connection_manager.py chunkhound/providers/database/lancedb_provider.py chunkhound/database.py tests/hub/test_provider_ownership.py tests/hub/test_provider_connect.py tests/hub/test_index_format.py
git commit -m "fix: retain database ownership through provider handles and final close"
```

---

---

### Task T4: Native run-wide and backend ownership with typed failures

**Files:** Modify `src/error.rs`, `src/db/mod.rs`, `src/pipeline/pipeline.rs`, `src/db/duckdb_backend/{mod,compaction}.rs`; create `src/db/duckdb_backend/ownership_tests.rs`, `tests/hub/test_native_pipeline_ownership.py`.

**Consumes:** T1 acquire_shared/Lease/LeaseError, existing DbConfig/DbBackend. **Produces:** guarded native work and typed errors. T3/T5 retain provider guard through handoff.

**User-approved I1 correction:** The String-backed native database interface rejects canonical targets that are not UTF-8 with `DbError::Coordination(LeaseError::Layout(_))`, exposed as `DatabaseLayoutError`, before any database preflight/read/recovery/write. Validate at direct open/read and before the pipeline empty shortcut; diff and writer receive the same checked owned String. Keep `Lease::target()` as an exact `PathBuf` identity and do not change lease-only acquisition or broaden arbitrary filename-byte support. A shared checked conversion helper in `src/db/mod.rs` avoids mismatched behavior. Never use `to_string_lossy()` for database operation targets; compaction ATTACH uses the already checked String plus its suffix. Existing stable coordination-file creation is permitted; on refusal, original DB/recovery files and replacement-character neighboring targets stay unchanged and acquired leases are released. Ordinary UTF-8 Unicode, aliases to valid UTF-8 targets and memory remain supported.

Add real Unix-gated raw-byte fixtures: ASCII symlink-parent aliases to non-UTF-8 directories; public pipeline empty/nonempty and absent/existing target cases; direct backend open/read. Include invalid DB/recovery inputs so typed layout errors establish rejection before preflight/recovery, snapshot bytes/mtime and directory entries for both original and replacement-character neighbors (including stable coordination files), and acquire exclusive ownership after failure while the caller object remains alive. Cover positive Unicode/valid-alias/memory behavior. Append RED/GREEN, full covering validation and self-review evidence to `activation-T4-report.md`; re-review I1 before accepting T4. Update this plan, the approved spec and extracted T4/global briefs together; T9 owns creating `docs/hub-client.md`, not this correction.

- [ ] **T4.1 Write failing direct-pipeline contracts.** `tests/hub/test_native_pipeline_ownership.py`:

```python
from pathlib import Path

import pytest
import chunkhound_native as native


@pytest.mark.parametrize("files", [[], [("source.py", "source.py")]])
def test_direct_pipeline_refuses_marker_before_even_empty_shortcut(tmp_path, files):
    marker = tmp_path / "chunks.db.chunkhound-install.json"
    marker.write_bytes(b"{")
    pipeline = native.IndexingPipeline({"db_path": str(tmp_path), "skip_embeddings": True})
    with pytest.raises(native.DatabaseInterruptedError):
        pipeline.run(files, lambda paths, config: [("python", []) for _ in paths])
    assert not (tmp_path / "chunks.db").exists()


def test_pipeline_owns_diff_to_store_and_after_backend_close(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("def answer(): return 42\n")
    db_dir = tmp_path / "index"
    target = db_dir / "chunks.db"
    phases = []
    failures = []

    def progress(phase, current, total, *extra):
        # Before writer open / after writer close but before joins.
        if (phase == "parse" and current == 0) or phase == "write-done":
            try:
                lease = native.acquire_install_lease(str(target), "duckdb")
            except native.DatabaseBusyError:
                phases.append(phase)
            else:
                lease.close()
                failures.append(phase)

    pipeline = native.IndexingPipeline({"db_path": str(db_dir), "skip_embeddings": True})
    pipeline.run(
        [(str(source), "source.py")],
        lambda paths, config: [("python", []) for _ in paths],
        progress_callback=progress,
    )
    assert failures == []  # native intentionally ignores callback exceptions
    assert "parse" in phases and "write-done" in phases
    native.acquire_install_lease(str(target), "duckdb").close()
```

`UV_NO_SYNC=1 uv run pytest tests/hub/test_native_pipeline_ownership.py -q` => RED: empty shortcut/unguarded phases.

- [ ] **T4.2 Rust contracts.** Add `#[cfg(test)] mod ownership_tests;` in backend mod.rs. Existing real-DB test fixtures:

```rust
use super::{test_support, DuckDbHnswBackend};
use crate::database_coordination::{error::LeaseError, lease::{acquire_exclusive, Backend, sibling}};
use crate::db::DbBackend;
use crate::error::DbError;

#[test]
fn direct_reader_checks_marker_before_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    std::fs::write(sibling(&path, ".chunkhound-install.json"), b"{").unwrap();
    std::fs::write(sibling(&path, ".old"), b"old recovery input").unwrap();
    std::fs::write(sibling(&path, ".swap_intent"), b"phase1").unwrap();
    let backend = DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    assert!(matches!(backend.read_file_states(), Err(DbError::Coordination(LeaseError::Interrupted))));
    assert!(!path.exists());
    assert_eq!(std::fs::read(sibling(&path, ".old")).unwrap(), b"old recovery input");
}

#[test]
fn open_close_and_implicit_drop_release_only_after_owned_connection() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    let mut backend = DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    backend.open().unwrap();
    backend.open().unwrap();
    assert!(matches!(acquire_exclusive(&path, Backend::DuckDb), Err(LeaseError::Busy)));
    backend.close().unwrap();
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
    backend.open().unwrap();
    drop(backend);
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
}

#[test]
fn compaction_keeps_ownership_while_live_connection_is_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("chunks.db");
    let mut backend = DuckDbHnswBackend::new(test_support::config(path.to_string_lossy().into_owned()));
    backend.open().unwrap();
    backend.write_batch(&test_support::single_file_batch("source.py")).unwrap();
    let observed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed_in_hook = observed.clone();
    backend.compaction_closed_hook = Some(Box::new(move |path| {
        assert!(matches!(acquire_exclusive(path, Backend::DuckDb), Err(LeaseError::Busy)));
        observed_in_hook.store(true, std::sync::atomic::Ordering::SeqCst);
    }));
    backend.run_compaction().unwrap();
    assert!(observed.load(std::sync::atomic::Ordering::SeqCst));
    backend.close().unwrap();
    drop(acquire_exclusive(&path, Backend::DuckDb).unwrap());
}
```

Add `#[cfg(test)] type CompactionClosedHook = Box<dyn Fn(&std::path::Path) + Send>;` and test-only field `compaction_closed_hook: Option<CompactionClosedHook>`, initialize None; after `self.conn = None;` in compaction:

```rust
#[cfg(test)]
if let Some(hook) = &self.compaction_closed_hook { hook(&db_path); }
```

No production hooks. `cargo test ownership_tests` => RED missing variant/hook, then unguarded access.

- [ ] **T4.3 Backend guard and exact target conversion.** Add a shared helper to `src/db/mod.rs`:

```rust
/// Convert a leased canonical target without substituting filesystem bytes.
pub(crate) fn database_path_string(target: &std::path::Path) -> Result<String, DbError> {
    target.to_str().map(str::to_owned).ok_or_else(|| {
        crate::database_coordination::error::LeaseError::Layout(
            "native database access requires a UTF-8 canonical target",
        ).into()
    })
}
```

Import `crate::db::database_path_string` at both backend and pipeline call sites. Import `crate::database_coordination::lease::Lease`; add `lease: Option<Lease>` after conn (drop order), initialize None. Move existing open body from crash-recovery comment through Ok into inherent `fn open_owned(&mut self) -> Result<(), DbError>` unchanged; trait open becomes:

```rust
fn open(&mut self) -> Result<(), DbError> {
    if self.conn.is_some() { return Ok(()); }
    if self.lease.is_none() {
        let lease = crate::database_coordination::lease::acquire_shared(
            std::path::Path::new(&self.config.db_path)
        )?;
        self.config.db_path = database_path_string(lease.target())?;
        self.lease = Some(lease);
    }
    let result = self.open_owned();
    if result.is_err() { let _ = self.close(); }
    result
}
```

Close tail: `self.conn = None; self.lease = None; result` even on errors. Local partial-open connections unwind first. Compaction's `conn=None` must not clear lease.

Direct read uses local primary-target lease, never recovery-sidecar lease:

```rust
fn read_file_states(&self) -> Result<Vec<DbFileEntry>, DbError> {
    let lease = crate::database_coordination::lease::acquire_shared(
        std::path::Path::new(&self.config.db_path)
    )?;
    let mut config = self.config.clone();
    config.db_path = database_path_string(lease.target())?;
    let reader = Self::new(config);
    Self::preflight_index_formats(lease.target())?;
    read::read_file_states(&reader)
}
```

read.rs connections/statements drop before lease; result owns data.

- [ ] **T4.4 Typed errors.** Add DbError variant:

```rust
#[error(transparent)]
Coordination(#[from] crate::database_coordination::error::LeaseError),
```

Replace its Python conversion with:

```rust
impl From<DbError> for PyErr {
    fn from(error: DbError) -> PyErr {
        match error {
            DbError::Coordination(error) => error.into(),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}
```

In `pipeline.rs`, keep parse/embed per-file error strings unchanged. Make **only** the orchestration/store DB channel typed:

```rust
// Return types at the existing definitions/declarations:
// pipeline_parse_embed_store: Result<StoreOutcome, DbError>
// store_handle: JoinHandle<Result<StoreOutcome, DbError>>
// write_result and post_write_result: Result<(), DbError>
fn close_backend_on_err<T>(
    backend: &mut dyn DbBackend,
    result: Result<T, DbError>,
) -> Result<T, DbError> {
    result.map_err(|error| { let _ = backend.close(); error })
}
```

Store DB operations use direct `?`, not string map_err. Post-join string errors become `DbError::Other(e)`; panic strings become DbError::Other as well. All three joins precede propagation. run() uses `.map_err(PyErr::from)?`. Existing helper tests need no change.

Mechanically wrap the existing pipeline_parse_embed_store body in a scope: insert `std::thread::scope(|scope| {` as its first expression and `})` at its end. Replace three std::thread::spawn calls with scope.spawn; annotated handles become `std::thread::ScopedJoinHandle<'_, Result<StoreOutcome, DbError>>` and `std::thread::ScopedJoinHandle<'_, Result<u64, String>>`. Preserve explicit joins. This also joins previously created workers if a later spawn panics.

- [ ] **T4.5 Run-wide guard.** Before `let started = Instant::now();`:

```rust
let requested_db_file = if self.config.db_path.as_os_str().is_empty()
    || self.config.db_path.as_os_str() == ":memory:"
{
    PathBuf::from(":memory:")
} else {
    self.config.db_path.join("chunks.db")
};
let run_lease = py.allow_threads(move || {
    crate::database_coordination::lease::acquire_shared(&requested_db_file)
})?;
let db_file = run_lease.target().to_path_buf();
let result = (|| -> PyResult<PipelineReport> {
    // Validate before any shortcut or DB preflight/recovery/read/write.
    let db_path = database_path_string(&db_file)?;
```

Keep run body inside closure, remove duplicate target resolution; replace no_db_yet:

```rust
let probe = db_file.clone();
let no_db_yet = py.allow_threads(move || {
    probe == Path::new(":memory:") || !probe.exists()
});
```

Both compute_diff_blocking calls gain final `db_path.clone()` argument; signature gains `db_path: String`. Remove its target recomputation and lossy conversion; memory shortcut checks `db_path == ":memory:"`. Writer `DbConfig` consumes the same checked `db_path` String. Keep the exact `db_file: PathBuf` for the existence probe only. Remove redundant run() create_dir_all (lease acquisition did it without GIL). Validation stays inside the result closure so its typed error also takes the explicit no-GIL run-lease release path below.

After the existing final report-construction expression, close the closure and add:

```rust
})();
// Also release on early/error returns, after all worker joins.
py.allow_threads(move || drop(run_lease));
result
```

Run guard starts before empty shortcut, spans joins; backend/provider guards intentionally overlap, no transfer/upgrade.

- [ ] **T4.6 Verify.** `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test`; rebuild; `UV_NO_SYNC=1 uv run pytest tests/hub/test_native_pipeline_ownership.py tests/hub/test_native_leases.py tests/contracts/test_pipeline_parallel.py -q` => PASS.
- [ ] **T4.7 Smoke/commit:** `UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto` => PASS; stage T4 files, `git commit -m "fix: retain native ownership across database work"`. Review T2–T5 lifetimes.

---

---

### Task T5: Drain cancelled workers, preserve handoff ownership, and spawn parser workers

**Files:**
- Create `chunkhound/utils/async_worker.py`.
- Modify `chunkhound/providers/database/serial_executor.py` (`execute_async`).
- Modify `chunkhound/providers/database/serial_database_provider.py` (`compact_database_async`).
- Modify `chunkhound/pipeline_bridge.py` (native await, cleanup await, error categories, parse pool context).
- Modify `chunkhound/services/indexing_coordinator.py` (both ProcessPoolExecutor constructors).
- Modify `chunkhound/services/rust_pipeline_runner.py` (outdated fork comment only; release/reconnect ordering stays).
- Create `tests/hub/test_worker_ownership.py`.

**Consumes:** T4 run-wide native leases; T3 release-without-unlock; existing bridge/runner/parser APIs.

**Produces:** `async await_worker_completion(future: asyncio.Future[T]) -> T`. Parent submits retained futures and awaits inside lease/staging scopes. No native API change.

- [ ] **T5.1 Write repeated-cancellation and handoff contracts.** Create `tests/hub/test_worker_ownership.py`:

```python
import asyncio
import threading
from pathlib import Path

import pytest
import chunkhound_native as native

from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.utils.async_worker import await_worker_completion


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_fails", [False, True])
async def test_cancelled_serial_work_drains_before_close(tmp_path, worker_fails):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work(value: int) -> int:
        started.set()
        try:
            assert release.wait(20)
            if worker_fails:
                raise OSError("injected worker failure")
            return value
        finally:
            finished.set()

    provider.connection.create_function("lifetime_blocked", work, side_effects=True)
    task = asyncio.create_task(provider._execute_in_db_thread(
        "execute_query", "SELECT lifetime_blocked(42)", None))
    try:
        assert await asyncio.to_thread(started.wait, 10)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


```

Real native exclusion and controlled worker boundary; no fake lock.

- [ ] **T5.2 RED.** Run `UV_NO_SYNC=1 uv run pytest tests/hub/test_worker_ownership.py -v`. Initially the async helper import is missing; once T5.3 exists, the same serial contract must still fail before T5.4 because cancellation returns early. Native handoff acceptance is added in T5.7.

- [ ] **T5.3 Implement the generic async completion helper.** Complete new file `chunkhound/utils/async_worker.py`:

```python
"""Do not release worker-owned resources just because an await was cancelled."""

import asyncio
from typing import TypeVar

T = TypeVar("T")


async def await_worker_completion(future: asyncio.Future[T]) -> T:
    """Drain retained worker; cancellation wins after retrieving its result/error."""
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            cancellation = error
        except BaseException:
            # Future holds the original error; retrieve it exactly once below.
            break
    try:
        result = future.result()
    except BaseException:
        if cancellation is not None:
            raise cancellation from None
        raise
    if cancellation is not None:
        raise cancellation
    return result
```

Do not uncancel the caller or cancel the underlying worker future.

- [ ] **T5.4 Apply drainage to serial operations and threaded compaction.** In `serial_executor.py` import the helper and replace the async submission/await from T3 with:

```python
future = asyncio.wrap_future(self._submit(ctx.run, executor_operation))
result = await await_worker_completion(future)
```

Keep sampled auto-compaction after this result; that path uses the same async executor method. In `serial_database_provider.py` import the helper and replace `compact_database_async` body with:

```python
future = asyncio.get_running_loop().run_in_executor(None, self.compact_database)
return await await_worker_completion(future)
```

Ordinary sync execute/prepare retain deadlines and lease after timeout; disconnect drains without timeout.

- [ ] **T5.5 Keep native run/cache cleanup inside the cancellation-safe boundary and preserve typed failures.** Import `contextvars` and the helper in `pipeline_bridge.py`. Inside `run_rust_pipeline` import the four native exception classes alongside `IndexingPipeline`. Retain the actual executor completion future for both native run and cleanup, not `create_task(to_thread(...))`: `asyncio.run` shutdown independently cancels all Tasks without stopping their threads. Copy the caller's context to preserve `to_thread` semantics. Submit `pipeline.run` with its complete existing arguments via `functools.partial`:

```python
worker = asyncio.get_running_loop().run_in_executor(
    None,
    contextvars.copy_context().run,
    functools.partial(pipeline.run, ...),  # Preserve all existing native arguments.
)
report = await await_worker_completion(worker)
```

This user-approved I1 correction supersedes the original retained-Task prescription. Do not fix it with another shield around a cancellable Task. Add real `asyncio.run` all-Tasks shutdown contracts at native run and cleanup boundaries: cleanup cannot overlap native work, and runner reconnect/resource release cannot precede actual cleanup completion. Preserve repeated cancellation, cancellation outcome and caller-context propagation.

Before its generic exception wrapper add:

```python
except (
    DatabaseBusyError,
    DatabaseInterruptedError,
    DatabaseLayoutError,
    DatabaseFilesystemError,
):
    raise
```

Leave generic non-coordination failures as the existing `RustPipelineError`. Replace cleanup await with:

```python
cleanup = asyncio.get_running_loop().run_in_executor(
    None,
    contextvars.copy_context().run,
    _shutdown_embed_thread_resources,
    embed_cache,
)
await await_worker_completion(cleanup)
```

Preserve runner finally-reconnect after bridge finishes. Replace outdated fork-safe comment:

```python
# The provider retains native shared ownership while its backend handles are
# closed. Native work independently owns a shared lease for its entire run.
# Parser pools use spawn, so children cannot inherit either database state or
# lease descriptors. run_rust_pipeline drains cancelled workers and embed-cache
# cleanup before this function clears the guard and reconnects in finally.
```

- [ ] **T5.6 Spawn all parser pools, including the Python indexing path.** Add `import multiprocessing` to `pipeline_bridge.py`; replace its constructor:

```python
pool = ProcessPoolExecutor(
    max_workers=max(1, max_workers),
    mp_context=multiprocessing.get_context("spawn"),
)
```

`indexing_coordinator.py` already imports multiprocessing. Replace **both** `with ProcessPoolExecutor(max_workers=num_workers) as executor:` occurrences with:

```python
with ProcessPoolExecutor(
    max_workers=num_workers,
    mp_context=multiprocessing.get_context("spawn"),
) as executor:
```

Keep callbacks/config; no global start-method mutation.

- [ ] **T5.7 Add real serial cancellation/timeout and native handoff cancellation acceptance.** Append these tests to `tests/hub/test_worker_ownership.py`; add imports `concurrent.futures`, `time`, `DatabaseConfig`, `run_rust_indexing_phase`:

```python
def test_timeout_does_not_release_running_work(tmp_path):
    path = tmp_path / "chunks.db"
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    provider._executor._execute_timeout_seconds = 0.05
    release = threading.Event()
    def blocked(value: int) -> int:
        assert release.wait(20)
        return value
    provider.connection.create_function("blocked", blocked, side_effects=True)
    with concurrent.futures.ThreadPoolExecutor(1) as caller:
        try:
            with pytest.raises(TimeoutError):
                provider.execute_query("SELECT blocked(42)")
            closing = caller.submit(provider.close)
            time.sleep(0.1)
            assert not closing.done()
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        finally:
            release.set()
    provider.close()
    native.acquire_install_lease(str(path), "duckdb").close()


@pytest.mark.asyncio
async def test_native_cancellation_drains_before_reconnect(tmp_path, monkeypatch):
    from chunkhound.services.rust_pipeline_runner import run_rust_indexing_phase
    import chunkhound.pipeline_bridge as bridge
    path, source = tmp_path / "chunks.db", tmp_path / "example.py"
    source.write_text("def example():\n    return 42\n")
    provider = DuckDBProvider(path, base_directory=tmp_path)
    provider.connect()
    started, release = threading.Event(), threading.Event()
    actual = bridge.parse_batch_callback
    def paused(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return actual(*args, **kwargs)
    monkeypatch.setattr(bridge, "parse_batch_callback", paused)
    task = asyncio.create_task(run_rust_indexing_phase(
        db=provider, config=None, embedding_provider=None, progress=None,
        files_to_process=[(source, None)], directory=tmp_path,
        force_reindex=True, do_cleanup=True, diff_task=None, parse_task=None))
    try:
        assert await asyncio.to_thread(started.wait, 20)
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and provider.is_rust_pipeline_in_progress()
            assert not provider.is_connected
            with pytest.raises(native.DatabaseBusyError):
                native.acquire_install_lease(str(path), "duckdb")
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.is_connected and not provider.is_rust_pipeline_in_progress()
        assert provider.get_stats()["files"] == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        provider.close()
```

The existing `_executor_execute_query(conn, state, query, params)` requires the explicit final `None` argument shown above; no new query API is introduced. If environment initialization lacks VSS/native runtime, fix prerequisites; do not turn the exclusion assertions into skips.

- [ ] **T5.8 Test owner exit while its real parser child survives.** Append to the same file; import `os`, `signal`, `subprocess`, `textwrap`:

```python
@pytest.mark.skipif(os.name != "posix", reason="SIGKILL owner/child contract")
def test_parser_worker_does_not_inherit_owner_lease(tmp_path):
    script = tmp_path / "owner.py"
    script.write_text(textwrap.dedent('''
        import os, sys, time
        from pathlib import Path
        import chunkhound_native as native
        from chunkhound.pipeline_bridge import _get_parse_pool

        def child(pid_path, stop_path):
            Path(pid_path).write_text(str(os.getpid()))
            deadline = time.monotonic() + 30
            while not Path(stop_path).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os._exit(0)

        if __name__ == '__main__':
            lease = native.acquire_database_lease(sys.argv[1])
            Path(sys.argv[4]).write_text(str(os.getpid()))
            _get_parse_pool(1).submit(child, sys.argv[2], sys.argv[3]).result()
    '''))
    target = tmp_path / "chunks.db"
    child_file, owner_file = tmp_path / "child.pid", tmp_path / "owner.pid"
    stop = tmp_path / "stop"
    process = subprocess.Popen(
        ["uv", "run", "python", str(script), str(target), str(child_file),
         str(stop), str(owner_file)],
        env={**os.environ, "UV_NO_SYNC": "1"},
    )
    try:
        deadline = time.monotonic() + 20
        while not child_file.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert child_file.exists()
        os.kill(int(owner_file.read_text()), signal.SIGKILL)
        process.wait(timeout=10)
        os.kill(int(child_file.read_text()), 0)
        native.acquire_install_lease(str(target), "duckdb").close()
    finally:
        stop.touch()
        if process.poll() is None and owner_file.exists():
            try:
                os.kill(int(owner_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=35)
```

The worker exits on stop/30-second deadline. Its death is not required before the exclusive probe; owner exit alone must release the lease.

- [ ] **T5.9 GREEN, both indexing paths and smoke.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_worker_ownership.py -v
CHUNKHOUND_USE_RUST=0 UV_NO_SYNC=1 uv run pytest tests/hub/test_provider_ownership.py tests/hub/test_inspection_ownership.py -v
UV_NO_SYNC=1 uv run pytest tests/test_pipeline_bridge_config.py tests/unit/test_rust_pipeline_runner.py tests/unit/test_rust_pipeline_flag.py tests/integration/test_rust_pipeline_lancedb_fallback.py -v
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/lifetime-l3-mypy.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
```

Expected targeted/regressions/smoke PASS; no new mypy errors. T4 runs Rust gates. T9 owns full Python-pipeline indexing/overlay acceptance, not inferred from these provider tests.

- [ ] **T5.10 Commit after smoke and lifetime review.**

```bash
git add chunkhound/utils/async_worker.py chunkhound/providers/database/serial_executor.py chunkhound/providers/database/serial_database_provider.py chunkhound/pipeline_bridge.py chunkhound/services/indexing_coordinator.py chunkhound/services/rust_pipeline_runner.py tests/hub/test_worker_ownership.py
git commit -m "fix: drain cancelled database workers before releasing ownership"
```

---

## Gate A: Review ownership before installer activation

Verify real provider/shared-holder contention, compatible transaction-preserving preflight, partial/lazy opens, final handle release, native direct readers/run-wide scope, cancelled workers, temporary handoff and parser child inheritance. T1–T5 test commands must pass. Refusal under a surviving marker must precede recovery/root mutation even for an empty direct native run. Do not start T6 activation while any path can release ownership before its work/handles stop.


---

### Task T6: Rust marker/quarantine installer and deterministic interruption recovery

**Files:** Create `src/database_coordination/{install,install_tests}.rs`, `tests/hub/test_native_install.py`; modify `src/database_coordination/{mod,python}.rs`.

**Consumes:** T1 Lease/Backend/LeaseError, T4 guarded opens, T7's verified closed candidate. **Produces:** `.install(candidate: str) -> list[str]`; no test exports. Caller owns close.

- [ ] **T6.1 Add real candidate/replacement contract before implementation.** `tests/hub/test_native_install.py` starts with:

```python
from pathlib import Path

import pytest
import chunkhound_native as native
from chunkhound.hub.inspect import inspect_database
from tests.hub.artifacts import build_database


DUCKDB_SUFFIXES = (
    ".wal", ".wal.corrupt", ".compact_backup", ".compact_backup.wal",
    ".compact_new", ".compact_new.wal", ".old", ".old.wal",
    ".compact", ".compact.wal", ".swap_intent", ".export_tmp",
    ".root.json", ".embedding.json",
)


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
def test_install_isolates_exact_old_inputs_and_preserves_neighbors(tmp_path, backend):
    directory = tmp_path / "index"
    target = build_database(directory, backend, model="old-model")
    candidate = build_database(directory / "private-stage", backend)
    expected = inspect_database(candidate, backend)
    suffixes = DUCKDB_SUFFIXES if backend == "duckdb" else (".embedding.json",)
    for suffix in suffixes:
        path = Path(str(target) + suffix)
        if suffix == ".export_tmp":
            path.mkdir(exist_ok=True)
            (path / "old-export").write_text("old")
        else:
            path.write_text("old")
    neighbors = [directory / "notes.txt", Path(str(target) + ".root.json.1.2.tmp")]
    for neighbor in neighbors:
        neighbor.write_text("keep")
    lease = native.acquire_install_lease(str(target), backend)
    try:
        assert lease.install(str(candidate)) == []
        assert not lease.interrupted
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_database_lease(str(target))
    finally:
        lease.close()
    assert inspect_database(target, backend) == expected
    assert not candidate.exists()
    assert all(not Path(str(target) + suffix).exists() for suffix in suffixes)
    assert all(neighbor.read_text() == "keep" for neighbor in neighbors)
    assert Path(str(target) + ".chunkhound.lock").exists()


def test_missing_candidate_does_not_touch_old_target_or_create_marker(tmp_path):
    target = tmp_path / "chunks.db"
    target.write_bytes(b"old")
    lease = native.acquire_install_lease(str(target), "duckdb")
    try:
        with pytest.raises(native.DatabaseLayoutError):
            lease.install(str(tmp_path / "missing" / "chunks.db"))
        assert target.read_bytes() == b"old"
        assert not lease.interrupted
    finally:
        lease.close()
```

`UV_NO_SYNC=1 uv run pytest tests/hub/test_native_install.py -q` => RED missing install. T9 acceptance adds CLI search/overlay.

- [ ] **T6.2 Exact paths/test boundary.** install.rs imports `std::{fs, io::{self, Write}, path::{Path, PathBuf}}` and `super::{error::LeaseError, lease::{Backend, Lease, canonical_target, metadata_optional, sibling}}`:

```rust
const DUCKDB_SUFFIXES: &[&str] = &[
    "", ".wal", ".wal.corrupt", ".compact_backup", ".compact_backup.wal",
    ".compact_new", ".compact_new.wal", ".old", ".old.wal",
    ".compact", ".compact.wal", ".swap_intent", ".export_tmp",
    ".root.json", ".embedding.json",
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
        let path = sibling(target, &format!(".chunkhound-quarantine-{:032x}", rand::random::<u128>()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err(LeaseError::Filesystem(io::Error::new(io::ErrorKind::AlreadyExists, "quarantine name exhausted")))
}

pub(crate) fn install(lease: &Lease, backend: Backend, candidate: &Path) -> Result<Vec<String>, LeaseError> {
    install_with(lease, backend, candidate, &mut |_| Ok(()))
}
```

Only tests supply install_with callbacks: fail before syscall or pause after it. No Python export or production env hooks; real filesystem protocol, fixed cleanup paths.

- [ ] **T6.3 Validate/preflight without touching the old database.** Implement the first half of `install_with`:

```rust
pub(super) fn install_with(
    lease: &Lease,
    backend: Backend,
    candidate: &Path,
    boundary: &mut impl FnMut(InstallPoint) -> io::Result<()>,
) -> Result<Vec<String>, LeaseError> {
    let target = lease.target();
    backend.validate_shape(target)?;
    backend.validate_shape(candidate)?; // symlink leaf rejected before canonicalization
    if metadata_optional(candidate)?.is_none() {
        return Err(LeaseError::Layout("candidate is absent"));
    }
    let candidate = canonical_target(candidate)?;
    if candidate.file_name() != Some(std::ffi::OsStr::new(backend.filename()))
        || candidate == target || candidate.starts_with(target)
    {
        return Err(LeaseError::Layout("candidate must be a separate private database"));
    }
    let marker = sibling(target, ".chunkhound-install.json");
    if let Some(metadata) = metadata_optional(&marker)? {
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(LeaseError::Layout("installation marker is not a regular file"));
        }
    }
    let quarantine = new_quarantine(target)?;
    let incoming = quarantine.join("candidate");
    // Same-filesystem proof; old database untouched.
    if let Err(error) = boundary(InstallPoint::CandidatePreflightRename)
        .and_then(|()| fs::rename(&candidate, &incoming))
    {
        let _ = fs::remove_dir_all(&quarantine);
        return Err(error.into());
    }
```

Actual rename rejects cross-volume/mount candidates before marker; no copy fallback. Marker-create failure cleans only this fresh quarantine, never old ones.

- [ ] **T6.4 Publish marker, isolate the exact set, place candidate, then commit.** Continue the **same function**:

```rust
    let transaction: Result<(), LeaseError> = (|| {
        if metadata_optional(&marker)?.is_none() {
            boundary(InstallPoint::MarkerCreate)?;
            let mut file = fs::OpenOptions::new().write(true).create_new(true).open(&marker)?;
            file.write_all(MARKER)?;
            file.flush()?;
            drop(file);
        }
        // Existing incomplete marker stays until commit.
        boundary(InstallPoint::MarkerPublished)?;
        let suffixes = match backend { Backend::DuckDb => DUCKDB_SUFFIXES, Backend::LanceDb => LANCEDB_SUFFIXES };
        for (index, suffix) in suffixes.iter().enumerate() {
            let old = sibling(target, suffix);
            if metadata_optional(&old)?.is_some() {
                let filename = old.file_name().ok_or(LeaseError::Layout("associated path has no filename"))?;
                boundary(InstallPoint::Isolate(index))?;
                fs::rename(&old, quarantine.join(filename))?;
                boundary(InstallPoint::Isolated(index))?;
            }
        }
        boundary(InstallPoint::CandidateRename)?;
        fs::rename(&incoming, target)?;
        boundary(InstallPoint::CandidateRenamed)?;
        boundary(InstallPoint::MarkerRemove)?;
        fs::remove_file(&marker)?; // process-level commit point
        Ok(())
    })();
    if let Err(error) = transaction {
        // An unreadable/ambiguous marker is refusal, not permission to index locally.
        match metadata_optional(&marker) {
            Ok(None) => { let _ = fs::remove_dir_all(&quarantine); return Err(error); }
            Ok(Some(_)) | Err(_) => return Err(LeaseError::Interrupted),
        }
    }
    // Post-commit cleanup only warns.
    let cleanup = boundary(InstallPoint::Committed)
        .and_then(|()| boundary(InstallPoint::QuarantineCleanup))
        .and_then(|()| fs::remove_dir_all(&quarantine));
    if cleanup.is_err() { return Ok(vec!["quarantine_cleanup_failed".into()]); }
    Ok(Vec::new())
}

#[cfg(test)]
#[path = "install_tests.rs"]
mod tests;
```

No marker cleanup paths, globs, quarantine restore or symlink following. Append suffixes. Abandoned quarantines are manual cleanup only.

- [ ] **T6.5 PyO3 install.** Declare `pub(crate) mod install;`. Insert in existing InstallLease pymethods block (multiple-pymethods disabled):

```rust
fn install(&mut self, py: Python<'_>, candidate: String) -> PyResult<Vec<String>> {
    let candidate = PathBuf::from(candidate);
    py.allow_threads(move || {
        let lease = self.held.as_ref().ok_or(LeaseError::Layout("lease is closed"))?;
        super::install::install(lease, self.backend, &candidate)
    }).map_err(Into::into)
}
```

The `&mut self` PyO3 borrow prevents concurrent install/close/inspection on this handle while the GIL is released. Remove T1's temporary backend-field dead-code allowance. Parent waits for completion before closing handle/stage.

- [ ] **T6.6 Cover deterministic ordinary filesystem failures.** Start `install_tests.rs` with these table-driven real-fs tests:

```rust
use super::*;
use crate::database_coordination::lease::{acquire_exclusive, acquire_shared};
use std::time::{Duration, Instant};

#[test]
fn filesystem_failures_preserve_old_or_refuse_open_and_cleanup_is_only_warning() {
    let cases = [
        (InstallPoint::CandidatePreflightRename, false, false),
        (InstallPoint::MarkerCreate, false, false),
        (InstallPoint::Isolate(0), true, false),
        (InstallPoint::Isolate(1), true, false),
        (InstallPoint::CandidateRename, true, false),
        (InstallPoint::MarkerRemove, true, false),
        (InstallPoint::QuarantineCleanup, false, true),
    ];
    for (point, interrupted, committed) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("chunks.db");
        let stage = tmp.path().join("stage");
        fs::create_dir(&stage).unwrap();
        let candidate = stage.join("chunks.db");
        fs::write(&target, b"old").unwrap();
        fs::write(sibling(&target, ".wal"), b"old wal").unwrap();
        fs::write(&candidate, b"new").unwrap();
        let lease = acquire_exclusive(&target, Backend::DuckDb).unwrap();
        let result = install_with(&lease, Backend::DuckDb, &candidate, &mut |at| {
            if at == point { Err(io::Error::new(io::ErrorKind::PermissionDenied, "injected")) } else { Ok(()) }
        });
        if committed {
            assert_eq!(result.unwrap(), vec!["quarantine_cleanup_failed"]);
            assert_eq!(fs::read(&target).unwrap(), b"new");
            assert!(!sibling(&target, ".wal").exists());
        } else if interrupted {
            assert!(matches!(result, Err(LeaseError::Interrupted)));
        } else {
            assert!(matches!(result, Err(LeaseError::Filesystem(_))));
            assert_eq!(fs::read(&target).unwrap(), b"old");
            assert_eq!(fs::read(sibling(&target, ".wal")).unwrap(), b"old wal");
        }
        assert_eq!(lease.interrupted().unwrap(), interrupted);
        drop(lease);
        assert_eq!(matches!(acquire_shared(&target), Err(LeaseError::Interrupted)), interrupted);
    }
}
```

Preflight rename rejects other filesystems; no copy fallback or assumed /tmp topology.

- [ ] **T6.7 Kill tests.** install_tests.rs re-enters Rust test binary. Env hooks only in cfg(test):

```rust
#[test]
#[ignore = "subprocess entry point, invoked by kill tests"]
fn installer_child() {
    let target = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_TARGET").unwrap());
    let candidate = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_CANDIDATE").unwrap());
    let ready = PathBuf::from(std::env::var_os("CH_TEST_INSTALL_READY").unwrap());
    let phase = std::env::var("CH_TEST_INSTALL_PHASE").unwrap();
    let backend = Backend::parse(&std::env::var("CH_TEST_INSTALL_BACKEND").unwrap()).unwrap();
    let point = match phase.as_str() {
        "marker" => InstallPoint::MarkerPublished,
        "isolate" => InstallPoint::Isolated(0),
        "rename" => InstallPoint::CandidateRenamed,
        "commit" => InstallPoint::Committed,
        _ => panic!("unknown test phase"),
    };
    let lease = acquire_exclusive(&target, backend).unwrap();
    install_with(&lease, backend, &candidate, &mut |at| {
        if at == point {
            fs::write(&ready, b"ready")?;
            loop { std::thread::park(); }
        }
        Ok(())
    }).unwrap();
}

#[test]
fn killed_installer_refuses_partial_state_and_fresh_repull_recovers() {
    for backend in [Backend::DuckDb, Backend::LanceDb] {
        for phase in ["marker", "isolate", "rename", "commit"] {
            let tmp = tempfile::tempdir().unwrap();
            let target = tmp.path().join(backend.filename());
            let stage = tmp.path().join("stage");
            fs::create_dir(&stage).unwrap();
            let candidate = stage.join(backend.filename());
            for (path, bytes) in [(&target, b"old".as_slice()), (&candidate, b"new".as_slice())] {
                match backend {
                    Backend::DuckDb => fs::write(path, bytes).unwrap(),
                    Backend::LanceDb => { fs::create_dir(path).unwrap(); fs::write(path.join("payload"), bytes).unwrap(); }
                }
            }
            let neighbor = tmp.path().join("notes.txt");
            fs::write(&neighbor, b"keep").unwrap();
            let ready = tmp.path().join("ready");
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["database_coordination::install::tests::installer_child", "--exact", "--ignored", "--nocapture"])
                .env("CH_TEST_INSTALL_TARGET", &target)
                .env("CH_TEST_INSTALL_CANDIDATE", &candidate)
                .env("CH_TEST_INSTALL_READY", &ready)
                .env("CH_TEST_INSTALL_PHASE", phase)
                .env("CH_TEST_INSTALL_BACKEND", if backend == Backend::DuckDb { "duckdb" } else { "lancedb" })
                .spawn().unwrap();
            let deadline = Instant::now() + Duration::from_secs(20);
            while !ready.exists() && Instant::now() < deadline {
                if child.try_wait().unwrap().is_some() { break; }
                std::thread::sleep(Duration::from_millis(10));
            }
            let reached = ready.exists();
            if reached {
                // A second pull and a cold opener both decline while installation
                // is paused, even after commit but before the owner releases.
                assert!(matches!(acquire_exclusive(&target, backend), Err(LeaseError::Busy)));
                assert!(matches!(acquire_shared(&target), Err(LeaseError::Busy)));
            }
            let _ = child.kill();
            child.wait().unwrap();
            assert!(reached, "installer did not reach {phase}");
            assert_eq!(fs::read(&neighbor).unwrap(), b"keep");
            if phase != "commit" {
                assert!(matches!(acquire_shared(&target), Err(LeaseError::Interrupted)));
            } else {
                drop(acquire_shared(&target).unwrap());
                let payload = if backend == Backend::DuckDb { target.clone() } else { target.join("payload") };
                assert_eq!(fs::read(payload).unwrap(), b"new");
            }
            // Fresh candidate; never restore/reuse data from abandoned quarantine.
            let fresh_parent = tmp.path().join("fresh");
            fs::create_dir(&fresh_parent).unwrap();
            let fresh = fresh_parent.join(backend.filename());
            if backend == Backend::DuckDb { fs::write(&fresh, b"recovered").unwrap(); }
            else { fs::create_dir(&fresh).unwrap(); fs::write(fresh.join("payload"), b"recovered").unwrap(); }
            let recovery = acquire_exclusive(&target, backend).unwrap();
            assert!(install(&recovery, backend, &fresh).unwrap().is_empty());
            drop(recovery);
            drop(acquire_shared(&target).unwrap());
            let payload = if backend == Backend::DuckDb { target.clone() } else { target.join("payload") };
            assert_eq!(fs::read(payload).unwrap(), b"recovered");
            assert_eq!(fs::read(&neighbor).unwrap(), b"keep");
        }
    }
}
```

Parent additionally tests explicit CLI re-pull/search with real backends.

- [ ] **T6.8 Verify.** `cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test` => PASS (child entry ignored except subprocess). Rebuild; `UV_NO_SYNC=1 uv run pytest tests/hub/test_native_install.py tests/hub/test_native_leases.py tests/hub/test_native_pipeline_ownership.py -q` => PASS. Existing Ubuntu/macOS/Windows Rust matrix must pass before cross-platform claim.
- [ ] **T6.9 Smoke/commit:** `UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto` => PASS; stage T6 files, `git commit -m "feat: install verified hub databases under native leases"`. T7 follows this installation gate.

---

### Task T7: Install verified candidates with safe Python orchestration

**Files:** Create `chunkhound/hub/install.py`, `chunkhound/hub/client.py`, `tests/hub/activation_support.py`, `tests/hub/test_activation_client.py`; modify `chunkhound/hub/{errors,artifact}.py`, `tests/hub/{artifacts,conftest}.py`.

**Consumes:** All fixed dependency interfaces above; existing `prepare`, `configured_key`, `select_manifest`, `fetch_manifest`, `resolve_indexed_model`, `active_model_path`, `read_active_model`, `build_database`, `pack_database`, `hub_http`.

**Produces:**

```python
@dataclass
class PreparationState:
    committed: bool = False
prepare(selected: Selected, work_parent: Path, *,
        state: PreparationState | None = None) -> AsyncIterator[Path]
class HubInstallationInterrupted(RuntimeError):
    """Fatal incomplete installation; never ordinary fallback."""
installation_target(config: Config) -> Path
installation_failure(exc: Exception) -> HubUnavailable | HubInstallationInterrupted
install_candidate(candidate: Path, target: Path, backend: str,
                  state: PreparationState, *,
                  should_install: Callable[[DatabaseOwnership], bool] | None = None) -> bool
local_needs_pull(target: Path, key: MatchKey, *,
                 ownership: DatabaseOwnership | None = None) -> bool
async def pull(config: Config, rel: str = "latest", automatic: bool = False) -> bool
```

`pull`: True only if this invocation committed an install; False for skipped/soft unavailable. Fatal mismatch/interruption are logged once then raised, not returned False. Explicit CLI maps False and fatal categories to exit 1. Callable recheck is supplied only for automatic mode; the adapter rejects interruption before invoking it.

**Approved T7 I1 amendment:** After fully draining actual worker completion, fatal hub mismatch/interruption takes precedence over concurrent cancellation, including native interruption from initial inspection. A successfully completed worker still reports cancellation; soft failure behavior is unchanged. Keep the shared T5 `await_worker_completion` helper unchanged and apply this outcome selection locally to both T7 worker boundaries. Retain private executor completion Futures with copied contextvars, not independently cancellable `to_thread` Tasks. Neither repeated cancellation nor `asyncio.run` shutdown may release leases or staging before the actual worker finishes. Add a runtime RED/GREEN regression using real native interruption at the initial inspection and installation/recheck boundaries, asserting one sanitized fatal diagnostic, marker retention and completed drainage/staging/lease lifetime; retain successful-worker cancellation coverage.

- [ ] **T7.1 Add fixture support and failing real installation contracts.** In `tests/hub/artifacts.py`, extend `build_database` with trailing `symbol_prefix: str = "function"`; replace only its chunk code and symbol expressions:

```python
code=f"def {symbol_prefix}_{i}(): return {i}",
symbol=f"{symbol_prefix}_{i}",
```

In `tests/hub/conftest.py`, add `before_get=None` to `hub_server`'s `SimpleNamespace`. Immediately after recording `state.requests` in `do_GET` add:

```python
if state.before_get is not None:
    state.before_get(self.path)
```

This is an HTTP service boundary for deterministic concurrent downloads, not an installer mock. Create `tests/hub/activation_support.py`:

```python
import json
import os
import subprocess
from pathlib import Path

from chunkhound.core.config.config import Config
from tests.hub.artifacts import pack_database

ROOT = Path(__file__).resolve().parents[2]


def config_for(project, hub, backend="duckdb", model="hub-test"):
    project.mkdir(parents=True, exist_ok=True)
    return Config(
        target_dir=project,
        database={"path": project / ".chunkhound" / "db", "provider": backend},
        embedding={"provider": "openai", "model": model, "output_dims": 8,
                   "api_key": "test", "base_url": "http://127.0.0.1:1/v1"},
        hub={"url": hub.url + "/catalogue?token=secret"},
        skip_layers={"global", "env", "local_config", "config_file", "cli"},
    )


def publish(hub, path, *, rel="latest", model="hub-test", prefix="base"):
    archive = path.parent / f"{prefix}.tar.gz"
    metadata = pack_database(path, archive)
    route = f"/{prefix}.tar.gz?token=secret"
    hub.resources[route] = archive.read_bytes()
    variant = {
        "embedding_id": "test-base", "db_provider":
        "duckdb" if path.name == "chunks.db" else "lancedb",
        "provider": "openai", "model": model, "dims": 8,
        "index_format_version": 1,
        "versions": [{"rel": rel, "artifact":
                      {"url": hub.url + route, **metadata}}],
    }
    manifest = {"manifest_version": 2, "project_id": "fixture-project",
                "embeddings": [variant]}
    hub.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    return manifest, route


def cli_env(project, rust="0"):
    global_config = project.parent / "empty-global.json"
    global_config.write_text("{}", encoding="utf-8")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CHUNKHOUND_", "OPENAI_", "VOYAGE_"))}
    env.update(UV_NO_SYNC="1", UV_PROJECT=str(ROOT),
               CHUNKHOUND_GLOBAL_CONFIG_FILE=str(global_config),
               CHUNKHOUND_USE_RUST=rust, CHUNKHOUND_NO_PROMPTS="1",
               CHUNKHOUND_NO_MODEL_SUGGESTIONS="1", CHUNKHOUND_NO_RICH="1",
               NO_COLOR="1")
    return env


def cli(project, *args, rust="0"):
    return subprocess.run(
        ["uv", "run", "chunkhound", *map(str, args)], cwd=project,
        env=cli_env(project, rust), text=True, capture_output=True, timeout=180,
    )


def write_config(project, backend, hub_url, embedding_url, model="hub-test"):
    project.mkdir(parents=True, exist_ok=True)
    data = {
        "database": {"provider": backend,
                     "path": str(project / ".chunkhound" / "db")},
        "embedding": {"provider": "openai", "model": model,
                      "output_dims": 8, "api_key": "test", "batch_size": 100,
                      "base_url": embedding_url},
        "indexing": {"include": ["*.py"], "exclude": [".chunkhound/**"]},
        "analytics": {"enabled": False},
        "hub": {"url": hub_url},
    }
    (project / ".chunkhound.json").write_text(json.dumps(data), encoding="utf-8")
    return data
```

Create `tests/hub/test_activation_client.py` with the following tests/imports. Snapshot only database/sidecar data, not the persistent coordination file: creating that file is expected, not index mutation. `read_active_model` input remains under shared/exclusive ownership in production.

```python
import asyncio
import json
import shutil
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

import chunkhound_native as native
import pytest
from loguru import logger

from chunkhound.hub.client import pull
from chunkhound.hub.errors import HubDatabaseMismatch, HubInstallationInterrupted
from chunkhound.hub.inspect import inspect_database
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.providers.database.lancedb_provider import LanceDBProvider
from tests.hub.activation_support import config_for, publish
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_pull_installs_replaces_and_is_searchable(tmp_path, hub_http, backend):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = config.database.get_db_path()
    neighbor = target.parent / "notes.txt"
    neighbor.write_text("operator-owned")
    for prefix in ("first", "replacement"):
        source = build_database(tmp_path / prefix, backend, symbol_prefix=prefix)
        publish(hub_http, source, prefix=prefix)
        if prefix == "replacement":
            target.with_name(target.name + ".embedding.json").write_text("old active model")
            if backend == "duckdb":
                target.with_name(target.name + ".root.json").write_text("old publisher root")
        assert await pull(config)
        assert not target.with_name(target.name + ".embedding.json").exists()
        if backend == "duckdb":
            assert not target.with_name(target.name + ".root.json").exists()
        cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
        db = cls(target, base_directory=config.target_dir)
        db.connect()
        try:
            rows, _ = db.search_regex(prefix + "_0")
            assert rows
            if prefix == "replacement":
                rows, _ = db.search_regex("first_0")
                assert rows == []
        finally:
            db.disconnect()
    assert neighbor.read_text() == "operator-owned"
    assert not list(target.parent.glob(".chunkhound-pull-*"))


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_downloaded_mismatch_preserves_existing_target_and_sidecars(
    tmp_path, hub_http, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = build_database(config.database.path, backend)
    sidecars = [target.with_name(target.name + suffix)
                for suffix in (".root.json", ".embedding.json")]
    for path in sidecars:
        path.write_text("local state must survive")
    before = {path: snapshot(path) for path in [target, *sidecars]}
    source = build_database(tmp_path / "wrong", backend, model="different")
    publish(hub_http, source)  # Manifest intentionally claims hub-test.
    with pytest.raises(HubDatabaseMismatch):
        await pull(config)
    assert {path: snapshot(path) for path in before} == before


@pytest.mark.parametrize("failure", ["http", "checksum", "archive", "space"])
async def test_preinstall_failures_preserve_local_index(
    tmp_path, hub_http, monkeypatch, failure
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = build_database(config.database.path)
    before = snapshot(target)
    source = build_database(tmp_path / "published")
    manifest, route = publish(hub_http, source)
    if failure == "http":
        hub_http.status = 503
    elif failure == "checksum":
        manifest["embeddings"][0]["versions"][0]["artifact"]["sha256"] = "0" * 64
        hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    elif failure == "archive":
        import hashlib
        body = b"not a gzip archive"
        hub_http.resources[route] = body
        artifact = manifest["embeddings"][0]["versions"][0]["artifact"]
        artifact.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    else:
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(shutil, "disk_usage", lambda _: usage._replace(free=0))
    assert not await pull(config)
    assert snapshot(target) == before


@pytest.mark.parametrize("automatic", [False, True])
async def test_busy_pull_does_not_replace_or_open_partial_state(
    tmp_path, hub_http, automatic
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    # A live cooperative holder can own an as-yet absent file.
    held = native.acquire_database_lease(str(target))
    try:
        assert not await pull(config, automatic=automatic)
        assert not target.exists()
    finally:
        held.close()
    assert await pull(config)


async def test_auto_marker_is_fatal_without_requests_but_explicit_recovers(
    tmp_path, hub_http
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    marker = target.with_name(target.name + ".chunkhound-install.json")
    marker.write_bytes(b"{")  # Incomplete marker also means interrupted.
    publish(hub_http, build_database(tmp_path / "published"))
    with pytest.raises(HubInstallationInterrupted):
        await pull(config, automatic=True)
    assert hub_http.requests == []
    assert marker.read_bytes() == b"{"
    assert await pull(config)
    assert not marker.exists()
    assert inspect_database(target, "duckdb").groups


@pytest.mark.parametrize("racing_marker", [False, True])
async def test_auto_rechecks_under_exclusive_ownership(
    tmp_path, hub_http, racing_marker
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    competitor = build_database(tmp_path / "competitor", symbol_prefix="race_winner")
    reached, release = threading.Event(), threading.Event()

    def delay_artifact(route):
        if route.startswith("/base.tar.gz"):
            reached.set()
            assert release.wait(20)

    hub_http.before_get = delay_artifact
    task = asyncio.create_task(pull(config, automatic=True))
    try:
        assert await asyncio.to_thread(reached.wait, 20)
        lease = native.acquire_install_lease(str(target), "duckdb")
        try:
            lease.install(str(competitor))
        finally:
            lease.close()
        before = snapshot(target)
        if racing_marker:
            target.with_name(target.name + ".chunkhound-install.json").write_bytes(b"{")
        release.set()
        if racing_marker:
            with pytest.raises(HubInstallationInterrupted):
                await task
        else:
            assert not await task  # Aligned race winner must not be replaced.
        assert snapshot(target) == before
    finally:
        release.set()
        if not task.done():
            await task


async def test_committed_pull_cleanup_failure_is_only_a_warning(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    publish(hub_http, build_database(tmp_path / "published"))
    original = TemporaryDirectory.cleanup
    messages = []
    sink = logger.add(lambda record: messages.append(str(record)))

    def failed_cleanup(workspace):
        if Path(workspace.name).name.startswith(".chunkhound-pull-"):
            raise OSError("private-path?secret")
        return original(workspace)

    monkeypatch.setattr(TemporaryDirectory, "cleanup", failed_cleanup)
    try:
        assert await pull(config)
    finally:
        logger.remove(sink)
    assert inspect_database(config.database.get_db_path(), "duckdb").groups
    text = "".join(messages)
    assert "staging cleanup" in text.lower()
    assert "private-path" not in text and "?secret" not in text


async def test_repeated_cancellation_waits_for_install_and_workspace_cleanup(
    tmp_path, hub_http, monkeypatch
):
    config = config_for(tmp_path / "checkout", hub_http)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published"))
    downloaded, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_exists = Path.exists

    def block_exclusive_recheck(path):
        # Deterministic filesystem boundary, after initial trigger and download.
        if path == target and downloaded.is_set() and not entered.is_set():
            entered.set()
            assert release.wait(20)
        return original_exists(path)

    def mark_download(route):
        if route.startswith("/base.tar.gz"):
            downloaded.set()

    monkeypatch.setattr(Path, "exists", block_exclusive_recheck)
    hub_http.before_get = mark_download
    task = asyncio.create_task(pull(config, automatic=True))
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert list(target.parent.glob(".chunkhound-pull-*"))
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(target), "duckdb")
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert target.exists()
    assert not list(target.parent.glob(".chunkhound-pull-*"))
    assert inspect_database(target, "duckdb").groups
```

Add the configured-layout rejection contract to this file (no network, including a legacy file explicitly named `chunks.db`):

```python
@pytest.mark.parametrize("layout", ["memory", "read_only", "explicit", "legacy"])
async def test_unsupported_install_layout_makes_no_request(tmp_path, hub_http, layout):
    config = config_for(tmp_path / "checkout", hub_http)
    if layout == "memory":
        config.database.path = Path(":memory:")
    elif layout == "read_only":
        config.database.read_only = True
    elif layout == "explicit":
        config.database.path = config.target_dir / "chunks.db"
    else:
        legacy = config.target_dir / "legacy-file"
        legacy.write_bytes(b"legacy")
        config.database.path = legacy
    assert not await pull(config)
    assert hub_http.requests == []


@pytest.mark.parametrize("automatic", [False, True])
async def test_unknown_identity_does_not_probe_services(
    tmp_path, hub_http, automatic
):
    config = config_for(tmp_path / "checkout", hub_http, model="unknown-offline-model")
    config.embedding.output_dims = None
    assert not await pull(config, automatic=automatic)
    assert hub_http.requests == []
```

Append two-pull/cold-opener and failed-recovery-download contracts to `tests/hub/test_activation_client.py` before RED:

```python
@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_two_pulls_and_cold_provider_never_observe_partial_install(
    tmp_path, hub_http, monkeypatch, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = config.database.get_db_path()
    publish(hub_http, build_database(tmp_path / "published", backend))
    downloaded, entered, release = threading.Event(), threading.Event(), threading.Event()
    original_exists = Path.exists

    def mark_download(route):
        if route.startswith("/base.tar.gz"):
            downloaded.set()

    def pause_exclusive_recheck(path):
        if path == target and downloaded.is_set() and not entered.is_set():
            entered.set()
            assert release.wait(20)
        return original_exists(path)

    hub_http.before_get = mark_download
    monkeypatch.setattr(Path, "exists", pause_exclusive_recheck)
    first = asyncio.create_task(pull(config, automatic=True))
    cls = DuckDBProvider if backend == "duckdb" else LanceDBProvider
    opener = cls(target, base_directory=config.target_dir)
    try:
        assert await asyncio.to_thread(entered.wait, 20)
        # Second explicit pull downloads/verifies, but cannot install over first.
        assert not await pull(config)
        with pytest.raises(native.DatabaseBusyError):
            opener.connect()
        assert not target.exists()
    finally:
        opener.close()
        release.set()
        await asyncio.gather(first, return_exceptions=True)
    assert first.result() is True
    assert inspect_database(target, backend).groups


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_failed_explicit_redownload_preserves_interruption_and_partial_state(
    tmp_path, hub_http, backend
):
    config = config_for(tmp_path / "checkout", hub_http, backend)
    target = build_database(config.database.path, backend)
    marker = Path(str(target) + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    before = snapshot(target.parent)
    publish(hub_http, build_database(tmp_path / "published", backend))
    hub_http.status = 503
    assert not await pull(config)
    assert snapshot(target.parent) == before
    with pytest.raises(native.DatabaseInterruptedError):
        inspect_database(target, backend)
```

The delay is at the real filesystem inspection boundary while native exclusive ownership is held. T6 separately pauses actual marker/isolation/rename/commit syscalls and probes both opener and second-installer exclusion. These tests are complementary, not mocked installation evidence.

- [ ] **T7.2 RED.** Run `UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py -v`. Expected failure: new client/fatal error exports absent. Do not relax preservation/search assertions to obtain green.

- [ ] **T7.3 Add a narrowly scoped staging commit state.** In `chunkhound/hub/errors.py` add:

```python
class HubInstallationInterrupted(RuntimeError):  # noqa: N818 - public hub category
    """Installation may be incomplete; ordinary opens and automatic fallback stop."""
```

In `chunkhound/hub/artifact.py`, import `dataclass` from `dataclasses`, add the following immediately above `prepare`, and change only its signature/final cleanup branch. Existing extraction/verification and its fatal-versus-soft categories stay unchanged.

```python
@dataclass
class PreparationState:
    """Set by the completed install worker, never inferred from candidate existence."""

    committed: bool = False


@asynccontextmanager
async def prepare(
    selected: Selected,
    work_parent: Path,
    *,
    state: PreparationState | None = None,
) -> AsyncIterator[Path]:
```

Replace the `except OSError:` body immediately surrounding `workspace.cleanup()` with:

```python
if state is not None and state.committed:
    logger.bind(hub_client=True).warning(
        "Hub base installed; staging cleanup could not be completed"
    )
elif failed:
    # Preserve a caller failure/fatal mismatch/cancellation category.
    logger.bind(hub_client=True).error("Hub staging workspace could not be removed")
else:
    raise HubUnavailable("Hub staging workspace could not be removed") from None
```

Default `state=None` preserves every foundation caller/test's existing cleanup behavior. A consumed candidate is not proof of commit. State is set synchronously inside the installer worker immediately after successful native return, even if its awaiter has been cancelled. The completion helper guarantees the workspace outlives that worker.

- [ ] **T7.4 Implement the thin installation adapter.** Complete `chunkhound/hub/install.py`:

```python
"""Configured-layout boundary and scoped native installation; no filesystem protocol."""

from collections.abc import Callable
from pathlib import Path

import chunkhound_native as native
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.hub.artifact import PreparationState
from chunkhound.hub.errors import HubInstallationInterrupted, HubUnavailable
from chunkhound.providers.database.ownership import DatabaseOwnership


def installation_target(config: Config) -> Path:
    database = config.database
    configured = database.path
    if configured is None or str(configured) == ":memory:" or database.read_only:
        raise HubUnavailable("Hub pull requires a writable directory-layout database")
    try:
        target = database.get_db_path()
    except (OSError, ValueError):
        raise HubUnavailable("Hub database destination is unavailable") from None
    # get_db_path is authoritative, including legacy files and .db suffix rules.
    # The configured single-file chunks.db is NOT a directory-layout install.
    if target == configured or target.parent != configured:
        raise HubUnavailable("Hub pull does not support configured database file paths")
    expected = "chunks.db" if database.provider == "duckdb" else "lancedb.lancedb"
    if target.name != expected:
        raise HubUnavailable("Hub database layout is unsupported")
    return target


def installation_failure(exc: Exception) -> HubUnavailable | HubInstallationInterrupted:
    if isinstance(exc, native.DatabaseInterruptedError):
        return HubInstallationInterrupted(
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing"
        )
    if isinstance(exc, native.DatabaseBusyError):
        return HubUnavailable("Hub destination is busy; stop database users and retry")
    if isinstance(exc, native.DatabaseLayoutError):
        return HubUnavailable("Hub database installation layout is unsupported")
    return HubUnavailable("Hub installation filesystem is unavailable")


def install_candidate(
    candidate: Path,
    target: Path,
    backend: str,
    state: PreparationState,
    *,
    should_install: Callable[[DatabaseOwnership], bool] | None = None,
) -> bool:
    try:
        lease = native.acquire_install_lease(str(target), backend)
        with DatabaseOwnership(lease) as ownership:
            if should_install is not None:
                if lease.interrupted:
                    raise native.DatabaseInterruptedError()
                if not should_install(ownership):
                    return False
            warnings = lease.install(str(candidate))
            state.committed = True
            for code in warnings:
                if code == "quarantine_cleanup_failed":
                    logger.bind(hub_client=True).warning(
                        "Hub base installed; old quarantine cleanup could not be completed"
                    )
                else:
                    logger.bind(hub_client=True).warning(
                        "Hub base installed with a cleanup warning"
                    )
            return True
    except (native.DatabaseBusyError, native.DatabaseInterruptedError,
            native.DatabaseLayoutError, native.DatabaseFilesystemError) as exc:
        raise installation_failure(exc) from None
```

Only this adapter deals with exclusive native handles. It never removes sidecars, markers or quarantines itself; it never calls a lease close while installation work continues.

- [ ] **T7.5 Implement policy and cancellation-safe orchestration.** Complete `chunkhound/hub/client.py`:

```python
"""Trusted hub selection and automatic-index policy, independent of registry."""

import asyncio
import contextvars
from pathlib import Path

import chunkhound_native as native
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.embedding_model_drift import (
    active_model_path,
    read_active_model,
    resolve_indexed_model,
)
from chunkhound.hub.artifact import PreparationState, prepare
from chunkhound.hub.errors import (
    HubDatabaseMismatch,
    HubInstallationInterrupted,
    HubUnavailable,
    safe_label,
)
from chunkhound.hub.inspect import inspect_database
from chunkhound.hub.install import (
    installation_failure,
    installation_target,
    install_candidate,
)
from chunkhound.hub.manifest import MatchKey, configured_key, select_manifest
from chunkhound.hub.transport import fetch_manifest
from chunkhound.providers.database.ownership import DatabaseOwnership, ownership_scope
from chunkhound.utils.async_worker import await_worker_completion

_NATIVE_ERRORS = (native.DatabaseBusyError, native.DatabaseInterruptedError,
                  native.DatabaseLayoutError, native.DatabaseFilesystemError)


def local_needs_pull(
    target: Path,
    key: MatchKey,
    *,
    ownership: DatabaseOwnership | None = None,
) -> bool:
    # Acquire before existence test: an absent target plus a marker is NOT cold.
    with ownership_scope(target, ownership) as held:
        if not target.exists():
            return True
        try:
            inventory = inspect_database(target, key.db_provider, ownership=held)
            sidecar = active_model_path(target)
            active = read_active_model(sidecar) if sidecar is not None else None
            indexed = resolve_indexed_model(inventory.groups, active)
        except _NATIVE_ERRORS:
            raise
        except Exception:
            # Unreadable/empty local state is not evidence of embedding drift.
            logger.bind(hub_client=True).info(
                "Hub auto-pull skipped: local embedding identity is unreadable"
            )
            return False
        return indexed is not None and (
            indexed.provider, indexed.model, indexed.dims
        ) != (key.provider, key.model, key.dims)


async def _await_pull_worker(worker: asyncio.Future[bool]) -> bool:
    """Drain actual completion, preserving fatal hub outcomes over cancellation."""
    try:
        return await await_worker_completion(worker)
    except asyncio.CancelledError:
        # The shared helper has fully drained this private executor Future.
        # Keep its cancellation policy except when recovery/fatal reporting is due.
        failure = worker.exception()
        if isinstance(
            failure,
            (
                HubDatabaseMismatch,
                HubInstallationInterrupted,
                native.DatabaseInterruptedError,
            ),
        ):
            raise failure from None
        raise


async def _pull(config: Config, rel: str, automatic: bool) -> bool:
    if automatic and (config.embeddings_disabled or not config.hub.url):
        return False
    if not config.hub.url:
        raise HubUnavailable("No hub manifest URL configured")
    key = configured_key(config)  # Offline; do not let drift pinning run first.
    target = installation_target(config)
    if automatic:
        # Keep actual executor completion private: shutdown cancels Tasks, not threads.
        worker = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, local_needs_pull, target, key
        )
        if not await _await_pull_worker(worker):
            return False
    selected = select_manifest(await fetch_manifest(config.hub.url), key, rel)
    logger.bind(hub_client=True).info(
        "Pulling hub base project={} identity={} embedding={} version={} revision={} built={} index_root={} artifact=index.tar.gz bytes={}",
        safe_label(selected.manifest.project_id), safe_label(selected.variant.key()),
        safe_label(selected.variant.embedding_id), rel,
        safe_label(selected.version.revision), safe_label(selected.version.created_at),
        safe_label(selected.manifest.index_root), selected.version.artifact.bytes,
    )
    state = PreparationState()
    async with prepare(selected, work_parent=target.parent, state=state) as candidate:
        def commit() -> bool:
            return install_candidate(
                candidate, target, key.db_provider, state,
                should_install=(lambda ownership: local_needs_pull(
                    target, key, ownership=ownership
                )) if automatic else None,
            )

        worker = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, commit
        )
        installed = await _await_pull_worker(worker)
    if installed:
        logger.bind(hub_client=True).info(
            "Hub base installed; run index to overlay local changes"
        )
    return installed


async def pull(config: Config, rel: str = "latest", automatic: bool = False) -> bool:
    try:
        try:
            return await _pull(config, rel, automatic)
        except _NATIVE_ERRORS as exc:
            # Initial ordinary inspection can fail before install.py is entered.
            raise installation_failure(exc) from None
        except OSError:
            raise HubUnavailable("Hub local filesystem is unavailable") from None
    except HubDatabaseMismatch:
        logger.bind(hub_client=True).error(
            "Downloaded database contradicts the hub manifest; local index retained"
        )
        raise
    except HubInstallationInterrupted:
        logger.bind(hub_client=True).error(
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing"
        )
        raise
    except HubUnavailable as exc:
        logger.bind(hub_client=True).log(
            "INFO" if automatic else "ERROR", "Hub pull unavailable: {}{}",
            str(exc), "; continuing local indexing" if automatic else "",
        )
        return False
```

The broad local inventory catch is only a **skip** before replacement, not permission to overwrite an unreadable database. All native interruption categories pass through it. Interrupted installation is never caught by the auto soft branch. Native pre-marker failures may allow local fallback; errors with a published marker may not.

- [ ] **T7.6 GREEN and commit checkpoint.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_client.py tests/hub/test_artifact.py tests/hub/test_inspection_ownership.py tests/hub/test_worker_ownership.py -v
UV_NO_SYNC=1 uv run ruff check chunkhound/hub tests/hub/activation_support.py tests/hub/test_activation_client.py
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/activation-client-c1-mypy.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
git diff --check
git add chunkhound/hub/install.py chunkhound/hub/client.py chunkhound/hub/errors.py chunkhound/hub/artifact.py tests/hub/activation_support.py tests/hub/test_activation_client.py tests/hub/artifacts.py tests/hub/conftest.py
git commit -m "feat: install verified hub bases through native ownership"
```

Expected targeted/smoke PASS, scoped lint clean, no new mypy errors relative to preamble. Review T7's worker/commit/fatal classification before dispatching T8.

---

---

### Task T8: Explicit pull and automatic index before registry/drift pinning

**Files:** Create `chunkhound/api/cli/parsers/pull_parser.py`, `chunkhound/api/cli/commands/pull.py`, `tests/hub/test_activation_cli.py`; modify `chunkhound/api/cli/{main.py,commands/run.py,utils/rich_output.py}`, `tests/hub/conftest.py`, `tests/test_smoke.py`.

**Consumes:** T7 `pull(config, rel="latest", automatic=False) -> bool`, fatal categories, all support functions. Current `run_command` returns early for simulate/check-ignores, then validates run arguments, then sets `drift_seen` and calls `configure_registry`. Config already rejects read-only for pull; missing/unknown identity is handled by `configured_key`. No new shared-config behavior/credential exception is needed.

**Produces:** `add_pull_subparser(subparsers: Any) -> argparse.ArgumentParser`, `pull_command(args: argparse.Namespace, config: Config) -> None`; command `pull [path] --version latest|-1|-2`, exits 0/1/argparse 2; auto only in index before registry, no model pin before selection.

- [ ] **T8.1 Add the embedding-service boundary and failing CLI contracts.** In `tests/hub/conftest.py`, import `json` and add this fixture. It serves real Python OpenAI SDK and native HTTP calls; no provider, registry, parser, install or backend fake. It returns float vectors even if SDK requests base64: the SDK accepts list-valued embeddings. Model/dims are explicit and all requests stay loopback.

```python
@pytest.fixture
def embedding_http():
    state = SimpleNamespace(requests=[], url="")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_POST(self):  # noqa: N802 - stdlib handler
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append(body)
            if self.path != "/v1/embeddings":
                self.send_error(404)
                return
            texts = body["input"]
            if isinstance(texts, str):
                texts = [texts]
            dims = body.get("dimensions", 8)
            payload = {"object": "list", "model": body["model"],
                       "data": [{"object": "embedding", "index": i,
                                 "embedding": [0.25] * dims}
                                for i in range(len(texts))],
                       "usage": {"prompt_tokens": len(texts),
                                 "total_tokens": len(texts)}}
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
```

Create `tests/hub/test_activation_cli.py`:

```python
import hashlib
import json
import subprocess

import chunkhound_native as native
import pytest

from chunkhound.hub.inspect import inspect_database
from tests.hub.activation_support import cli, cli_env, publish, write_config
from tests.hub.artifacts import build_database
from tests.hub.test_index_format import snapshot


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rel", ["latest", "-1", "-2"])
def test_explicit_relative_version_is_installed_and_searchable(
    tmp_path, hub_http, embedding_http, backend, rel
):
    project = tmp_path / "checkout"
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    names = {"latest": "newest", "-1": "previous", "-2": "oldest"}
    versions = []
    for label, symbol in names.items():
        source = build_database(tmp_path / symbol, backend, symbol_prefix=symbol)
        manifest, _ = publish(hub_http, source, rel=label, prefix=symbol)
        versions.extend(manifest["embeddings"][0]["versions"])
    manifest["embeddings"][0]["versions"] = versions
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    prefix = names[rel]
    result = cli(project, "pull", project, "--version", rel, "--verbose")
    assert result.returncode == 0, result.stdout + result.stderr
    search = cli(project, "search", prefix + "_0", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "source.py" in search.stdout + search.stderr
    assert "No results found" not in search.stdout + search.stderr
    assert hub_http.requests == [("/catalogue?token=secret", None),
                                 (f"/{prefix}.tar.gz?token=secret", None)]
    assert embedding_http.requests == []  # Explicit pull installs only.
    assert "token=secret" not in result.stdout + result.stderr
    assert "/catalogue?" not in result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
@pytest.mark.parametrize("initial", ["absent", "misaligned", "aligned"])
def test_auto_installs_before_drift_pinning_and_aligned_makes_no_request(
    tmp_path, hub_http, embedding_http, backend, rust, initial
):
    project = tmp_path / "checkout"
    project.mkdir()
    (project / "source.py").write_text(
        "\n".join(f"def local_{i}(): return 'local_marker_{i}'" for i in range(100)),
        encoding="utf-8",
    )
    if initial != "absent":
        old_model = "hub-test" if initial == "aligned" else "old-model"
        write_config(project, backend, None, embedding_http.url, old_model)
        old = cli(project, "index", project, rust=rust)
        assert old.returncode == 0, old.stdout + old.stderr
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published", backend))
    embedding_http.requests.clear()
    result = cli(project, "index", project, "--verbose", rust=rust)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(hub_http.requests) == (0 if initial == "aligned" else 2)
    target = project / ".chunkhound" / "db" / (
        "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    )
    assert {g.model for g in inspect_database(target, backend).groups} == {"hub-test"}
    assert all(request["model"] == "hub-test" for request in embedding_http.requests)
    if initial != "aligned":
        assert embedding_http.requests  # Normal overlay followed base install.
    assert "token=secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["--no-embeddings", "--simulate", "--check-ignores"])
def test_index_subtools_do_not_request_hub(tmp_path, hub_http, embedding_http, mode):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    (project / "source.py").write_text("def local_only(): return 1\n")
    result = cli(project, "index", project, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hub_http.requests == []
    assert embedding_http.requests == []


def test_unknown_dimensions_do_not_probe_hub_or_embedding_service(tmp_path, hub_http):
    project = tmp_path / "checkout"
    data = write_config(project, "duckdb", hub_http.url + "/catalogue?token=secret",
                        "http://127.0.0.1:1/v1", model="unknown-offline-model")
    del data["embedding"]["output_dims"]
    (project / ".chunkhound.json").write_text(json.dumps(data))
    result = cli(project, "pull", project)
    assert result.returncode == 1
    assert hub_http.requests == []
    assert "known embedding dimensions" in result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
@pytest.mark.parametrize("failure", ["http", "checksum", "archive", "space", "mismatch"])
def test_cli_failure_taxonomy_is_real_not_a_mocked_registry_assertion(
    tmp_path, hub_http, embedding_http, mode, failure
):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    (project / "local.py").write_text("def local_fallback(): return 7\n")
    source = build_database(tmp_path / "published", model=(
        "wrong-model" if failure == "mismatch" else "hub-test"
    ))
    manifest, route = publish(hub_http, source)
    if failure == "http":
        hub_http.status = 503
    elif failure == "checksum":
        manifest["embeddings"][0]["versions"][0]["artifact"]["sha256"] = "0" * 64
    elif failure == "archive":
        body = b"not an archive"
        hub_http.resources[route] = body
        manifest["embeddings"][0]["versions"][0]["artifact"].update(
            bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
        )
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    if failure == "space":
        script = """
import shutil
import sys
from pathlib import Path
from chunkhound.api.cli.main import main
original = shutil.disk_usage
def disk_usage(path):
    usage = original(path)
    return usage._replace(free=0) if Path(path).name.startswith('.chunkhound-pull-') else usage
shutil.disk_usage = disk_usage
sys.argv = ['chunkhound', *sys.argv[1:]]
main()
"""
        result = subprocess.run(
            ["uv", "run", "python", "-c", script, mode, str(project), "--verbose"],
            cwd=project, env=cli_env(project), capture_output=True, text=True, timeout=180,
        )
    else:
        result = cli(project, mode, project, "--verbose")
    fatal = mode == "pull" or failure == "mismatch"
    assert result.returncode == (1 if fatal else 0), result.stdout + result.stderr
    text = result.stdout + result.stderr
    assert "token=secret" not in text and "response-body-secret" not in text
    assert "Traceback" not in text
    target = project / ".chunkhound" / "db" / "chunks.db"
    assert target.exists() is (not fatal)
    if not fatal:
        search = cli(project, "search", "local_fallback", project, "--regex")
        assert "local.py" in search.stdout + search.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
def test_candidate_mismatch_exit_preserves_local_database_sidecars(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", None, embedding_http.url, "old-model")
    (project / "local.py").write_text("def retained_local_work(): return 7\n")
    baseline = cli(project, "index", project)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    target = project / ".chunkhound" / "db" / "chunks.db"
    before = snapshot(target.parent)
    write_config(project, "duckdb", hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published", model="contradiction"))
    result = cli(project, mode, project, "--verbose")
    assert result.returncode == 1
    assert snapshot(target.parent) == before
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
def test_preinstall_busy_is_explicit_failure_and_auto_soft(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    (project / "local.py").write_text("def local_busy_fallback(): return 7\n")
    target = project / ".chunkhound" / "db" / "chunks.db"
    target.parent.mkdir(parents=True)
    publish(hub_http, build_database(tmp_path / "published"))
    # This owner excludes installation but owns no backend writer connection.
    held = native.acquire_database_lease(str(target))
    try:
        result = cli(project, mode, project, "--verbose")
    finally:
        held.close()
    assert result.returncode == (1 if mode == "pull" else 0)
    assert "destination is busy" in result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert target.exists() is (mode == "index")


def test_zero_is_not_a_relative_version(tmp_path):
    assert cli(tmp_path, "pull", "--version", "0").returncode == 2
```

The `{backend} × {rust}` matrix is intentional: DuckDB uses both execution paths; LanceDB with Rust enabled **must use the existing Python fallback**, not a nonexistent native LanceDB writer. No fake embedding provider and no monkeypatching `configure_registry` to establish hook ordering.

- [ ] **T8.2 RED.** `UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_cli.py -v` must fail unknown pull command and missing auto installs. The original CLI intentionally cannot pass these contracts.

- [ ] **T8.3 Add the parser/command and registration.** Complete `chunkhound/api/cli/parsers/pull_parser.py`:

```python
import argparse
from pathlib import Path
from typing import Any, cast

from .common_arguments import add_common_arguments, add_config_arguments


def add_pull_subparser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("pull", help="Install a matching hub base index")
    parser.add_argument("path", nargs="?", default=Path("."), type=Path)
    parser.add_argument("--version", choices=("latest", "-1", "-2"), default="latest")
    add_common_arguments(parser)
    add_config_arguments(parser, ["database", "embedding"])
    return cast(argparse.ArgumentParser, parser)
```

Complete `chunkhound/api/cli/commands/pull.py`:

```python
import argparse

from chunkhound.core.config.config import Config
from chunkhound.hub.client import pull
from chunkhound.hub.errors import HubDatabaseMismatch, HubInstallationInterrupted


async def pull_command(args: argparse.Namespace, config: Config) -> None:
    try:
        installed = await pull(config, rel=args.version)
    except (HubDatabaseMismatch, HubInstallationInterrupted):
        # The hub already logged a sanitized error; no CLI traceback/causes.
        raise SystemExit(1) from None
    if not installed:
        raise SystemExit(1)
```

In `main.py:create_parser`, import/register beside the run parser:

```python
from .parsers.pull_parser import add_pull_subparser
```

```python
add_pull_subparser(subparsers)
```

In `async_main`, insert this branch after the `index` branch:

```python
elif args.command == "pull":
    from .commands.pull import pull_command

    await pull_command(args, config)
```

Do not add hub URLs/versions to analytics action fields. Pull is not implicitly added to analytics-wrapped commands in this change.

- [ ] **T8.4 Add exactly one index hook and admit safe info logs.** In `commands/run.py:run_command`, insert after `_validate_run_arguments` succeeds and before `drift_seen` / `_make_model_drift_prompt` / the registry `try`:

```python
if (not _mcp_mode()
        and not getattr(args, "no_embeddings", False)
        and not config.embeddings_disabled
        and config.hub.url):
    from chunkhound.hub.client import pull
    from chunkhound.hub.errors import HubDatabaseMismatch, HubInstallationInterrupted

    try:
        await pull(config, automatic=True)
    except (HubDatabaseMismatch, HubInstallationInterrupted):
        raise SystemExit(1) from None
```

All actual `configure_registry` calls, including daemon-conflict retry, remain below this hook. Hub busy cannot invoke daemon stop: the hub client logs/returns before that unrelated existing indexing behavior. Simulation/check-ignore early returns remain above the hook. `get_db_path()` and the offline configured key handle unsupported layouts/unknown dimensions; do not change the global registry/factory or search/MCP setup.

In `utils/rich_output.py:default_sink_filter`, replace the current Rust-only condition with:

```python
if record["extra"].get("rust_native") or record["extra"].get("hub_client"):
    return True
```

Retain its final severity filter; do not attach another sink. In `tests/test_smoke.py`'s help command list, insert immediately after index help:

```python
["chunkhound", "pull", "--help"],
```

- [ ] **T8.5 GREEN and commit checkpoint.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_cli.py tests/hub/test_activation_client.py tests/hub/test_config.py tests/hub/test_manifest.py -v
UV_NO_SYNC=1 uv run ruff check chunkhound/hub chunkhound/api/cli/parsers/pull_parser.py chunkhound/api/cli/commands/pull.py tests/hub/activation_support.py tests/hub/test_activation_cli.py
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/activation-client-c2-mypy.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
git diff --check
git add chunkhound/api/cli/parsers/pull_parser.py chunkhound/api/cli/commands/pull.py chunkhound/api/cli/main.py chunkhound/api/cli/commands/run.py chunkhound/api/cli/utils/rich_output.py tests/hub/test_activation_cli.py tests/hub/conftest.py tests/test_smoke.py
git commit -m "feat: activate pull and pre-registry automatic hub installation"
```

Expected targeted/smoke PASS; no new mypy errors. Real drift/no-request tests are a gate, not optional coverage deferred to T9.

---

---

### Task T9: Pull/search/overlay acceptance, recovery exits and operator documentation

**Files:** Create `tests/hub/test_activation_acceptance.py`, `docs/hub-client.md`; modify `README.md`. No indexing/diff engine source modifications are planned. If acceptance reveals a real defect, stop and report the failing contract before expanding this task's source scope.

**Consumes:** T7/T8 public CLI, real artifact/HTTP/embedding fixtures; existing `SubprocessJsonRpcClient` for real MCP; T1–T6 already own process-kill, direct native holders, provider reconnect/alias/handoff race and native cleanup fault tests.

**Produces:** End-to-end evidence for actual installed-base search, unchanged/new/changed/deleted overlay across a moved checkout, no hub requests from MCP/search, fatal interrupted auto-index and explicit re-pull recovery. User-facing documentation with no unsupported durability/platform claim.

- [ ] **T9.1 Add real acceptance contracts.** Create `tests/hub/test_activation_acceptance.py`:

```python
import asyncio
import json
import shutil

import pytest

from chunkhound.hub.inspect import inspect_database
from tests.hub.activation_support import cli, cli_env, publish, write_config
from tests.hub.artifacts import build_database
from tests.utils import SubprocessJsonRpcClient


def source_text(marker):
    # Distinct, sizeable functions avoid a tiny-file/chunk coalescing fixture.
    return "\n\n".join(
        f'def function_{i}():\n'
        f'    """{marker}: document a meaningful example with independent inputs, '
        f'validation, returned values and stable error handling for operation {i}."""\n'
        f'    return "{marker}_{i}"\n'
        for i in range(100)
    )


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
def test_moved_checkout_overlays_unchanged_new_changed_and_deleted_files(
    tmp_path, hub_http, embedding_http, backend, rust
):
    publisher = tmp_path / "publisher-checkout"
    write_config(publisher, backend, None, embedding_http.url)
    for filename, marker in [("unchanged.py", "UNCHANGED_BASE"),
                             ("changed.py", "OLD_CHANGED_BASE"),
                             ("deleted.py", "DELETED_BASE")]:
        (publisher / filename).write_text(source_text(marker), encoding="utf-8")
    built = cli(publisher, "index", publisher, rust=rust)
    assert built.returncode == 0, built.stdout + built.stderr
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    publisher_target = publisher / ".chunkhound" / "db" / name
    publish(hub_http, publisher_target)
    checkout = tmp_path / "different-parent" / "moved-checkout"
    write_config(checkout, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    for filename in ("unchanged.py", "changed.py", "deleted.py"):
        shutil.copy2(publisher / filename, checkout / filename)
    pulled = cli(checkout, "pull", checkout, rust=rust)
    assert pulled.returncode == 0, pulled.stdout + pulled.stderr
    before = cli(checkout, "search", "UNCHANGED_BASE", checkout, "--regex", rust=rust)
    assert before.returncode == 0 and "unchanged.py" in before.stdout + before.stderr
    (checkout / "changed.py").write_text(source_text("NEW_CHANGED_LOCAL"), encoding="utf-8")
    (checkout / "new.py").write_text(source_text("NEW_LOCAL_FILE"), encoding="utf-8")
    (checkout / "deleted.py").unlink()
    embedding_http.requests.clear()
    hub_http.requests.clear()
    overlaid = cli(checkout, "index", checkout, "--verbose", rust=rust)
    assert overlaid.returncode == 0, overlaid.stdout + overlaid.stderr
    assert hub_http.requests == []  # Aligned base, no second pull.
    assert embedding_http.requests
    embedded_text = json.dumps(embedding_http.requests)
    assert "UNCHANGED_BASE" not in embedded_text  # Real hash-based unchanged skip.
    assert "NEW_CHANGED_LOCAL" in embedded_text
    assert "NEW_LOCAL_FILE" in embedded_text
    for marker, filename in [("UNCHANGED_BASE", "unchanged.py"),
                             ("NEW_CHANGED_LOCAL", "changed.py"),
                             ("NEW_LOCAL_FILE", "new.py")]:
        result = cli(checkout, "search", marker, checkout, "--regex", rust=rust)
        assert result.returncode == 0, result.stdout + result.stderr
        assert filename in result.stdout + result.stderr
        assert "No results found" not in result.stdout + result.stderr
    for marker in ("OLD_CHANGED_BASE", "DELETED_BASE"):
        result = cli(checkout, "search", marker, checkout, "--regex", rust=rust)
        assert result.returncode == 0
        assert "No results found" in result.stdout + result.stderr
    semantic = cli(checkout, "search", "UNCHANGED_BASE", checkout,
                   "--single-hop", rust=rust)
    assert semantic.returncode == 0, semantic.stdout + semantic.stderr
    assert "No results found" not in semantic.stdout + semantic.stderr
    assert hub_http.requests == []
    assert {g.model for g in inspect_database(
        checkout / ".chunkhound" / "db" / name, backend
    ).groups} == {"hub-test"}


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
async def test_mcp_and_search_do_not_pull_even_when_configured_model_differs(
    tmp_path, hub_http, embedding_http, backend
):
    project = tmp_path / "checkout"
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published", backend))
    pulled = await asyncio.to_thread(cli, project, "pull", project)
    assert pulled.returncode == 0, pulled.stdout + pulled.stderr
    # Keep the source represented by the synthetic base available to MCP indexing.
    (project / "source.py").write_text(source_text("MCP_LOCAL"), encoding="utf-8")
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url, model="different-configured-model")
    hub_http.requests.clear()
    searched = await asyncio.to_thread(cli, project, "search", "function_0", project,
                                       "--regex")
    assert searched.returncode == 0, searched.stdout + searched.stderr
    assert hub_http.requests == []
    process = await asyncio.create_subprocess_exec(
        "uv", "run", "chunkhound", "mcp", str(project), "--no-daemon",
        cwd=project, env=cli_env(project), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stderr_task = asyncio.create_task(process.stderr.read())
    client = SubprocessJsonRpcClient(process)
    await client.start()
    try:
        initialized = await client.send_request(
            "initialize", {"protocolVersion": "2024-11-05", "capabilities": {},
                           "clientInfo": {"name": "hub-acceptance", "version": "1"}},
            timeout=60,
        )
        assert "serverInfo" in initialized
        await client.send_notification("notifications/initialized", {})
        result = await client.send_request(
            "tools/call", {"name": "search", "arguments":
                           {"type": "regex", "query": "function_0"}}, timeout=60,
        )
        assert not result.get("isError", False), result
        assert hub_http.requests == []
    finally:
        await client.close()
        await stderr_task


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
def test_interrupted_auto_refuses_and_explicit_repull_restores_search(
    tmp_path, hub_http, embedding_http, backend, rust
):
    project = tmp_path / "checkout"
    write_config(project, backend, hub_http.url + "/catalogue?token=secret",
                 embedding_http.url)
    (project / "local.py").write_text("def local_reindex_work(): return 1\n")
    name = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    target = project / ".chunkhound" / "db" / name
    target.parent.mkdir(parents=True)
    marker = target.with_name(target.name + ".chunkhound-install.json")
    marker.write_bytes(b"{")
    publish(hub_http, build_database(tmp_path / "published", backend))
    refused = cli(project, "index", project, "--verbose", rust=rust)
    assert refused.returncode == 1
    assert "explicit chunkhound pull" in refused.stderr
    assert "Traceback" not in refused.stdout + refused.stderr
    assert hub_http.requests == []
    assert marker.read_bytes() == b"{" and not target.exists()
    restored = cli(project, "pull", project, rust=rust)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert not marker.exists()
    result = cli(project, "search", "function_0", project, "--regex", rust=rust)
    assert result.returncode == 0
    assert "source.py" in result.stdout + result.stderr
    assert "No results found" not in result.stdout + result.stderr
```

A marker-only CLI fixture checks policy wiring, not the whole crash protocol. T6 must separately kill real installer processes at marker/isolation/rename/commit boundaries and prove normal opens/explicit recovery. Do not replace those native tests with this fixture. T7 cancellation uses a real exclusive install plus deterministic filesystem-boundary delay.

- [ ] **T9.2 Run the acceptance RED/characterization checkpoint.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_acceptance.py -v
```

Before T7/T8 this must fail at pull; after T8 it may pass immediately because T9 primarily supplies acceptance/docs, not a production behavior change. Do not manufacture a red by weakening working code. If it fails now, retain exact output and stop for a bounded defect plan/review; do not rewrite overlay or add mock-only assertions.

- [ ] **T9.3 Write operator documentation and README link.** Complete `docs/hub-client.md`:

```markdown
# Pulling a central base index

ChunkHound can install a compatible base from a private, organization-operated
hub. The publisher is trusted; the client still verifies checksums, safely
extracts the archive, and checks stored backend/format/embedding identity.
ChunkHound never publishes an index.

Configure a full manifest URL in `.chunkhound.json`:

    {"hub": {"url": "https://hub.example/catalogue?branch=main"}}

Or set `CHUNKHOUND_HUB__URL`. The URL is opaque: the client appends no path.
Manifest v2 supplies full artifact URLs. HTTP and HTTPS are supported without
another setting. HTTPS verifies certificates and refuses HTTPS-to-HTTP
redirects. HTTP cannot protect against network interception. Do not commit
secrets even though committed hub configuration is allowed and hub diagnostics
redact URLs. Existing embedding configuration/credential requirements apply;
pull resolves its configured identity offline and does not request embeddings.

    chunkhound pull /path/to/checkout --version latest
    chunkhound pull /path/to/checkout --version -1
    chunkhound index /path/to/checkout
    chunkhound search 'search term' /path/to/checkout

Relative versions are exactly `latest`, `-1`, `-2`; there is no automatic
version walking. Selection matches backend, configured storage provider,
model, known dimensions and owned index format exactly. Unknown dimensions or
Azure/openai storage-identity ambiguity cannot select a base. Metadata such as
project/revision is diagnostic, not a selection shortcut.

Explicit pull installs only. It exits 1 if it cannot install a matching base.
It replaces local index state, so local work may need re-indexing. A subsequent
ordinary index overlays changed/new/deleted local files and keeps unchanged
base content through existing hash-based indexing. Moved checkouts are
supported by existing relative-path indexing, not a new hub diff engine.

`index` considers automatic pull only with a configured hub and known usable
embedding identity, when the target is missing or embedding-misaligned. It
runs before registry initialization and model-drift pinning. An aligned index
makes no hub request. Simulation, ignore checking, no-embeddings indexing,
MCP and search do not pull. Auto-index continues into ordinary local overlay
following installation. Download/selection/checksum/archive/space/pre-install
busy failures log and follow existing local behavior; that behavior can still
fail if another process owns an incompatible backend connection. A downloaded
database contradicting its manifest is fatal and leaves the old target alone.

## Ownership and interrupted installation

Installation supports directory-layout databases only: `chunks.db` or
`lancedb.lancedb` inside the configured database directory. A configured
single-file path, even one named `chunks.db`, memory and read-only installation
are unsupported. A symlinked installation target is refused. Ordinary relative
paths and symlinked parents coordinate through a stable adjacent lock file.
The native database interface requires the resolved canonical database path to
be UTF-8. An ASCII alias pointing to a non-UTF-8 target is also refused with
`DatabaseLayoutError` before database preflight, recovery, read or write, even
on an empty indexing run. Use a UTF-8 database location; no replacement-character
neighbor is accessed instead. UTF-8 Unicode paths, aliases to valid UTF-8
locations and ordinary memory/legacy database use remain supported. This
restriction does not change exact lease-only path identity. Refusal may create
the stable coordination file, but leaves original database/recovery files and
neighboring targets untouched and releases acquired ownership.
Do not delete that lock file: its existence does not mean a process is live.

Current-version providers, read-only holders and native work share installation
exclusion for their complete database lifetime. A busy pull declines promptly;
it does not kill daemons. Shared ownership is not permission for incompatible
concurrent backend writers. `CHUNKHOUND_USE_RUST=0` selects Python indexing but
still uses the native ownership/install protocol. LanceDB uses the Python
pipeline even with Rust indexing enabled.

Stop older ChunkHound clients and direct third-party database users before
pulling: they do not participate in cooperative exclusion. Hard-link aliases
and filesystems without reliable advisory locks/same-filesystem rename are
outside this guarantee; network-filesystem behavior is not assumed. Read-only
users need a usable existing coordination file or permission to create it;
they never open without coordination merely because creation is denied.

Download and validation occur before exclusive installation ownership. Auto
rechecks its trigger while exclusively owned. Rust publishes an in-progress
marker, isolates the old database and recognized sidecars/recovery files in a
unique quarantine, moves the candidate, and removes the marker as the
process-level commit point. This is not an atomic multi-file swap or a
power-loss rollback guarantee. Unrelated neighboring files are not cleanup
targets. New bases do not inherit publisher-root or active-model sidecars.

If installation is interrupted, normal opens and auto-index refuse before
backend recovery/migration. Run explicit `chunkhound pull` again. It downloads
and verifies a fresh candidate before replacing partial state; failed download
leaves the marker and files alone. Local index work may be lost and require
re-indexing. Never replay an old WAL against a new base or restore a quarantine
by guessing. Abandoned quarantines may remain for operator removal after
stopping database users; they are never automatically selected for recovery.

Cleanup failure after commit is a warning, not a failed pull: the new base is
usable even if old quarantine or staging work remains. Interrupted awaits wait
for their filesystem worker to finish before releasing leases/staging. There
is no permanent artifact cache. Windows/POSIX support is verified by the
project's platform CI; a local Linux run alone does not establish it.
```

In `README.md`, insert this short paragraph immediately before the existing `## Community` heading (currently line 205), without moving other content:

```markdown
## Central base indexes

Use `chunkhound pull [path] --version latest|-1|-2` to install a compatible
organization-hosted base, then `chunkhound index [path]` to overlay local changes.
See [Hub client operation and recovery](docs/hub-client.md) for configuration,
automatic-index behavior, ownership requirements and interruption recovery.
```

- [ ] **T9.4 Run regression/type/platform gates and commit.**

```bash
UV_NO_SYNC=1 uv run pytest tests/hub/ -v
UV_NO_SYNC=1 uv run pytest tests/hub/test_activation_acceptance.py tests/hub/test_activation_cli.py -v
UV_NO_SYNC=1 uv run ruff check chunkhound > .superpowers/sdd/activation-client-ruff.log 2>&1
UV_NO_SYNC=1 uv run mypy chunkhound > .superpowers/sdd/activation-client-mypy-after.log 2>&1
UV_NO_SYNC=1 uv run pytest tests/test_smoke.py -v -n auto
git diff --check
git add tests/hub/test_activation_acceptance.py docs/hub-client.md README.md
git commit -m "test: prove hub pull search overlay and document recovery"
```

Expected hub/acceptance/smoke PASS, no introduced mypy/ruff errors. Existing unrelated baseline failures must be distinguished by exact diagnostic/file; do not claim all static checks passed if only the introduced-diagnostic comparison did. Native/lifetime commits already ran `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, and `cargo test`; rerun those before any combined Rust-change commit. Existing `.github/workflows/ci.yml` tests the full suite with built native extension on Ubuntu/macOS/Windows; these tests must collect there, with no local-only skip or xfail replacing required coverage. Review actual platform results before claiming cross-platform support.

Before a later authorized PR push (not authorized here), run:

```bash
UV_NO_SYNC=1 uv run pytest tests/ -v
```

Record pinned-runtime/full-suite failures as remaining validation debt, not erased by this trusted-publisher scope. No push follows this command in this plan.

## Gate B: Evidence and human execution/integration checkpoint

Report each task's targeted tests, the last smoke result, Rust fmt/clippy/test results, introduced-versus-baseline mypy/ruff diagnostics, and available platform results. Do not infer a pass from a plan command or isolated dependency probe. Do not push, merge or release. If the full suite/platform jobs are not run or have existing failures, record that explicitly before requesting any later integration approval.

## Author's inline self-review

This section records plan coverage, not implementation/test success.

| Approved spec section / contract | Planned implementation and verification |
|---|---|
| Status/precedence; trusted publisher; retained foundation | Header/global constraints; T7 reuses manifest/transport/artifact/inspection rather than rewriting them |
| Rust/Python boundaries; safe maintained lock crate/MSRV/GIL | T1 exact fs4 dependency and owned PyO3 handles; T6 filesystem-only core; T7 thin policy adapter |
| Identity/layout; relative and symlink-parent aliases; read-only cooperation | T1 alias/permission/symlink-leaf tests; T4 checked UTF-8 canonical database target refusal/nonmutation without changing lease identity; T7 configured legacy/memory/read-only refusal; T9 support boundaries |
| Full Python holder lifetime, lazy/partial/reconnect, compatible users | T2 whole inspection; T3 provider/executor/manager teardown and committed-cursor preflight; original transaction preservation tests retained |
| Native readers/writers/recovery/compaction; handoff interval | T4 backend and run guards through worker joins; T5 retained provider guard through disconnect/native/reconnect |
| Cancellation/resources and process inheritance | T5 shield-and-drain actual workers; spawn all parser pools; child-survives-owner contract; T7 stage/lease delay contract |
| Exclusive auto recheck, no nested acquisition, candidate closed first | T2 borrowed ownership; T7 initial trigger plus exclusive recheck; aligned-race winner and racing-marker contracts |
| Stable lock, marker-before-isolation, exact sidecars, rename/commit | T6 exact suffix set and marker/quarantine code; unrelated neighbor preservation; no cleanup paths read from marker |
| Interrupted refusal and explicit fresh recovery | T1/T3/T4 refusal before recovery; T6 process kills at marker/isolation/rename/commit; T7 failed recovery download preservation; T9 real CLI recovery/search |
| Cleanup warning only after commit; no rollback/power-loss claim | T6 native warning code; T7 commit state before staging cleanup; T9 operator wording |
| Cold/replacement pull and retained relative versions | T7 real backend replacement/search; T8 CLI latest/-1/-2 selection and explicit exit contracts |
| Auto missing/misaligned before pinning; aligned/no-request exclusions | T8 real subprocess backend/Rust-mode matrix; no-embedding/simulate/ignore checks; T9 MCP/search exclusion |
| Soft download/checksum/archive/space/preinstall busy; fatal mismatch/interruption | T7 preservation/category tests; T8 subprocess exits/fallback and sanitized diagnostics; T9 interrupted auto refusal |
| Live Python/native work, two pulls and cold opener | T1 shared-owner tests; T3/T4 holder tests; T6 paused installer probes; T7 concurrent pull/provider contract |
| Actual search and moved-checkout unchanged/new/changed/deleted overlay | T9 existing indexing paths with HTTP embedding boundary and actual regex/semantic search, both backends/Rust settings |
| Testing/regression/platform/delegation requirements | Per-task RED/GREEN/smoke; Rust gates; baseline static-check comparison; T9 full-suite/platform gate; single-writer delegation table and Gate A |

Self-review corrections applied inline: root-sidecar isolation/assertions are DuckDB-only (LanceDB neighbors are not cleanup targets); added explicit symlink-leaf refusal, competing pulls/cold opener, and failed re-download preservation coverage. Public native names, ownership keyword types and worker completion signatures match across all tasks. No task depends on the ignored planning fragments: exact implementation/test content is contained here.

Plan checks: nine ordered tasks; placeholder scan clear; 77 complete Python blocks parse. Eight additional Python-labelled blocks are deliberately interface signatures or exact insertion fragments (`with`/`except`/`elif`/function header), not standalone modules. These checks do not compile Rust, typecheck the proposed implementation or establish backend test success. Implementation remains unstarted.
