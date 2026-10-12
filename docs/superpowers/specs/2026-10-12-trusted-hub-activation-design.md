# Trusted-hub activation: Rust install core, Python hub client

## Status and precedence

The user approved the minimal cooperative-locking/recovery direction, then selected a Rust locking/install core while retaining the completed Python hub foundation. This document records that direction for written-spec review before implementation planning.

This addendum supersedes the always-blocked installation proposal in Tasks 8–9 of `2026-10-11-central-indexing-hub-client.md`. It also supersedes adversarial-publisher audit closure as an activation requirement. Foundation Tasks 1–7 are already implemented and are not to be repeated. Existing tests and ordinary compatibility checks remain valuable; this work is not a broad cleanup or rewrite.

## Goal and boundaries

Deliver the normal lifecycle: select a matching base, download, validate, install, search, and overlay local changes through existing indexing.

Hubs are private, organization-operated and trusted. Keep checksum verification, bounded safe archive extraction, database-format and embedding compatibility checks, TLS verification, and existing sanitized hub diagnostics. Do not pursue deliberately crafted catalog/collation attacks as release blockers.

A developer may lose local index work and need to re-pull or rebuild after interrupted installation. Do not delete unrelated files or silently open mixed old/new database state. Do not promise a durable rollback copy through machine power loss or storage failure.

No publisher, permanent artifact cache, authentication service, new diff engine, generation-pointer storage layout, or native LanceDB backend is introduced. No push, merge, or release is authorized by this document.

## Language and component boundaries

### Python retains policy and backend knowledge

Keep `chunkhound/hub/{manifest,transport,artifact,inspect}.py` and existing configuration/format work. Python owns:

- Configured-key selection and full manifest/artifact URLs.
- HTTP retries, checksum verification, extraction and candidate inspection.
- CLI policy, logging, failure classification and the pre-registry auto-pull hook.
- DuckDB/LanceDB provider integration, search, and local overlay.

A small Python installation adapter converts paths and native error categories; it does not implement a second filesystem protocol.

### Rust owns coordination and installation mechanics

Add focused modules for database leases and installation. Expose the necessary operations through the existing PyO3 extension. The Rust core accepts owned local paths and a backend discriminator; it does not accept manifests, URLs, embedding credentials or arbitrary cleanup paths.

Its responsibilities are:

1. Acquire/release shared database-use leases and exclusive installation leases.
2. Check interruption state before database access.
3. Isolate the old database and its recognized sidecars/recovery artifacts.
4. Move the verified candidate into place and commit the installation marker state.
5. Return typed busy, interrupted, unsupported-layout and filesystem failures.

The same Rust lease implementation is used by Python providers and native database entry points. There is no Python-only alternative locking protocol. `CHUNKHOUND_USE_RUST=0` continues to select the Python indexing pipeline; it does not disable installation exclusion. The native extension is already a project dependency.

Use a maintained cross-platform advisory file-locking crate compatible with the project's Rust minimum. No custom unsafe platform layer. CPU/IO operations release the GIL, with owned values crossing `allow_threads`; errors become explicit Python exceptions rather than panics.

## Cooperative database ownership

### Identity and scope

Use a stable lock file beside the actual database target, outside any database directory that is renamed. Never unlink/recreate the lock as part of installation. Resolve ordinary relative paths and symlinked parents to a consistent target before choosing the lock. For absent targets, normalize the existing parent and append the target name. Reject a symlinked installation target rather than replacing its link or chasing it during cleanup.

Initial installation supports existing directory-layout targets: `chunks.db` and `lancedb.lancedb` inside the configured database directory. Legacy single-file configurations, in-memory databases and read-only installation are unsupported. Ordinary non-hub database use retains its existing supported layouts. Filesystem deployments must support reliable advisory locks and same-filesystem rename; network-filesystem behavior is not assumed.

Hard-link aliases, clients predating this protocol, and direct third-party database connections are outside cooperative exclusion. Operators must stop such users before pulling. This is a documented support boundary, not a claim that the installer detects them.

### Lifetime rules

- Every current-version file-backed database holder acquires a shared lease before format preflight, recovery or opening backend handles. Hold it until all owned handles and pending database work have finished.
- Shared here means "installation exclusion", not permission for concurrent backend writers. DuckDB/LanceDB retain their own read/write compatibility rules. Multiple compatible providers must not fail merely because another shared lease exists.
- The installer acquires an exclusive lease before modifying the destination. Contention returns a busy result; it does not kill a daemon or wait indefinitely.
- Read-only holders participate as well. New read-only access needs an existing usable coordination file or permission to create it; inability to coordinate must produce an explicit error, not an unlocked open.
- Provider reconnect is idempotent with respect to leases. A failed open releases ownership only after any partially opened handles are closed. Do not roll back or discard another live provider's transaction to acquire a lease.
- Python keeps its shared lease during the existing disconnect-for-Rust/reconnect bracket. The native pipeline also acquires shared ownership before its first diff-reader/recovery operation, so direct native calls are protected. Shared ownership overlaps; there is no unlocked handoff interval.
- Native readers, writers, compaction and recovery remain covered until their handles close. Process exit releases OS locks; an abandoned lock file alone is not evidence of a live owner.

Download/staging happens without exclusive ownership of the target. The candidate is a private temporary database, not the destination. Close all candidate inspection handles before installation. Automatic pull rechecks the local trigger under the exclusive installation lease before replacing, since another process may have populated the target during download. Expose that lease as a scoped native handle so Python can inspect while it remains held, then close inspection handles before committing. Inspection under that lease must not try to acquire a nested shared lease; unrelated ordinary inspection uses shared ownership. No shared-to-exclusive lock upgrade is required.

A normal open must acquire its own shared lease after installation; no guarantee is made that a racing independent pull cannot win before that open. Async cancellation must not drop leases or staging directories while native filesystem/database work is still running. Wait for that work to finish before releasing its resources; do not use task cancellation as proof that a worker thread stopped.

## Installation and interruption protocol

Use a small, versioned in-progress marker beside the target and a uniquely named quarantine directory in the same parent. The marker and quarantine names are installer-owned; they are not database recovery sidecars.

Under exclusive ownership:

1. Confirm supported layout and the presence of the already-validated candidate. Recheck the automatic-install condition when relevant.
2. Publish an in-progress marker before changing any old index file. Marker creation failure leaves the old index untouched.
3. Move the existing target and recognized associated files into the quarantine directory. For DuckDB this includes its WAL, supported compaction/recovery artifacts, root sidecar and active-embedding sidecar. For LanceDB move the database directory as a unit and any recognized external sidecars. Use the actual filenames used by current providers/recovery code; never sweep the configured parent or use a broad user-file glob.
4. Rename the candidate into the destination, on the same filesystem. All old sidecars must already be isolated. Newly installed bases start without publisher-root or active-model sidecars.
5. Remove the marker only after the new target is in place and isolation is complete. This is the process-level commit point. Release ownership after finishing the install operation.
6. Quarantine cleanup is best-effort and limited to installer-owned paths. Cleanup failure after commit is a warning, not a failed pull or a reason to restore stale sidecars.

The multi-file operation is not called an atomic swap. The lease prevents participating readers from observing intermediate state; the marker prevents normal opens after a process dies during it.

If a marker survives, normal database opens refuse before backend recovery/migration, with guidance to run explicit `chunkhound pull` again. Do not guess which quarantined database is best, replay an old WAL against a new database, or let auto-index silently create a database over partial state. This refusal is an intentional exception to ordinary automatic-pull fallback: continuing local indexing would be unsafe.

Explicit re-pull may recover by first downloading and verifying a fresh candidate, then acquiring exclusive ownership and isolating any partial current target/recognized sidecars before completing the same protocol. Failed re-download leaves the marker and existing files alone. Abandoned quarantine contents are not automatically restored; diagnostics explain that local work may need re-indexing. Do not require an intact old index to recover.

Tests cover process interruption and ordinary filesystem errors. Power-loss durability across platforms/filesystems is not guaranteed by a marker or rename. Keep incomplete or ambiguous marker state as a refusal during ordinary opens, never a license for unrestricted cleanup. Explicit re-pull uses only the target-derived recognized paths and a fresh quarantine, not cleanup paths read from a possibly incomplete marker. Old abandoned quarantines may remain for manual removal; they are never candidates for backend recovery.

## CLI and error behavior

- `chunkhound pull [path] --version latest|-1|-2` explicitly installs a matching base. Any inability to install returns exit 1. No automatic version walking.
- `chunkhound index` considers automatic pull only when a hub is configured and a usable embedding identity is available, and the local target is absent or embedding-misaligned. Aligned local databases cause zero hub requests.
- Run the hook before registry initialization and drift pinning. Simulation, ignore checking and no-embeddings indexing do not pull. MCP and search never initiate a pull.
- Network, selection, checksum, extraction, space and pre-install busy failures remain soft for auto-index: log and follow existing local behavior. This is not a promise that local indexing can succeed against a busy backend.
- A downloaded database contradicting its manifest is fatal for both modes; the old target and its sidecars remain untouched.
- Errors after installation starts retain the interruption marker unless the commit point was reached. They must not be swallowed as ordinary fallback followed by a provider open.
- Use typed native error categories at the Python boundary; do not parse engine exception text or expose raw paths/URLs through new diagnostic causes.
- Explicit pull installs only. A subsequent normal index overlays local changes; auto-index continues into that overlay in the same command. Search operates unchanged on the installed base.

## Verification contracts

Prefer real backends and subprocesses, with fakes only for external HTTP/embedding services and deterministic fault injection at filesystem boundaries.

1. Cold pull and explicit replacement followed by actual search for DuckDB and LanceDB; relative versions select the requested retained build.
2. Auto-index absent/misaligned installation before drift pinning; aligned index makes no request; MCP/search/no-embeddings paths make no request.
3. Moved-checkout overlay preserves unchanged base content and incorporates new, changed and deleted files through existing indexing logic.
4. Common download, checksum, archive, space and candidate-mismatch failures have the promised exits/fallback and preserve the pre-install target/sidecars.
5. Live Python DuckDB/LanceDB users and direct native work prevent installation. Compatible same-process providers/reconnect remain usable. Python/native handoff has no installation window. Two pulls and a concurrent cold opener cannot observe partial state.
6. Kill the installer after marker creation, during old-file isolation, and after candidate rename but before marker removal: normal open refuses; explicit re-pull restores a searchable index. Kill after marker removal: the new index remains usable even if quarantine cleanup was incomplete.
7. Normal relative/symlink-parent aliases coordinate; unrelated neighboring files survive success and failure. OS exit releases leases without manual lock-file deletion.
8. Exercise Rust-enabled and Python-pipeline indexing. Run platform CI for supported Windows/POSIX lock and rename behavior before claiming cross-platform support.

Run targeted contracts, Rust fmt/clippy/tests for Rust changes, and smoke before each commit. Run mypy and distinguish existing baseline errors from introduced errors. Full suite before any future PR push. Existing pinned-runtime/full-suite validation debt is not erased by trusting the publisher.

## Implementation planning and delegation

Produce a successor activation plan, not the old refusal-only Tasks 8–9:

1. Rust lease primitive and PyO3 interface; bounded lifecycle mapping supplies exact entry points.
2. Python/native lifetime integration and real contention/handoff contracts.
3. Rust marker/quarantine installer and interruption/re-pull contracts.
4. Python orchestration, CLI and pre-registry hook.
5. Pull/search/overlay acceptance tests, documentation and regression validation.

Use one writer per dependency-ordered task. Delegate bounded lifecycle discovery or independent contract review where useful, not competing writers in the same subsystem. Review the lease/lifetime contract before implementing installer activation. Do not restart a broad adversarial metadata audit or port the completed Python transport/inspection code.
