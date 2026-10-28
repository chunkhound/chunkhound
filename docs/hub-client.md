# Pulling a central base index

ChunkHound can install a compatible base from a private, organization-operated
hub. The publisher is trusted; the client still verifies checksums, safely
extracts the archive, and checks stored backend/format/embedding identity.
ChunkHound never publishes an index. Format and checksum verification establish
compatibility and artifact integrity, not publisher content freshness.

DuckDB and LanceDB support pulling a base and then indexing local changes.
LanceDB remains experimental; it uses the Python indexing pipeline even when
Rust indexing is enabled. DuckDB is the recommended backend.

## Configuration and local overlay

Configure a backend and a full manifest URL in `.chunkhound.json`, alongside
your existing embedding configuration:

```json
{
  "database": {"provider": "duckdb"},
  "hub": {"url": "https://hub.example/catalogue?branch=main"}
}
```

Or set `CHUNKHOUND_HUB__URL`. The URL is opaque: the client appends no path.
Manifest v2 supplies full artifact URLs. HTTP and HTTPS are supported without
another setting. HTTPS verifies certificates and refuses HTTPS-to-HTTP
redirects. HTTP cannot protect against network interception. Do not commit
secrets even though committed hub configuration is allowed and hub diagnostics
redact URLs. Existing embedding configuration/credential requirements apply;
pull resolves its configured identity offline and does not request embeddings.

```bash
chunkhound pull /path/to/checkout --version latest
chunkhound pull /path/to/checkout --version -1
chunkhound index /path/to/checkout
chunkhound search 'search term' /path/to/checkout
```

Relative versions are exactly `latest`, `-1`, `-2`; there is no automatic
version walking. Selection matches backend, configured storage provider,
model, known dimensions and owned index format exactly. Unknown dimensions or
Azure/openai storage-identity ambiguity cannot select a base. Metadata such as
project/revision is diagnostic, not a selection shortcut.

Explicit pull installs only. It exits 1 if it cannot install a matching base.
It replaces local index state, so local work may need re-indexing. For DuckDB
and LanceDB, a subsequent ordinary index overlays changed/new/deleted local
files and keeps unchanged base content through existing hash-based indexing.
Changed and deleted files do not retain their old searchable chunks after a
successful overlay. Moved checkouts use existing relative-path indexing, not a
new hub diff engine. Pull alone does not reconcile the publisher's contents
with the local checkout or repair arbitrary stale rows in a publisher's base.

`index` considers automatic pull only with a configured hub and known usable
embedding identity, when the target is missing or embedding-misaligned. It
runs before registry initialization and model-drift pinning. An aligned index
makes no hub request. Simulation, ignore checking, no-embeddings indexing,
MCP and search do not pull. Auto-index continues into ordinary local indexing
following installation. Download/selection/checksum/archive/space/pre-install
busy failures log and follow existing local behavior; that behavior can still
fail if another process owns an incompatible backend connection. A downloaded
database contradicting its manifest is fatal and leaves the old target alone.

## Publisher packaging and client archive limits

Publish a gzip-compressed tar containing exactly the backend's database root:
`chunks.db` as a regular file for DuckDB, or `lancedb.lancedb/` and its contents
for LanceDB. Do not add a checkout-parent directory or unrelated root entries.
Publish a closed, consistent database snapshot, not an actively written file
copy. Checkpoint and close DuckDB before packaging; do not package a WAL or
publisher-root/active-model sidecars. Direct native installation also refuses
any adjacent DuckDB candidate `.wal` entry, including directories and dangling
symlinks, before installation mutation; it does not delete or replay that WAL.

USTAR, GNU and PAX archives are supported, including implicit parent directories
and highly compressible content. Normal directory modes such as `0755` and
regular-file modes such as `0644` are accepted; publishers need not zero modes.
Special permission bits are rejected for every member, and executable bits are
rejected for regular files. Only ordinary files/directories are accepted, not
links, devices or sparse members. Paths must be portable relative paths, with
no traversal or conflicting/duplicate case-insensitive spellings. Publisher
ownership, permissions and timestamps are not applied to extracted files.

`hub.archive_limits` sets finite client-side extraction budgets:

| Field | Default | What is counted |
| --- | ---: | --- |
| `max_members` | 100,000 | Physical tar headers, including extension headers; separately, distinct retained paths including implicit parent directories |
| `max_path_depth` | 32 | Components of the effective archive-relative path, including the database root |
| `max_path_bytes` | 4,096 | UTF-8 bytes of the effective archive-relative path |
| `max_total_path_bytes` | 33,554,432 (32 MiB) | Aggregate retained unique path prefixes, charging both original and casefolded spellings |
| `max_extracted_bytes` | 34,359,738,368 (32 GiB) | Aggregate effective regular-file payload sizes |
| `max_decompressed_bytes` | 42,949,672,960 (40 GiB) | Every byte emitted by gzip: headers, metadata, payloads, alignment, EOF padding and trailing/concatenated output, including parser read-ahead |

These are initial operational defaults, not thresholds measured against
production artifacts. Larger legitimate indexes require explicitly larger
limits. For example, this raises only the decompressed-byte limit to 48 GiB:

```json
{
  "hub": {
    "archive_limits": {"max_decompressed_bytes": 51539607552}
  }
}
```

Every field must be a positive finite integer: booleans, fractions, zero,
negative values, `null` and unlimited sentinels are rejected. Environment names
are `CHUNKHOUND_HUB__ARCHIVE_LIMITS__` followed by the uppercase field name, for
example `CHUNKHOUND_HUB__ARCHIVE_LIMITS__MAX_DECOMPRESSED_BYTES=51539607552`.
Configuration follows the usual precedence: explicit config and project/global
files can override environment values. These are configurable client limits,
not an administrator-enforced security policy. There are no archive-limit CLI
flags. Direct extraction callers receive the same defaults.

Existing metadata allocation guards still apply. A fixed parser guard permits
at most 32 consecutive extension headers and cannot be raised in configuration.
There is no compression-ratio heuristic; high compression is allowed within the
budgets. Trailing gzip output remains accepted within its byte budget and is
drained through the bounded reader so CRC validation still applies. Live
free-space checks remain active. Exceeding a budget refuses activation, cleans
staging when its resources can safely close, and preserves the installed
database. Manifest v2 is unchanged. These extraction limits do not add a
separate compressed-download ceiling.

## Ownership and process start behavior

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
concurrent backend writers. Initial DuckDB ownership contention before opening
a database can be retried on the same provider once contention ends, including
a later MCP database tool call. This is not an automatic startup retry loop or
a guarantee that every initialization error is retryable.

`CHUNKHOUND_USE_RUST=0` selects Python indexing but still uses the native
ownership/install protocol. LanceDB always uses the Python pipeline. ChunkHound's
parser pools explicitly use multiprocessing `spawn`, without changing the
process-global start method. `CHUNKHOUND_MP_START_METHOD` is no longer honored;
remove it from deployment configuration. There is no supported `fork` override:
forked workers could inherit ownership descriptors and prolong installation
exclusion after their parent exits. Applications embedding ChunkHound must make
their process entry point safe for spawn, including the usual
`if __name__ == "__main__":` guard around application startup.

Stop older ChunkHound clients and direct third-party database users before
pulling: they do not participate in cooperative exclusion. Hard-link aliases
and filesystems without reliable advisory locks/same-filesystem rename are
outside this guarantee; network-filesystem behavior is not assumed. Read-only
users need a usable existing coordination file or permission to create it;
they never open without coordination merely because creation is denied.

## WAL provisioning and incompatible recovery inputs

Before opening an existing DuckDB index, ChunkHound checks its stored format
and recognized recovery inputs without automatic migration/recovery. This
includes nonempty `.compact_backup`, `.compact_new`, `.old` and `.compact`
files beside the primary database. An incompatible backup can therefore block
an otherwise compatible live database: it must not be silently restored or
deleted just because the live file looks healthy.

When a WAL is present, inspection requires locally available compatible DuckDB
VSS support and HNSW persistence configuration before attaching the database
read-only. Inspection disables automatic extension installation/loading. A
missing VSS extension is an explicit provisioning issue, not evidence that the
WAL is corrupt. Provision VSS for the actual DuckDB engine version, platform
and service account, using an explicitly authorized setup step or a vetted
extension supplied by your deployment. Python and native DuckDB runtimes must
each have compatible support; installing for one engine version does not prove
that another engine can load it.

For the Python DuckDB runtime, an explicit setup command (which may download
extension code from DuckDB's configured repository) is:

```bash
uv run python - <<'PY'
import duckdb

with duckdb.connect(":memory:") as connection:
    connection.execute("INSTALL vss")
    connection.execute("LOAD vss")
    connection.execute("SET hnsw_enable_experimental_persistence = true")
PY
```

This command opens no production database and is not WAL recovery. Use it only
when the download is permitted and this is the runtime/account to provision.
For native or offline deployments, provision the matching extension through
your deployment tooling; do not assume the Python command provisions both.

If format compatibility cannot be established, automatic recovery is blocked
for both writable and read-only opens. Some healthy WALs need writes to replay
and cannot be inspected read-only. Reopening writable does not bypass this
check. Stop database users; preserve the original database together with every
WAL/recovery file and its names. Work on a separate copy with compatible
DuckDB/VSS tooling, or restore a known-good complete compatible index. Validate
the recovered copy before deliberately replacing live state. If an incompatible
backup is the blocker, have an operator identify and preserve it and select a
compatible recovery set; ChunkHound does not guess which file to discard.
Never replay a WAL against an unrelated/new base, remove a WAL to force an open,
or treat an uninspectable WAL as known corruption.

## Interrupted installation

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
This is interruption recovery, not a publisher content-freshness guarantee.

Cleanup failure after commit is a warning, not a failed pull: the new base is
usable even if old quarantine or staging work remains. Interrupted awaits wait
for their filesystem worker to finish before releasing leases/staging. There
is no permanent artifact cache.

## Cancellation and safe cleanup

Cancellation requests cooperative stop; completion may wait for safe cleanup.
Blocking handoff and reconnect work run off the event-loop thread. Existing
configuration conversion, provider concurrency detection and per-file path
preparation before native submission remain synchronous; this is not a blanket
latency guarantee for preparing arbitrarily large input lists. Native work
stops at safe boundaries, joins all pipeline stages and closes its database-use
handles before reconnecting Python access. This is not proof that every idle
OS worker has exited or every safely retained old handle has finalized.
A write window that has already prepared replacement deletions must finish the
corresponding replacement writes before stopping; new optional compaction is
skipped and necessary indexes over committed data are restored. Cancellation
is not rollback: completed writes can remain visible.

The caller ultimately receives cancellation, including after repeated cancel
requests. Ownership and in-progress guards remain held through necessary
cleanup; requesting cancellation alone does not make the database available
for installation or another operation. Cleanup failures are reported separately
and must leave an explicit unavailable state rather than advertise recovery.
If storage failure prevents a prepared replacement from finishing, do not
assume rollback or readiness; preserve the state and resolve the failure before
retrying cleanup/recovery.

There is no hard completion deadline for an already-running database operation,
network request, filesystem operation or Python callback. Already-entered
Python embedding callbacks drain their own work/retries. Offloading preserves
event-loop responsiveness for cooperative/blocking boundaries, but an arbitrary
third-party callback monopolizing the Python GIL can still delay the loop.
ChunkHound does not forcibly terminate workers or return promptly while a
detached background owner cleans up.

Cross-platform lifecycle/install support must be validated with source-matched
Windows, macOS and Linux builds. Passing a finite acceptance run is evidence
for the tested behavior, not proof of the root cause of a historical native
crash.
