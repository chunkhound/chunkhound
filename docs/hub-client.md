# Pulling a central base index

ChunkHound can install a compatible base from a private, organization-operated
hub. The publisher is trusted; the client still verifies checksums, safely
extracts the archive, and checks stored backend/format/embedding identity.
ChunkHound never publishes an index.

Use DuckDB for the pull-then-overlay workflow below; it is the operational
local-overlay target. LanceDB overlay is nonoperational/deferred: changed-file
updates can leave stale chunks searchable. This does not disable LanceDB or
remove its verified ownership, installation, no-pull and interruption/recovery
contracts. Format and checksum verification establish compatibility and artifact
integrity, not that publisher content is free of stale rows.

Configure DuckDB and a full manifest URL in `.chunkhound.json`, alongside your
existing embedding configuration:

    {"database": {"provider": "duckdb"}, "hub": {"url": "https://hub.example/catalogue?branch=main"}}

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
It replaces local index state, so local work may need re-indexing. With DuckDB,
a subsequent ordinary index overlays changed/new/deleted local files and keeps
unchanged base content through existing hash-based indexing. Moved checkouts
use existing relative-path indexing, not a new hub diff engine. Do not rely on
LanceDB overlay for fresh changed-file content while its correction is deferred.

`index` considers automatic pull only with a configured hub and known usable
embedding identity, when the target is missing or embedding-misaligned. It
runs before registry initialization and model-drift pinning. An aligned index
makes no hub request. Simulation, ignore checking, no-embeddings indexing,
MCP and search do not pull. Auto-index continues into ordinary local indexing
following installation; the same LanceDB overlay limitation applies.
Download/selection/checksum/archive/space/pre-install
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
This is interruption recovery, not a content-freshness guarantee or a repair
for stale LanceDB rows in a publisher's base; the overlay correction is deferred.

Cleanup failure after commit is a warning, not a failed pull: the new base is
usable even if old quarantine or staging work remains. Interrupted awaits wait
for their filesystem worker to finish before releasing leases/staging. There
is no permanent artifact cache. Cross-platform support claims require actual
passing Windows/POSIX platform CI results; a local Linux run alone does not
establish them. This documentation does not claim that platform CI has passed.
