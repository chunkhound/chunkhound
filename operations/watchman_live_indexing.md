# Watchman Live Indexing

This note documents the Watchman-specific live-indexing model in ChunkHound.
It is written for operators running `chunkhound mcp` or the daemon-backed MCP
flow against a local project checkout.

## Current rollout posture

- `watchman` is the default realtime backend only in installs that already ship
  a packaged native Watchman runtime. Today that means the platform-specific
  Linux `x86_64` and Windows `x86_64` wheels.
- On macOS, ChunkHound stays fallback-only in this rollout: `watchdog` is the
  default, `polling` remains an explicit fallback, and `backend=watchman`
  should be treated as unsupported because no packaged Watchman runtime slot is
  shipped for macOS.
- Source checkouts and editable installs do not embed the native payloads, so
  they default to `watchdog` unless operators explicitly set
  `realtime_backend=watchman` and accept the pinned-source hydration path.
- That source-hydration path is network-bounded and fails fast on repeated
  download timeout/connection errors instead of waiting indefinitely.
- `watchdog` and `polling` remain available as explicit fallback backends.
- The rollout gate in [Rollout gate](#rollout-gate) is satisfied only for the
  current native Linux and Windows support paths once both hosted validations
  are green.

Override the default explicitly with either config or a CLI flag when you need
to force a fallback backend:

```json
{
  "indexing": {
    "realtime_backend": "watchdog"
  }
}
```

```bash
chunkhound mcp . --realtime-backend polling
```

## Tenancy and on-disk layout

- One ChunkHound daemon owns one private Watchman sidecar.
- ChunkHound clients do not connect to Watchman directly.
- Sidecar state is project-local under `<project>/.chunkhound/watchman/`.

Expected private-sidecar artifacts:

- `runtime/`: materialized packaged Watchman binary/runtime payload
- `sock`: private Watchman Unix socket on Linux; Windows uses a private named
  pipe endpoint instead of a filesystem socket artifact
- `pid`: private Watchman pidfile
- `state`: private Watchman statefile
- `watchman.log`: Watchman sidecar log
- `metadata.json`: ChunkHound-owned sidecar metadata

The regular daemon log remains `<project>/.chunkhound/daemon.log`.

## Startup and failure behavior

- `backend=watchman` is fail-fast. ChunkHound must start the private sidecar and
  complete the Watchman session/subscription startup before the daemon is
  considered ready.
- There is no implicit fallback from failed Watchman startup to `watchdog` or
  `polling`.
- A Watchman startup failure should stop daemon publication before
  `.chunkhound/daemon.lock` is written.
- Proxy startup errors include recent daemon-log context; inspect both
  `.chunkhound/daemon.log` and `.chunkhound/watchman/watchman.log`.

Common operator expectation:

- Healthy startup: the daemon comes up, `daemon_status` reports Watchman as the
  configured/effective backend, and the Watchman sidecar/session fields move to
  running/connected.
- Failed startup: the proxy exits with a Watchman startup error, the daemon does
  not stay reachable, and there is no silent backend downgrade.

## Health checks via `daemon_status`

Use the MCP `daemon_status` tool as the primary health surface for Watchman.
Healthy Watchman-backed live indexing should normally show:

- `status == "ready"`
- `scan_progress.realtime.service_state == "running"`
- `scan_progress.realtime.configured_backend == "watchman"`
- `scan_progress.realtime.effective_backend == "watchman"`
- `scan_progress.realtime.watchman_sidecar_state == "running"`
- `scan_progress.realtime.watchman_connection_state == "connected"`
- `scan_progress.realtime.watchman_subscription_count == 1`

Fields that are useful during diagnosis:

- `startup.state`, `startup.mode`, `startup.current_phase`,
  `startup.last_error`: bounded daemon-side bootstrap summary for the current
  process
- `server_version`: the ChunkHound daemon build version serving this
  `daemon_status` response
- `startup.phases.*`: per-phase wall-clock timestamps plus monotonic
  `duration_seconds` for `initialize`, `db_connect`, `realtime_start`,
  `startup_barrier`, `daemon_publish`, and backend-specific setup such as
  `watchman_sidecar_start`, `watchman_watch_project`,
  `watchman_scope_discovery`, or `watchman_subscription_setup`
- `startup.exposure_ready_at`: in daemon mode, the first point after IPC bind
  and lock-file publication when a proxy client can connect; this remains
  `null` in stdio mode
- `startup.total_duration_seconds`: daemon-side blocking bootstrap budget only;
  it does not include the deferred initial directory scan that starts after
  realtime readiness
- `live_indexing_state` and `live_indexing_hint`: backend-neutral summary of
  whether live indexing is `uninitialized`, `idle`, `busy`, `stalled`, or
  `degraded`
- `pipeline.last_source_event_*`: the most recent source mutation observed
  before filtering/queueing
- `pipeline.last_accepted_event_*`: the most recent mutation accepted into the
  live-indexing pipeline by a real admission, not a coalesced refresh of
  already-pending work
- `pipeline.last_processing_started_*` and
  `pipeline.last_processing_completed_*`: the latest downstream processing
  progress point
- `pipeline.filtered_event_count`, `pipeline.suppressed_duplicate_count`,
  `pipeline.translation_error_count`, `pipeline.processing_error_count`: counts
  for common “connected but not converging” failure modes
- `event_pressure`: bounded hot-path diagnosis for one representative noisy path
  or subtree, including whether the culprit is `included` or `excluded`, the
  recent `events_in_window`, and how many same-path updates were collapsed into
  `coalesced_updates`
- `watchman_watch_root` and `watchman_relative_root`: the resolved
  `watch-project` mapping
- `watchman_socket_path`, `watchman_statefile_path`, `watchman_logfile_path`,
  `watchman_metadata_path`: private-sidecar locations; on Windows,
  `watchman_socket_path` reports the named-pipe endpoint string rather than a
  filesystem socket path
- `last_warning` and `last_error`: operator-visible runtime warnings/errors
- `watchman_loss_of_sync`: counters and last observed fresh-instance/recrawl/
  disconnect signal
- `resync.needs_resync`, `resync.in_progress`, `resync.last_reason`,
  `resync.last_error`: ChunkHound-side reconciliation state
- These fields summarize observed mutations and pipeline progress only.
  If no filesystem mutations have been observed yet, they do not actively prove
  end-to-end live-indexing health.

Quick interpretation guide:

- `watchman_connection_state == "connected"`: sidecar and session are both up.
- `watchman_connection_state == "sidecar_only"`: sidecar is alive, but the MCP
  session bridge is not healthy.
- `startup.state == "running"`: bootstrap is still underway; inspect
  `startup.current_phase` and `startup.phases.*.duration_seconds`.
- `startup.state == "failed"`: bootstrap did not reach readiness; inspect
  `startup.last_error` and the failed `startup.phases.*` entry.
- `live_indexing_state == "idle"`: monitoring is ready and no backlog is
  pending.
- `live_indexing_state == "busy"`: ChunkHound is actively processing changes or
  advancing accepted backlog work.
- `live_indexing_state == "stalled"`: accepted events exist, but downstream
  processing has not advanced for at least 30 seconds.
- `event_pressure.sample_scope == "excluded"` with rising
  `event_pressure.events_in_window`: an excluded noisy path is dominating watch
  traffic, but it is still being filtered before queue admission; adjust your
  explicit include/exclude config or move that subtree outside the watched root
  if the noise is expected.
- `event_pressure.sample_scope == "included"` with rising
  `event_pressure.coalesced_updates`: one in-scope file is being rewritten
  continuously; ChunkHound is collapsing same-path follow-up work to the newest
  generation instead of growing the queue without bound.
- `status == "degraded"` or `service_state == "degraded"`: inspect
  `last_error`, `watchman_loss_of_sync`, and the daemon/Watchman log files.
- Top-level `daemon_status.status` remains the coarse readiness summary.
  Stall diagnosis lives under `scan_progress.realtime.live_indexing_state`
  rather than automatically degrading the daemon summary.

## Loss of sync and resync

ChunkHound treats Watchman loss-of-sync signals as a reconciliation problem, not
as a hidden backend swap.

- Watchman fresh-instance notifications increment
  `watchman_loss_of_sync.fresh_instance_count`.
- Watchman recrawl warnings increment
  `watchman_loss_of_sync.recrawl_count`.
- Unexpected Watchman session exits increment
  `watchman_loss_of_sync.disconnect_count`.
- These events schedule a ChunkHound resync request and surface through
  `watchman_loss_of_sync.*`, `last_warning` or `last_error`, and `resync.*`.

During an incident, confirm that:

- `watchman_loss_of_sync.count` increased for the expected reason
- `resync.last_reason == "realtime_loss_of_sync"` once the request is recorded
- `resync.needs_resync` eventually clears after reconciliation completes

## Rollout gate

The Linux and Windows native Watchman rollout is gated on all of the following:

1. The `watchman-runtime-validation` job in
   `.github/workflows/smoke-tests.yml` is green on `ubuntu-latest` and
   `windows-latest`.
2. Built wheel artifacts pass
   `uv run python scripts/verify_watchman_runtime_resources.py <wheel>`.
3. A Watchman-backed daemon smoke run reaches steady state with
   `daemon_status.status == "ready"`,
   `scan_progress.realtime.service_state == "running"`,
   `watchman_sidecar_state == "running"`,
   `watchman_connection_state == "connected"`, and
   `watchman_subscription_count == 1`.
4. A forced fresh-instance, recrawl, or disconnect path proves that
   `watchman_loss_of_sync.count` increments and the resync contract surfaces
   through `resync.last_reason == "realtime_loss_of_sync"`.

Current status:

- Native Watchman rollout is currently scoped to Linux `x86_64` and Windows
  `x86_64`.
- On installed Linux and Windows platform wheels, Watchman remains the default
  realtime backend.
- In source checkouts and editable installs, `watchdog` remains the default
  unless operators explicitly opt into `watchman`.
- On macOS, no Watchman runtime slot is staged in this rollout; `watchdog` is
  the default and `polling` remains an explicit fallback.
