# Watchman Live Indexing

This note documents the Watchman-specific live-indexing model in ChunkHound.
It is written for operators running `chunkhound mcp` or the daemon-backed MCP
flow against a local project checkout.

## Current rollout posture

- `watchman` is available as an explicit realtime backend.
- The default backend is still `watchdog`.
- Do not treat Watchman as the default path until the rollout gate in
  [Rollout gate](#rollout-gate) is green.

Enable Watchman explicitly with either config or a CLI override:

```json
{
  "indexing": {
    "realtime_backend": "watchman"
  }
}
```

```bash
chunkhound mcp . --realtime-backend watchman
```

## Tenancy and on-disk layout

- One ChunkHound daemon owns one private Watchman sidecar.
- ChunkHound clients do not connect to Watchman directly.
- Sidecar state is project-local under `<project>/.chunkhound/watchman/`.

Expected private-sidecar artifacts:

- `runtime/`: materialized packaged Watchman binary/runtime payload
- `sock`: private Watchman socket
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

- `watchman_watch_root` and `watchman_relative_root`: the resolved
  `watch-project` mapping
- `watchman_socket_path`, `watchman_statefile_path`, `watchman_logfile_path`,
  `watchman_metadata_path`: private-sidecar artifact locations
- `last_warning` and `last_error`: operator-visible runtime warnings/errors
- `watchman_loss_of_sync`: counters and last observed fresh-instance/recrawl/
  disconnect signal
- `resync.needs_resync`, `resync.in_progress`, `resync.last_reason`,
  `resync.last_error`: ChunkHound-side reconciliation state

Quick interpretation guide:

- `watchman_connection_state == "connected"`: sidecar and session are both up.
- `watchman_connection_state == "sidecar_only"`: sidecar is alive, but the MCP
  session bridge is not healthy.
- `status == "degraded"` or `service_state == "degraded"`: inspect
  `last_error`, `watchman_loss_of_sync`, and the daemon/Watchman log files.

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

Watchman remains opt-in until all of the following are true:

1. The `watchman-runtime-validation` job in
   `.github/workflows/smoke-tests.yml` is green on `ubuntu-latest`,
   `macos-latest`, and `windows-latest`.
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

Until that gate is satisfied:

- keep Watchman behind explicit `realtime_backend=watchman`
- keep `watchdog` and `polling` as the default/fallback operator choices
- do not flip the default backend in config, docs, or release messaging
