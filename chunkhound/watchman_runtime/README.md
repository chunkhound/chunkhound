This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current payloads are lifecycle-capable placeholder launchers that prove the
packaging, materialization, and private-sidecar ownership path for this epic.
They support `--version`, the private-sidecar flags used by Step 03, and the
minimal persistent JSON client flow needed by Step 06 (`version`,
`watch-project`, `subscribe`). They are still not real Watchman binaries and do
not provide real filesystem subscription traffic.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
