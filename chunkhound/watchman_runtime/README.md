This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current payloads are lifecycle-capable placeholder launchers that prove the
packaging, materialization, and private-sidecar ownership path for this epic.
They support `--version` plus the private-sidecar flags used by Step 03, but
they are still not real Watchman binaries and do not provide the later session
or subscription behavior.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
