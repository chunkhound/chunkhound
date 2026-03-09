This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current payloads are placeholder launchers that prove the packaging and
materialization path for this epic. Later steps can replace them with real
Watchman binaries without changing the loader contract.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
