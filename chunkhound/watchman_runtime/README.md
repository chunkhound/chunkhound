This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current native support is intentionally narrow:
- Linux `x86_64` ships an upstream native Watchman daemon payload plus the
  shared libraries it needs at runtime.
- Windows `x86_64` ships an upstream native Watchman daemon payload plus the
  helper executables and DLLs it needs at runtime.
- macOS does not currently ship a claimed native Watchman payload in this
  package and must use explicit fallback realtime backends instead of
  `backend=watchman`.

The Python bridge remains in this package only as an internal compatibility
implementation; it does not satisfy the epic's native-daemon closure criteria.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
