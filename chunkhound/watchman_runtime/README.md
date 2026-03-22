This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current native support is intentionally narrow:
- Linux `x86_64` ships an upstream native Watchman daemon payload plus the
  shared libraries it needs at runtime.
- Windows `x86_64` ships an upstream native Watchman daemon payload plus the
  helper executables and DLLs it needs at runtime.
- macOS intentionally ships no Watchman runtime slot in this rollout and must
  stay on the fallback realtime backends instead of `backend=watchman`
  (`watchdog` by default, `polling` as an explicit fallback).

The Python bridge remains in this package only as an internal compatibility
implementation; it does not satisfy the epic's native-daemon closure criteria.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
- Building a Watchman-carrying wheel on any host outside the declared Linux and
  Windows support matrix must fail instead of silently producing a generic
  artifact. macOS remains fallback-only until a separate macOS-native follow-up
  exists.
- Source checkouts and editable installs do not ship these native payloads, so
  they default to fallback realtime backends unless operators explicitly opt
  into `backend=watchman` and allow runtime hydration from the pinned sources.
- Source-hydration downloads use bounded timeout/retry settings and fail fast
  on repeated network stalls instead of hanging indefinitely.
