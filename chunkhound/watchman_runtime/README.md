This package hosts platform-specific Watchman runtime payloads for ChunkHound.

Current payloads are thin platform launchers over a shared Python runtime
bridge. The bridge preserves the private-sidecar ownership model and the
persistent JSON client contract (`version`, `watch-project`, `subscribe`), and
it now emits real filesystem subscription traffic for file mutations on the
subscribed root instead of only synthetic placeholder PDUs.

Packaging decision:
- Wheels that carry this package must be platform-specific.
- Publishing a Watchman-carrying `py3-none-any` wheel is forbidden.
