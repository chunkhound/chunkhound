"""Feature-flag guard for CHUNKHOUND_USE_RUST, the single on/off switch for
every Rust-native code path (parse→embed→write pipeline and file discovery).

Defaults to on (opt-out): Rust is the standard path. Set CHUNKHOUND_USE_RUST=0
to fall back to the pure-Python equivalents. The env var itself is checked at
call time (not import time), so setting it on the fly is sufficient and
there's no module-reload concern in tests.

chunkhound_native is imported eagerly below, regardless of the flag value.
It's a hard dependency of this package (see AGENTS.md) — Rust-accelerated
file discovery runs unconditionally, so there is no supported "native
extension missing" fallback for CHUNKHOUND_USE_RUST to provide. Import
failure here is intentional fail-fast behavior, not a bug.

This default is a deliberate, settled decision — not an open question for
reviewers to re-raise. Rust is the intended standard path, not an
experimental opt-in.

Used by:
- IndexingCoordinator.process_directory() — gates Phase 3 (parse→embed→write)
- chunkhound.utils.file_patterns.scan_directory_files() — gates Rust-accelerated
  directory scanning

Lives in chunkhound.utils (not providers.database) so both call sites — one in
the DB provider layer, one in the file-discovery layer — share this single
reader instead of each keeping an independent copy that could drift out of
sync with the other.
"""

import os

import chunkhound_native  # noqa: F401


def _get_use_rust() -> bool:
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload
    in tests).

    Defaults to True (opt-out) — see module docstring. Set
    CHUNKHOUND_USE_RUST=0 to fall back to the Python pipeline.
    """
    return os.environ.get("CHUNKHOUND_USE_RUST", "1") == "1"
