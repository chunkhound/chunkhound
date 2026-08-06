"""Feature-flag guard for CHUNKHOUND_USE_RUST, the single on/off switch for
every Rust-native code path (parse→embed→write pipeline and file discovery).

Defaults to on (opt-out): Rust is the standard path. Set CHUNKHOUND_USE_RUST=0
to fall back to the pure-Python equivalents. Checked at call time (not import
time), so setting the env var on the fly is sufficient and there's no
module-reload concern in tests.

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
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload in tests).

    Defaults to True (opt-out) — see module docstring. Set
    CHUNKHOUND_USE_RUST=0 to fall back to the Python pipeline.
    """
    return os.environ.get("CHUNKHOUND_USE_RUST", "1") == "1"
