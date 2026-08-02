"""Feature-flag guard for the Rust-native parse→embed→write pipeline.

Defaults to on (opt-out): the Rust pipeline is the standard path for the
parse→embed→write phase in IndexingCoordinator.process_directory(). Set
CHUNKHOUND_USE_RUST=0 to fall back to the Python pipeline. Checked at call
time (not import time), so setting the env var on the fly is sufficient.

Used by:
- IndexingCoordinator.process_directory() — gates Phase 3 (parse→embed→write)

Note: Rust file discovery (chunkhound.utils.file_patterns.scan_directory_files)
reads the same CHUNKHOUND_USE_RUST env var independently and also defaults on.
"""

import os

import chunkhound_native  # noqa: F401


def _get_use_rust() -> bool:
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload in tests).

    Defaults to True (opt-out) — see module docstring. Set
    CHUNKHOUND_USE_RUST=0 to fall back to the Python pipeline.
    """
    return os.environ.get("CHUNKHOUND_USE_RUST", "1") == "1"
