"""Feature-flag guard for the Rust-native pipeline path.

Set CHUNKHOUND_USE_RUST=0 to force the Python path even when the native
extension is available.  Checked at call time (not import time), so setting
the env var on the fly is sufficient.

Used by:
- IndexingCoordinator.process_directory() — gates Phase 3 (parse→embed→write)
- file_patterns.scan_directory_files() — gates file discovery
"""

import os

try:
    import chunkhound_native  # noqa: F401

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


def _get_use_rust() -> bool:
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload in tests)."""
    return os.environ.get("CHUNKHOUND_USE_RUST", "1" if _RUST_AVAILABLE else "0") == "1"