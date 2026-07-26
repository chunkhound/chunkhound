"""Feature-flag guard for the Rust-native parse→embed→write pipeline.

Defaults to off (opt-in): the Rust pipeline branch in
IndexingCoordinator.process_directory() has known gaps (e.g. it disconnects
the DB provider before honoring its own DuckDB-only fallback check, breaking
non-DuckDB providers; disk-usage limits, orphan-cleanup ordering, search
pagination, and compaction ordering all diverge from the Python path under
Rust). Set CHUNKHOUND_USE_RUST=1 to opt in once you've verified it works for
your setup. Checked at call time (not import time), so setting the env var on
the fly is sufficient.

Used by:
- IndexingCoordinator.process_directory() — gates Phase 3 (parse→embed→write)

Note: Rust file discovery (chunkhound.utils.file_patterns.scan_directory_files)
reads the same CHUNKHOUND_USE_RUST env var independently, but defaults on —
discovery has no equivalent known gaps.
"""

import os

import chunkhound_native  # noqa: F401


def _get_use_rust() -> bool:
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload in tests).

    Defaults to False (opt-in only) — see module docstring for why. Set
    CHUNKHOUND_USE_RUST=1 to enable the Rust parse→embed→write pipeline.
    """
    return os.environ.get("CHUNKHOUND_USE_RUST", "0") == "1"
