"""Feature-flagged bridge from Python indexing pipeline to Rust DB writer.

Set CHUNKHOUND_USE_RUST=0 to force the Python path even when the native
extension is available. Checked at instantiation time (not import time),
so setting the env var before creating RustWriterBridge is sufficient.
"""

import os
import logging

_log = logging.getLogger(__name__)

try:
    from chunkhound_native import RustDbWriter as _RustDbWriter
    _RUST_AVAILABLE = True
except ImportError:
    _RustDbWriter = None
    _RUST_AVAILABLE = False


def _get_use_rust() -> bool:
    """Read the CHUNKHOUND_USE_RUST env var at call time (avoids module-reload in tests)."""
    return os.environ.get("CHUNKHOUND_USE_RUST", "1" if _RUST_AVAILABLE else "0") == "1"


class RustWriterBridge:
    """Thin shim that routes writes to RustDbWriter when available.

    Phase 0 (current): intentionally NOT wired into any production code path.
    The class and the native extension are exercised only by the tests in
    tests/test_rust_db_writer.py.  No production writes go through this class yet.

    Phase 1 (TODO): wire into SerialDatabaseExecutor
    (chunkhound/providers/database/serial_executor.py) to replace the Python
    DuckDB write path — see the db_writer branch for the implementation plan.

    Error-behaviour contract
    ------------------------
    ``write_batch``, ``needs_compaction``, and ``run_compaction`` raise
    ``RuntimeError`` when the writer is unavailable — callers must guard with
    ``available()`` before calling them.

    ``drop_all_hnsw_indexes`` and ``ensure_all_hnsw_indexes`` are intentional
    no-ops when unavailable; they are optional bulk-mode performance hints and
    safe to call unconditionally.
    """

    def __init__(self, db_config: dict) -> None:
        self._writer = None
        if not (_get_use_rust() and _RUST_AVAILABLE):
            _log.debug("Rust DB writer disabled (CHUNKHOUND_USE_RUST=0 or native not built)")
            return
        try:
            self._writer = _RustDbWriter(db_config)
            self._writer.open()
            _log.debug("Rust DB writer initialised")
        except Exception as exc:
            _log.warning("Failed to initialise Rust DB writer, falling back to Python: %s", exc)
            self._writer = None

    def available(self) -> bool:
        return self._writer is not None

    def __enter__(self) -> "RustWriterBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.finalize()

    def write_batch(self, batch: dict) -> dict:
        if self._writer is None:
            raise RuntimeError("RustWriterBridge is not available")
        return self._writer.write_batch(batch)

    def needs_compaction(self) -> bool:
        if self._writer is None:
            raise RuntimeError("RustWriterBridge is not available")
        return self._writer.needs_compaction()

    def run_compaction(self) -> None:
        if self._writer is None:
            raise RuntimeError("RustWriterBridge is not available")
        self._writer.run_compaction()

    def drop_all_hnsw_indexes(self) -> None:
        """No-op when unavailable — callers use as an optional bulk-mode hint."""
        if self._writer is not None:
            self._writer.drop_all_hnsw_indexes()

    def ensure_all_hnsw_indexes(self) -> None:
        """No-op when unavailable — callers use as an optional bulk-mode hint."""
        if self._writer is not None:
            self._writer.ensure_all_hnsw_indexes()

    def finalize(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                _log.warning("Error closing Rust DB writer: %s", exc)
            self._writer = None
