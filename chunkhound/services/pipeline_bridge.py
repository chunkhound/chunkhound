"""Feature-flagged bridge from Python indexing pipeline to Rust DB writer.

Set CHUNKHOUND_USE_RUST=0 to force the Python path even when the native
extension is available. On by default when chunkhound_native is importable.
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

_USE_RUST = os.environ.get("CHUNKHOUND_USE_RUST", "1" if _RUST_AVAILABLE else "0") == "1"


class RustWriterBridge:
    """Thin shim that routes writes to RustDbWriter when available."""

    def __init__(self, db_config: dict) -> None:
        self._writer = None
        if not (_USE_RUST and _RUST_AVAILABLE):
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

    def write_batch(self, batch: dict) -> dict:
        assert self._writer is not None
        return self._writer.write_batch(batch)

    def needs_compaction(self) -> bool:
        assert self._writer is not None
        return self._writer.needs_compaction()

    def run_compaction(self) -> None:
        assert self._writer is not None
        self._writer.run_compaction()

    def finalize(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as exc:
                _log.warning("Error closing Rust DB writer: %s", exc)
            self._writer = None
