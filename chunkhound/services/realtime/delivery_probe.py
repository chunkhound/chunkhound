"""Verify native event delivery without admitting probe events to indexing."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from chunkhound.utils.service_temp_names import SERVICE_TEMP_PROBE_FILENAME

# Deterministic per-root path: a crash between write and unlink leaves one
# artifact that the next start replaces instead of accumulating uuid files.
PROBE_FILENAME = SERVICE_TEMP_PROBE_FILENAME


class FilesystemDeliveryProbe:
    def __init__(self, root: Path) -> None:
        self.path = root.resolve() / PROBE_FILENAME
        self._delivered = threading.Event()

    def intercept(self, event: Any) -> bool:
        """Suppress all probe-path events, including delayed deletions."""
        paths = (event.src_path, getattr(event, "dest_path", None))
        if not any(path and Path(path).resolve() == self.path for path in paths):
            return False
        if not event.is_directory and event.event_type in {"created", "modified"}:
            self._delivered.set()
        return True

    def verify(self, abort: threading.Event, deadline: float) -> bool:
        """Write only after the observer starts; wait within its startup budget."""
        self._create_probe_exclusively("ChunkHound filesystem delivery probe\n")
        while not abort.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Watchdog filesystem delivery probe timed out")
            if self._delivered.wait(min(0.05, remaining)):
                return not abort.is_set()
        return False

    def ping(self, delivery_window: float) -> bool:
        """Rewrite the probe file; report whether watchdog delivered the event.

        Reuses the bootstrap path so every ping event stays suppressed from
        indexing. This is the only way to notice a registered-but-silent
        FSEvents stream: thread liveness cannot detect it.
        """
        self._delivered.clear()
        try:
            self._create_probe_exclusively("ChunkHound delivery ping\n")
            return self._delivered.wait(delivery_window)
        finally:
            self._delivered.clear()
            self.path.unlink(missing_ok=True)

    def _create_probe_exclusively(self, content: str) -> None:
        """Create the probe file without following or truncating a symlink.

        Exclusive "x" creation refuses an existing path, so a symlink planted
        at the probe location can never redirect the write outside the
        workspace; unlink first removes the link itself, never its target.
        """
        if self.path.exists() or self.path.is_symlink():
            self.path.unlink()
        with self.path.open("x") as probe_file:
            probe_file.write(content)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)
