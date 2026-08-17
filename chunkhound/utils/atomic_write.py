"""Atomic JSON write helper shared across daemon discovery and remote-config
persistence.

Truncate-then-write (``open(path, "w")`` + ``json.dump``) leaves the target
in a partially-written state on SIGKILL, disk-full, or concurrent writers.
This helper writes to a sibling temp file, then commits via
``Path.replace`` — atomic on POSIX; on Windows the ``replace`` step is
retried through a short backoff to absorb transient ``PermissionError``
from antivirus scanners or other short-lived open-handle contention.

Public entry point: :func:`write_json_atomically`.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_WINDOWS_REPLACE_RETRIES = 20
_WINDOWS_REPLACE_RETRY_DELAY = 0.01


def _is_windows_platform() -> bool:
    return sys.platform == "win32"


def write_json_atomically(
    path: Path,
    data: dict[str, Any],
    *,
    private: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
    trailing_newline: bool = False,
) -> None:
    """Write ``data`` as JSON to ``path`` atomically via a sibling temp file.

    ``indent`` and ``sort_keys`` are forwarded to ``json.dump``; other
    ``json.dump`` formatting is fixed to keep the helper small.
    ``private=True`` chmods the final file to ``0o600`` on POSIX (Windows
    no-op) — for lock/registry files that may embed secrets.
    ``trailing_newline=True`` appends a final ``\\n`` for human-readable
    config files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent, sort_keys=sort_keys)
            if trailing_newline:
                f.write("\n")
        for attempt in range(_WINDOWS_REPLACE_RETRIES):
            try:
                tmp_path.replace(path)
                break
            except PermissionError:
                if (
                    not _is_windows_platform()
                    or attempt >= _WINDOWS_REPLACE_RETRIES - 1
                ):
                    raise
                time.sleep(_WINDOWS_REPLACE_RETRY_DELAY)
    except Exception:
        # Best-effort tmp cleanup; swallow unlink errors so the original
        # write failure is what surfaces to the caller.
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass
        raise
    if private and sys.platform != "win32":
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
