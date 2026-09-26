"""Shared UTF-8 subprocess execution for site tests.

WHY: ``subprocess.run(text=True)`` decodes child pipes with the host locale
code page. On Windows that is the ANSI code page (cp1252), so the UTF-8 bytes
Node/npm emit (em dashes, middots, arrows) are silently corrupted into
mojibake. Every site test that reads child text must go through this helper so
the encoding is pinned in exactly one place. Static guard:
``tests/site/test_subprocess_encoding_guard.py``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


def run_text_process(
    args: Sequence[str],
    *,
    cwd: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a child process and decode its captured output as UTF-8.

    Accepted trade-off: ``errors="replace"`` swaps a loud UnicodeDecodeError on a
    single stray byte for mojibake that stays visible in assertion output. The
    UTF-8 contract itself is pinned byte-exactly by the guard test
    (``test_subprocess_encoding_guard.py``).
    """
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
        env=env,
        timeout=timeout,
        check=check,
    )
