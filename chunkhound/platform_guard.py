"""Keep the standard library's ``platform`` module from querying WMI on Windows.

On Python 3.12, ``platform.system()``, ``win32_ver()``, ``machine()`` and
``processor()`` ask WMI first, through the private ``platform._wmi_query``,
and fall back to ``sys.getwindowsversion()``, ``cmd /c ver`` and environment
variables when that raises ``OSError``. Under CPU load a WMI query in a newly
spawned process can close one of that process's handles after it returns
(CPython gh-130727; fixed in 3.13.4 and 3.14.0b2, 3.12 got no fix):
parse-pool workers that ran ``platform.system()`` while importing ChunkHound
later failed with ``[WinError 6] The handle is invalid`` releasing their
call-queue semaphore, or released the wrong semaphore and stalled the pool.
Raising ``OSError`` instead sends ``platform`` down its own fallback, which
reports the same values. On the fixed Python versions this module does
nothing; it can go once Python 3.12 is no longer supported.

This module imports only the standard library: it runs from the package
``__init__``, so in every ChunkHound process (spawned workers re-import the
package) before anything calls ``platform``, including build environments
that have none of ChunkHound's dependencies.
"""

from __future__ import annotations

import logging
import sys
from typing import NoReturn

# The first releases whose _wmi query thread no longer reads the caller's
# finished stack frame: 3.13.4, and 3.14.0b2 (every 3.14 release is later).
_FIRST_FIXED_PYTHON = (3, 13, 4)


def _wmi_query_disabled(*_args: object, **_kwargs: object) -> NoReturn:
    """Stand-in for ``platform._wmi_query`` that sends callers to the fallback."""
    raise OSError("ChunkHound does not query WMI from platform (CPython gh-130727)")


def disable_platform_wmi_queries() -> None:
    """Make ``platform`` use its non-WMI fallback in this process.

    Does nothing off Windows or on a Python with the CPython fix, and logs at
    debug level and does nothing when this Python's ``platform`` has no
    ``_wmi_query`` (before 3.12, or a release that renamed it).
    """
    if sys.platform != "win32" or sys.version_info >= _FIRST_FIXED_PYTHON:
        return
    import platform

    if not hasattr(platform, "_wmi_query"):
        logging.getLogger(__name__).debug(
            "platform has no _wmi_query; WMI queries are not disabled"
        )
        return
    platform._wmi_query = _wmi_query_disabled
