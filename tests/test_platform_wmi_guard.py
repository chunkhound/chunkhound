"""Importing ChunkHound keeps ``platform`` off WMI without changing its answers.

A WMI query under load could close a spawned parse worker's call-queue
semaphore handle (CPython gh-130727); see chunkhound/platform_guard.py. Each
check runs in a fresh interpreter, because ``platform`` caches what it learns
and the guard acts at package import.
"""

import json
import subprocess
import sys

import pytest

_PLATFORM_REPORT = (
    "import json, platform; print(json.dumps([platform.system(), "
    "platform.release(), platform.version(), platform.machine(), "
    "list(platform.win32_ver()), platform.processor(), platform.platform()]))"
)

_COUNT_WMI_QUERIES = """
import _wmi
queries = []
_exec_query = _wmi.exec_query
def _counting_exec_query(query):
    queries.append(query)
    return _exec_query(query)
_wmi.exec_query = _counting_exec_query
import chunkhound
import platform
platform.win32_ver()
platform.machine()
platform.processor()
print(len(queries))
"""


def _run_python(code: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return completed.stdout.strip()


def test_a_chunkhound_process_never_queries_wmi_through_platform() -> None:
    if sys.version_info >= (3, 13, 4):
        pytest.skip(
            "CPython 3.13.4 and 3.14.0b2 fixed gh-130727; the guard is off there"
        )
    try:
        import _wmi  # noqa: F401
    except ImportError:
        pytest.skip("this Python's platform module does not use WMI")

    assert _run_python(_COUNT_WMI_QUERIES) == "0"


def test_platform_answers_are_unchanged_by_importing_chunkhound() -> None:
    plain = json.loads(_run_python(_PLATFORM_REPORT))
    guarded = json.loads(_run_python(f"import chunkhound; {_PLATFORM_REPORT}"))

    assert guarded == plain
