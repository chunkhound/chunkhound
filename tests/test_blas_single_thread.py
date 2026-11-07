"""ChunkHound's entry points run OpenBLAS single-threaded unless told otherwise.

numpy's wheels bundle OpenBLAS, which starts a thread per core on import and commits
a work buffer for each: on a 10-core Windows machine `import numpy` alone committed
308 MB, against 18 MB single-threaded. Every ChunkHound process (CLI, MCP front,
daemon, and each spawned parse worker) paid it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("threadpoolctl")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Import an entry module first, as the console scripts do, then report OpenBLAS's
# thread count from the library itself.
_PROBE = """
import importlib, json, sys
importlib.import_module(sys.argv[1])
import numpy
from threadpoolctl import threadpool_info
print(json.dumps([i["num_threads"] for i in threadpool_info()
                  if i.get("internal_api") == "openblas"]))
"""


def _openblas_threads(entry: str, extra_env: dict[str, str]) -> list[int]:
    env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_NUM_THREADS"}
    env.update(extra_env)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    out = subprocess.run(
        [sys.executable, "-c", _PROBE, entry],
        capture_output=True, text=True, env=env, check=True, timeout=120,
    ).stdout.strip().splitlines()[-1]
    return json.loads(out)


@pytest.mark.parametrize(
    "entry", ["chunkhound.api.cli.main", "chunkhound.mcp_server.stdio"]
)
def test_entry_points_run_openblas_single_threaded(entry: str) -> None:
    threads = _openblas_threads(entry, {})
    assert threads and set(threads) == {1}, threads


def test_an_explicit_openblas_setting_wins() -> None:
    explicit = {"OPENBLAS_NUM_THREADS": "2"}
    threads = _openblas_threads("chunkhound.api.cli.main", explicit)
    assert threads and set(threads) == {2}, threads
