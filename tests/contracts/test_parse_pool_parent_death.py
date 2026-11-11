"""Contract test: parse-pool workers must exit when their parent process dies.

Gap this guards against: a ProcessPoolExecutor worker waits on its call queue,
whose pipe it holds both ends of, so it never sees the parent go away. When a
ChunkHound process was killed (a crashed daemon, a terminated MCP server) its
parse workers kept running indefinitely, each holding hundreds of MB of
commit, until someone killed them by hand.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Start the real process-wide parse pool, make it spawn a worker, report the
# worker's pid, then wait to be killed.
_PARENT_SCRIPT = """
import json, os, sys, time
import chunkhound.pipeline_bridge as pb
pool = pb._get_parse_pool(1)
worker_pid = pool.submit(os.getpid).result(timeout=120)
print(json.dumps({"worker": worker_pid}), flush=True)
time.sleep(600)
"""


def test_parse_pool_worker_exits_when_parent_is_killed():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", _PARENT_SCRIPT],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    worker_pid = None
    try:
        line = parent.stdout.readline()
        assert line, f"parent exited before reporting a worker (rc={parent.poll()})"
        worker_pid = json.loads(line)["worker"]
        assert psutil.pid_exists(worker_pid)

        # Hard kill: no atexit, no pool shutdown -- the parent just vanishes.
        parent.kill()
        parent.wait(timeout=30)

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and _alive(worker_pid):
            time.sleep(0.2)
        assert not _alive(worker_pid), (
            f"parse-pool worker {worker_pid} outlived its killed parent"
        )
    finally:
        if parent.poll() is None:
            parent.kill()
        if worker_pid is not None and _alive(worker_pid):
            psutil.Process(worker_pid).kill()


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
