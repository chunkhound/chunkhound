"""Contract: the parse pool shutdown really stops its workers.

Parse workers inherit the sandbox cwd and environment. The test harness must
stop them before it deletes that state, so shutdown has to be a real public
operation, not a private attribute swap.
"""

from __future__ import annotations

import os
import time

import psutil

from chunkhound import pipeline_bridge


def _worker_pid(_: None) -> int:
    return os.getpid()


def _wait_until_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not psutil.pid_exists(pid):
            return True
        time.sleep(0.05)
    return not psutil.pid_exists(pid)


def test_shutdown_parse_pool_stops_workers() -> None:
    pipeline_bridge.shutdown_parse_pool()
    pool = pipeline_bridge._get_parse_pool(1)
    worker_pid = pool.submit(_worker_pid, None).result(timeout=30)
    assert psutil.pid_exists(worker_pid)

    pipeline_bridge.shutdown_parse_pool()

    assert _wait_until_gone(worker_pid), "parse worker survived shutdown"
