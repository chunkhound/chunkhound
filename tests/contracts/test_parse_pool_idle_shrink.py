"""Contract test: the parse pool is released once idle, and never while in use.

The pool is a process-wide singleton used only by bulk parsing (start-up scans,
resyncs, `index`). Kept alive forever, a long-lived MCP server holds
max_workers idle worker processes for its whole session. After
_PARSE_POOL_IDLE_SECONDS with no batch in flight it is shut down and the next
batch creates a new one.
"""

import os
import time
from types import SimpleNamespace

import chunkhound.pipeline_bridge as pb

_CFG = SimpleNamespace(
    detect_embedded_sql=False,
    per_file_timeout_secs=0.0,
    per_file_timeout_min_size_kb=128,
    config_file_size_threshold_kb=20,
    parse_thread_pool_size=1,
)


def _reset_pool() -> None:
    shut_down = getattr(pb, "_shut_down_parse_pool_at_exit", None)
    if shut_down is not None:
        shut_down()
    elif pb._parse_pool is not None:
        pb._parse_pool.shutdown(wait=True)
        pb._parse_pool = None


def _wait_for_no_pool(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while pb._parse_pool is not None and time.monotonic() < deadline:
        time.sleep(0.05)


def test_idle_parse_pool_is_shut_down_and_recreated(monkeypatch, tmp_path):
    _reset_pool()
    monkeypatch.setattr(pb, "_PARSE_POOL_IDLE_SECONDS", 0.3, raising=False)
    src = tmp_path / "a.py"
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    try:
        first = pb.parse_batch_callback([str(src)], _CFG)
        pool = pb._parse_pool
        workers = list(pool._processes.values())
        assert workers

        _wait_for_no_pool(15)
        assert pb._parse_pool is None, "idle parse pool was never shut down"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive()

        assert pb.parse_batch_callback([str(src)], _CFG) == first
        assert pb._parse_pool is not None and pb._parse_pool is not pool
    finally:
        _reset_pool()


def test_parse_pool_is_never_shut_down_while_leased(monkeypatch):
    """Overlapping batches (a scan and a resync) keep the pool alive past the
    idle time, and an idle-timer callback that runs after a newer batch
    started (the timer firing as a batch arrives) leaves the pool alone."""
    _reset_pool()
    monkeypatch.setattr(pb, "_PARSE_POOL_IDLE_SECONDS", 0.3)
    try:
        with pb._leased_parse_pool(1) as pool:
            with pb._leased_parse_pool(1) as resync_pool:
                assert resync_pool is pool
                pid = resync_pool.submit(os.getpid).result()
            time.sleep(1.0)  # well past the idle time, outer lease still held
            assert pb._parse_pool is pool
            assert pool.submit(os.getpid).result() == pid

        stale = pb._parse_pool_idle_token  # the timer started by that release
        with pb._leased_parse_pool(1):
            pass
        pb._shut_down_idle_parse_pool(stale)  # as if that timer fired late
        assert pb._parse_pool is pool

        _wait_for_no_pool(15)
        assert pb._parse_pool is None
    finally:
        _reset_pool()
