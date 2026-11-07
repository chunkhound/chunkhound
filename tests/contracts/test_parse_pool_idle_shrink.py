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


def test_idle_parse_pool_is_shut_down_and_recreated(monkeypatch, tmp_path):
    _reset_pool()
    # Long enough that the lease-free reads below cannot race the timer.
    monkeypatch.setattr(pb, "_PARSE_POOL_IDLE_SECONDS", 2.0, raising=False)
    src = tmp_path / "a.py"
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    try:
        first = pb.parse_batch_callback([str(src)], _CFG)
        pool = pb._parse_pool
        assert pool is not None
        workers = list(pool._processes.values())
        assert workers

        deadline = time.monotonic() + 30
        while pb._parse_pool is not None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pb._parse_pool is None, "idle parse pool was never shut down"
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive()

        assert pb.parse_batch_callback([str(src)], _CFG) == first
        assert pb._parse_pool is not None and pb._parse_pool is not pool
    finally:
        _reset_pool()


def test_parse_pool_is_never_shut_down_while_leased():
    """Overlapping batches (a scan and a resync) share one pool, an idle-timer
    callback that runs after a newer batch started (the timer firing as a batch
    arrives) leaves the pool alone, and only the current timer shuts it down.
    Timer callbacks are invoked directly, so no real timer decides the outcome
    (the default 60 s idle time never elapses during the test)."""
    _reset_pool()
    try:
        with pb._leased_parse_pool(1) as pool:
            with pb._leased_parse_pool(1) as resync_pool:
                assert resync_pool is pool
                pid = resync_pool.submit(os.getpid).result()
            assert pb._parse_pool_idle_token is None  # a lease is still held
            assert pool.submit(os.getpid).result() == pid

        stale = pb._parse_pool_idle_token  # the timer started by that release
        assert stale is not None
        with pb._leased_parse_pool(1):
            pass
        pb._shut_down_idle_parse_pool(stale)  # as if that timer fired late
        assert pb._parse_pool is pool

        pb._shut_down_idle_parse_pool(pb._parse_pool_idle_token)
        assert pb._parse_pool is None
    finally:
        _reset_pool()
