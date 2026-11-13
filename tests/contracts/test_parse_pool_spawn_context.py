"""Contract test: the parse pool starts its workers with 'spawn', on every OS.

Gap this guards against: without an explicit context, ProcessPoolExecutor uses
the platform default, which on Linux (Python < 3.14) is 'fork'. The MCP server
creates the pool from a process that already runs threads, among them the stdio
reader that holds sys.stdin's lock. A forked worker inherits that lock held and
deadlocks in multiprocessing's _close_stdin() before it runs any task, so the
start-up scan waits forever at files_processed=0 and the server never reaches
query_ready. Measured in a Linux container (py-spy showed every worker in
_close_stdin); with a spawn context the same session was query_ready in 13.8 s.
Windows and macOS already default to spawn, which is why this only appeared on
Linux, and why the test checks the context the pool is given rather than the
platform default.
"""

from concurrent.futures import ProcessPoolExecutor

import chunkhound.pipeline_bridge as pb


def _reset_pool() -> None:
    if pb._parse_pool is not None:
        pb._parse_pool.shutdown(wait=True)
        pb._parse_pool = None


def test_parse_pool_is_created_with_a_spawn_context(monkeypatch):
    _reset_pool()
    created: list[object] = []

    class RecordingPool(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            created.append(kwargs.get("mp_context"))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(pb, "ProcessPoolExecutor", RecordingPool)
    try:
        pb._get_parse_pool(1)
    finally:
        _reset_pool()

    assert len(created) == 1
    ctx = created[0]
    assert ctx is not None, "parse pool was created without an mp_context"
    assert ctx.get_start_method() == "spawn"
