"""End a multiprocessing child when the process that started it dies.

A ``ProcessPoolExecutor`` worker blocks reading its call queue. The queue is
pickled into the worker with both pipe ends, so the parent dying never closes
the pipe from the worker's point of view and the read never returns: the
worker runs forever, holding its memory, until something kills it.

multiprocessing already gives every child a sentinel for its parent (a process
handle on Windows, a pipe on POSIX) through ``multiprocessing.parent_process()``.
Waiting on it costs one blocked daemon thread and no polling, needs no
platform-specific code, and covers every start method ChunkHound uses.
"""

from __future__ import annotations

import multiprocessing
import os
import threading

_lock = threading.Lock()
_started = False


def exit_when_parent_dies() -> None:
    """Exit this process as soon as its multiprocessing parent is gone.

    Safe to call more than once; only the first call starts the watcher. Does
    nothing in a process that multiprocessing did not start. Use it as a
    ``ProcessPoolExecutor`` initializer, or at the top of a ``Process`` target.
    """
    global _started
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    with _lock:
        if _started:
            return
        _started = True

    def _watch() -> None:
        parent.join()
        # The parent cannot collect results or run cleanup any more, so there
        # is nothing to flush; leave immediately rather than through
        # interpreter shutdown, which would wait on the blocked queue reader.
        os._exit(1)

    threading.Thread(
        target=_watch, name="chunkhound-parent-watch", daemon=True
    ).start()
