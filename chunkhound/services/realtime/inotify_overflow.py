"""Signal a kernel inotify queue overflow to the realtime service.

When the kernel inotify queue overflows, the kernel drops events and queues one
IN_Q_OVERFLOW record. watchdog 7.1 rebuilds its watches on that record but
emits no event, so the service cannot know that changes were lost. This module
builds an inotify observer whose FD reader also calls ``on_overflow``.

Tested with watchdog 7.1.0. The hook relies on two private watchdog internals:
``inotify_c.InotifyFD.handle_event`` and ``InotifyObserver._inotify_fd``. On
watchdog < 7.1, on a version whose internals no longer match, or where inotify
is unusable, no hook is installed and detection reports "unavailable".
tests/realtime/test_realtime_inotify_overflow.py exercises the hook on whatever
watchdog the lock resolves to, so a version that changes these internals fails
that test when the lock moves to it.

watchdog is imported inside the function so importing this module never loads
it (see tests/unit/test_watchdog_import_isolation.py).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    from .startup import RealtimeObserver

# Whether a kernel inotify overflow reaches the service: "active" (hook
# installed), "unavailable" (Linux, but no hook could be installed) or
# "not_applicable" (not Linux).
InotifyOverflowDetection = Literal["active", "unavailable", "not_applicable"]


def create_inotify_observer(
    on_overflow: Callable[[], None],
) -> RealtimeObserver | None:
    """Return a started-FD inotify observer that reports overflows, or None.

    None means no hook can be installed (not Linux, inotify unusable, or
    watchdog < 7.1, which drops the overflow record); the caller then uses
    watchdog's default observer.
    """
    if not sys.platform.startswith("linux"):
        return None
    from watchdog.utils import UnsupportedLibcError

    # The same guard watchdog's own observer selection uses on Linux.
    try:
        from watchdog.observers import inotify_c
        from watchdog.observers.inotify import InotifyObserver
    except (ImportError, UnsupportedLibcError):
        return None
    inotify_fd_cls = getattr(inotify_c, "InotifyFD", None)
    if inotify_fd_cls is None:
        return None

    class _OverflowSignalingInotifyFD(inotify_fd_cls):  # type: ignore[misc,valid-type]
        def handle_event(self, event: Any) -> None:
            try:
                super().handle_event(event)
            finally:
                # After watchdog's synchronous watch rebuild, so the resync
                # debounce starts only once every watch is back.
                if event.is_queue_overflow:
                    try:
                        on_overflow()
                    except Exception as error:
                        logger.warning(
                            f"inotify overflow could not be reported: {error}"
                        )

    observer = InotifyObserver()
    fd = getattr(observer, "_inotify_fd", None)
    if not isinstance(fd, inotify_fd_cls):
        observer.stop()
        return None
    # No watch is scheduled yet, so no event can reach the FD thread before
    # the swap; its run() resolves self.handle_event per event.
    fd.__class__ = _OverflowSignalingInotifyFD
    return observer
