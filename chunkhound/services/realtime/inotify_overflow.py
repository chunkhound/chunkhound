"""Signal a kernel inotify queue overflow to the realtime service.

When the kernel inotify queue overflows, the kernel drops events and queues one
IN_Q_OVERFLOW record. watchdog 7.1.0 emits no public event for that record,
and 7.1.1 emits an EventsLostEvent that ChunkHound does not map. The FD reader
calls ``on_overflow`` after watchdog handles the record on either version.

The hook relies on two private watchdog internals:
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
    """Return an inotify observer that reports overflows, or None.

    None means no hook can be installed (not Linux, inotify unusable, or
    watchdog < 7.1, which drops the overflow record); the caller then uses
    watchdog's default observer.
    """
    if not sys.platform.startswith("linux"):
        return None
    from watchdog.utils import UnsupportedLibcError

    # The same guard watchdog's own observer selection uses on Linux.
    try:
        from watchdog.observers.inotify import InotifyObserver
        from watchdog.observers.inotify_c import InotifyFD
    except (ImportError, UnsupportedLibcError):
        return None
    if not callable(getattr(InotifyFD, "handle_event", None)):
        return None

    class _OverflowSignalingInotifyFD(InotifyFD):
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

    class _OverflowObserver(InotifyObserver):
        def __setattr__(self, name: str, value: Any) -> None:
            # watchdog assigns the reader during construction in 7.1.0 and
            # during the first schedule() in 7.1.1, before installing a watch.
            if name == "_inotify_fd" and isinstance(value, InotifyFD):
                value.__class__ = _OverflowSignalingInotifyFD
            super().__setattr__(name, value)

    observer = _OverflowObserver()
    fd = getattr(observer, "_inotify_fd", ...)
    lazy = fd is None and isinstance(getattr(observer, "_emitter_kwargs", None), dict)
    if not lazy and not isinstance(fd, _OverflowSignalingInotifyFD):
        observer.stop()
        return None
    return observer
