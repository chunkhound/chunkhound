"""Whether a kernel inotify overflow can reach the realtime service.

watchdog 7.1.1 emits ``EventsLostEvent`` after it rebuilds the watches a
kernel ``IN_Q_OVERFLOW`` dropped. The realtime handler accepts that event and
requests a resync. watchdog 6 has no such event and drops the record, so
detection stays "unavailable" on that line.

watchdog is imported inside the function so importing this module never loads
it (see tests/unit/test_watchdog_import_isolation.py).
"""

from __future__ import annotations

import sys
from typing import Literal

# Whether a kernel inotify overflow reaches the service: "active" (EventsLostEvent
# is available), "unavailable" (Linux, but this watchdog has no such event) or
# "not_applicable" (not Linux).
InotifyOverflowDetection = Literal["active", "unavailable", "not_applicable"]


def inotify_overflow_detection() -> InotifyOverflowDetection:
    """Return whether this process can observe a kernel inotify overflow."""
    if not sys.platform.startswith("linux"):
        return "not_applicable"
    try:
        from watchdog import events as watchdog_events
    except ImportError:
        return "unavailable"
    if getattr(watchdog_events, "EventsLostEvent", None) is None:
        return "unavailable"
    return "active"
