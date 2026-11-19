"""Platform observer selection for the watchdog backend."""

from __future__ import annotations


def test_watchdog_factory_uses_the_platform_observer(monkeypatch) -> None:
    """A missing inotify hook must not drop the platform's own observer.

    macOS needs its rescan-aware FSEvents observer, whose emitter counts
    silently dropped native batches; a plain ``Observer`` would make the
    dropped-batch rescan path unreachable.
    """
    import chunkhound.services.realtime.adapters.watchdog as adapters
    import chunkhound.services.realtime.inotify_overflow as inotify
    import chunkhound.services.realtime.startup as startup

    sentinel = object()
    monkeypatch.setattr(inotify, "create_inotify_observer", lambda on_overflow: None)
    monkeypatch.setattr(adapters, "create_watchdog_observer", lambda: sentinel)

    observer, detection = startup._create_watchdog_observer(lambda: None)

    assert observer is sentinel
    assert detection in {"unavailable", "not_applicable"}
