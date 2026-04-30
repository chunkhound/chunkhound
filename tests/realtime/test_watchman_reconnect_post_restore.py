from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.services.realtime.adapters.watchman import WatchmanRealtimeAdapter
from chunkhound.services.realtime.context import RealtimeServiceContext


class _FakeRealtimeService:
    def __init__(self, target_dir: Path) -> None:
        self.config = SimpleNamespace(target_dir=target_dir)
        self.watchman_scope_plan = None
        self.observer = None
        self._needs_resync = False
        self._using_polling = False
        self._polling_task = None
        self.resync_requests: list[tuple[str, dict[str, Any] | None]] = []
        self.errors: list[str] = []

    def _debug(self, message: str) -> None:
        del message

    def _set_error(self, message: str) -> None:
        self.errors.append(message)

    async def request_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> bool:
        self.resync_requests.append((reason, details))
        return True


@pytest.mark.asyncio
async def test_watchman_post_restore_resync_is_requested_inline(
    tmp_path: Path,
) -> None:
    service = _FakeRealtimeService(tmp_path)
    context = RealtimeServiceContext(
        service,
        sidecar_factory=lambda target_dir, debug: object(),
        session_factory=lambda metadata, sidecar, overflow_handler: object(),
        nested_mount_discoverer=lambda watch_path: (),
        junction_scope_discoverer=lambda watch_path: (),
        scope_plan_builder=lambda *args, **kwargs: None,
        subscription_name_builder=lambda **kwargs: "chunkhound-live-indexing",
        subscription_names_builder=lambda **kwargs: ("chunkhound-live-indexing",),
    )
    adapter = WatchmanRealtimeAdapter(context)

    await adapter._request_post_restore_resync()

    assert service.errors == []
    assert service.resync_requests == [
        (
            "realtime_loss_of_sync",
            {
                "backend": "watchman",
                "loss_of_sync_reason": "disconnect",
                "post_restore_reconciliation": True,
                "reconnect_state": "restored",
            },
        )
    ]
