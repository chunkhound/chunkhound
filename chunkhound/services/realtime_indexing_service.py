"""Compatibility module alias for the packaged realtime implementation."""

# Transitional shim: keep the legacy import path working until a later
# dedicated import-cleanup follow-up removes it.

from loguru import logger

from .realtime import (
    HotPathPressure,
    PollingRealtimeAdapter,
    QueueResultCallback,
    RealtimeIndexingService,
    RealtimeMonitorAdapter,
    RealtimeMutation,
    RealtimePathFilter,
    RealtimePathFilterSettings,
    RealtimeStartupStatusTracker,
    SimpleEventHandler,
    WatchdogRealtimeAdapter,
    WatchmanRealtimeAdapter,
    normalize_file_path,
)
from .realtime.service import (
    PrivateWatchmanSidecar,
    WatchmanCliSession,
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    build_watchman_subscription_name_for_scope,
    build_watchman_subscription_names_for_scope_plan,
    discover_nested_linux_mount_roots,
    discover_nested_windows_junction_scopes,
)
from .realtime.startup import default_realtime_backend_for_current_install

__all__ = [name for name in globals() if not name.startswith("_")]
