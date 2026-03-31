from .scope import (
    WatchmanScopePlan,
    WatchmanSubscriptionScope,
    build_watchman_scope_plan,
    discover_nested_linux_mount_roots,
    discover_nested_windows_junction_scopes,
)
from .session import (
    WatchmanCliSession,
    WatchmanSessionSetup,
    build_watchman_base_command,
    build_watchman_subscription_names_for_scope_plan,
    build_watchman_subscription_name_for_scope,
)
from .sidecar import (
    PrivateWatchmanSidecar,
    WatchmanSidecarMetadata,
    WatchmanSidecarPaths,
)

__all__ = [
    "PrivateWatchmanSidecar",
    "WatchmanCliSession",
    "WatchmanScopePlan",
    "WatchmanSidecarMetadata",
    "WatchmanSidecarPaths",
    "WatchmanSessionSetup",
    "WatchmanSubscriptionScope",
    "build_watchman_base_command",
    "build_watchman_subscription_names_for_scope_plan",
    "build_watchman_subscription_name_for_scope",
    "build_watchman_scope_plan",
    "discover_nested_linux_mount_roots",
    "discover_nested_windows_junction_scopes",
]
