"""Configured-layout boundary and scoped native installation; no filesystem protocol."""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

import chunkhound_native as native
from chunkhound.core.config.config import Config
from chunkhound.hub.artifact import PreparationState
from chunkhound.hub.errors import HubInstallationInterrupted, HubUnavailable
from chunkhound.providers.database.ownership import DatabaseOwnership


def installation_target(config: Config) -> Path:
    database = config.database
    configured = database.path
    if configured is None or str(configured) == ":memory:" or database.read_only:
        raise HubUnavailable("Hub pull requires a writable directory-layout database")
    try:
        target = database.get_db_path()
    except (OSError, ValueError):
        raise HubUnavailable("Hub database destination is unavailable") from None
    # get_db_path is authoritative, including legacy files and .db suffix rules.
    # The configured single-file chunks.db is NOT a directory-layout install.
    if target == configured or target.parent != configured:
        raise HubUnavailable("Hub pull does not support configured database file paths")
    expected = "chunks.db" if database.provider == "duckdb" else "lancedb.lancedb"
    if target.name != expected:
        raise HubUnavailable("Hub database layout is unsupported")
    return target


def installation_failure(exc: Exception) -> HubUnavailable | HubInstallationInterrupted:
    if isinstance(exc, native.DatabaseInterruptedError):
        return HubInstallationInterrupted(
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing"
        )
    if isinstance(exc, native.DatabaseBusyError):
        return HubUnavailable("Hub destination is busy; stop database users and retry")
    if isinstance(exc, native.DatabaseLayoutError):
        return HubUnavailable("Hub database installation layout is unsupported")
    return HubUnavailable("Hub installation filesystem is unavailable")


def install_candidate(
    candidate: Path,
    target: Path,
    backend: str,
    state: PreparationState,
    *,
    should_install: Callable[[DatabaseOwnership], bool] | None = None,
) -> bool:
    try:
        lease = native.acquire_install_lease(str(target), backend)
        with DatabaseOwnership(lease) as ownership:
            if should_install is not None:
                if lease.interrupted:
                    raise native.DatabaseInterruptedError()
                if not should_install(ownership):
                    return False
            warnings = lease.install(str(candidate))
            state.committed = True
            for code in warnings:
                if code == "quarantine_cleanup_failed":
                    logger.bind(hub_client=True).warning(
                        "Hub base installed; old quarantine cleanup "
                        "could not be completed"
                    )
                else:
                    logger.bind(hub_client=True).warning(
                        "Hub base installed with a cleanup warning"
                    )
            return True
    except (
        native.DatabaseBusyError,
        native.DatabaseInterruptedError,
        native.DatabaseLayoutError,
        native.DatabaseFilesystemError,
    ) as exc:
        raise installation_failure(exc) from None
