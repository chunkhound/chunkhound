"""Trusted hub selection and automatic-index policy, independent of registry."""

import asyncio
import contextvars
from pathlib import Path

from loguru import logger

import chunkhound_native as native
from chunkhound.core.config.config import Config
from chunkhound.core.embedding_model_drift import (
    active_model_path,
    read_active_model,
    resolve_indexed_model,
)
from chunkhound.hub.artifact import PreparationState, prepare
from chunkhound.hub.errors import (
    HubDatabaseMismatch,
    HubInstallationInterrupted,
    HubReaderCloseError,
    HubUnavailable,
    safe_label,
)
from chunkhound.hub.inspect import inspect_database
from chunkhound.hub.install import (
    install_candidate,
    installation_failure,
    installation_target,
)
from chunkhound.hub.manifest import MatchKey, configured_key, select_manifest
from chunkhound.hub.transport import fetch_manifest
from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
    ownership_scope,
)
from chunkhound.utils.async_worker import await_worker_completion

_NATIVE_ERRORS = (
    native.DatabaseBusyError,
    native.DatabaseInterruptedError,
    native.DatabaseLayoutError,
    native.DatabaseFilesystemError,
)


def local_needs_pull(
    target: Path,
    key: MatchKey,
    *,
    ownership: DatabaseOwnership | None = None,
) -> bool:
    # Acquire before existence test: an absent target plus a marker is NOT cold.
    with ownership_scope(target, ownership) as held:
        if not target.exists():
            return True
        try:
            inventory = inspect_database(target, key.db_provider, ownership=held)
            sidecar = active_model_path(target)
            active = read_active_model(sidecar) if sidecar is not None else None
            indexed = resolve_indexed_model(inventory.groups, active)
        except DatabaseReaderCloseError:
            raise
        except _NATIVE_ERRORS:
            raise
        except Exception:
            # Unreadable/empty local state is not evidence of embedding drift.
            logger.bind(hub_client=True).info(
                "Hub auto-pull skipped: local embedding identity is unreadable"
            )
            return False
        return indexed is not None and (
            indexed.provider,
            indexed.model,
            indexed.dims,
        ) != (key.provider, key.model, key.dims)


async def _await_pull_worker(worker: asyncio.Future[bool]) -> bool:
    """Drain actual completion, preserving fatal hub outcomes over cancellation."""
    try:
        return await await_worker_completion(worker)
    except asyncio.CancelledError:
        # The shared helper has fully drained this private executor Future.
        # Keep its cancellation policy except when recovery/fatal reporting is due.
        failure = worker.exception()
        if isinstance(
            failure,
            (
                HubDatabaseMismatch,
                HubInstallationInterrupted,
                native.DatabaseInterruptedError,
                DatabaseReaderCloseError,
                HubReaderCloseError,
            ),
        ):
            raise failure from None
        raise


async def _pull(config: Config, rel: str, automatic: bool) -> bool:
    if automatic and (config.embeddings_disabled or not config.hub.url):
        return False
    if not config.hub.url:
        raise HubUnavailable("No hub manifest URL configured")
    key = configured_key(config)  # Offline; do not let drift pinning run first.
    target = installation_target(config)
    if automatic:
        # Keep actual executor completion private: shutdown cancels Tasks, not threads.
        worker = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, local_needs_pull, target, key
        )
        if not await _await_pull_worker(worker):
            return False
    selected = select_manifest(await fetch_manifest(config.hub.url), key, rel)
    logger.bind(hub_client=True).info(
        "Pulling hub base project={} identity={} embedding={} version={} revision={} "
        "built={} index_root={} artifact=index.tar.gz bytes={}",
        safe_label(selected.manifest.project_id),
        safe_label(selected.variant.key()),
        safe_label(selected.variant.embedding_id),
        rel,
        safe_label(selected.version.revision),
        safe_label(selected.version.created_at),
        safe_label(selected.manifest.index_root),
        selected.version.artifact.bytes,
    )
    state = PreparationState()
    async with prepare(
        selected,
        work_parent=target.parent,
        state=state,
        limits=config.hub.archive_limits,
    ) as candidate:

        def commit() -> bool:
            return install_candidate(
                candidate,
                target,
                key.db_provider,
                state,
                should_install=(
                    lambda ownership: local_needs_pull(target, key, ownership=ownership)
                )
                if automatic
                else None,
            )

        worker = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, commit
        )
        installed = await _await_pull_worker(worker)
    if installed:
        logger.bind(hub_client=True).info(
            "Hub base installed; run index to overlay local changes"
        )
    return installed


async def pull(config: Config, rel: str = "latest", automatic: bool = False) -> bool:
    try:
        try:
            return await _pull(config, rel, automatic)
        except DatabaseReaderCloseError as exc:
            raise HubReaderCloseError(exc) from None
        except _NATIVE_ERRORS as exc:
            # Initial ordinary inspection can fail before install.py is entered.
            raise installation_failure(exc) from None
        except OSError:
            raise HubUnavailable("Hub local filesystem is unavailable") from None
    except HubReaderCloseError as exc:
        logger.bind(hub_client=True).error("{}", str(exc))
        raise
    except HubDatabaseMismatch:
        logger.bind(hub_client=True).error(
            "Downloaded database contradicts the hub manifest; local index retained"
        )
        raise
    except HubInstallationInterrupted:
        logger.bind(hub_client=True).error(
            "Hub installation was interrupted; run explicit chunkhound pull again. "
            "Local work may require re-indexing"
        )
        raise
    except HubUnavailable as exc:
        logger.bind(hub_client=True).log(
            "INFO" if automatic else "ERROR",
            "Hub pull unavailable: {}{}",
            str(exc),
            "; continuing local indexing" if automatic else "",
        )
        return False
