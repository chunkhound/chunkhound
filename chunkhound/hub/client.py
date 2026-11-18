"""Trusted hub selection and automatic-index policy, independent of registry."""

import asyncio
import contextvars
from dataclasses import dataclass
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
    FATAL_HUB_ERRORS,
    HubAuthorizationError,
    HubDatabaseMismatch,
    HubInstallationInterrupted,
    HubLocalStateError,
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
from chunkhound.hub.manifest import (
    Manifest,
    MatchKey,
    configured_key,
    parse_manifest,
    select_manifest,
)
from chunkhound.hub.transport import fetch_manifest
from chunkhound.hub.trust import (
    HubTrustError,
    checkout_digest,
    is_trusted,
    manifest_url_digest,
)
from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
    ownership_scope,
)
from chunkhound.utils.async_worker import await_worker_completion


@dataclass(frozen=True)
class ValidatedHubSource:
    manifest: Manifest
    checkout_digest: str
    url_digest: str


async def _fetch_bound_manifest(
    url: str, checkout_binding: str, url_binding: str
) -> ValidatedHubSource:
    return ValidatedHubSource(
        manifest=parse_manifest(await fetch_manifest(url)),
        checkout_digest=checkout_binding,
        url_digest=url_binding,
    )


async def fetch_validated_manifest(config: Config) -> ValidatedHubSource:
    url = config.hub.url
    if not url:
        raise HubUnavailable("No hub manifest URL configured")
    checkout = config.target_dir or Path.cwd()
    try:
        checkout_binding = checkout_digest(checkout)
        url_binding = manifest_url_digest(url)
    except OSError:
        raise HubLocalStateError("Hub checkout identity is unavailable") from None
    return await _fetch_bound_manifest(url, checkout_binding, url_binding)


_NATIVE_ERRORS = (
    native.DatabaseBusyError,
    native.DatabaseInterruptedError,
    native.DatabaseLayoutError,
    native.DatabaseFilesystemError,
)

_INSTALL_SUFFIXES: dict[str, tuple[str, ...]] = {
    "duckdb": (
        "",
        ".wal",
        ".wal.corrupt",
        ".compact_backup",
        ".compact_backup.wal",
        ".compact_new",
        ".compact_new.wal",
        ".old",
        ".old.wal",
        ".compact",
        ".compact.wal",
        ".swap_intent",
        ".export_tmp",
        ".root.json",
        ".embedding.json",
    ),
    "lancedb": ("", ".embedding.json"),
}


def automatic_inventory_paths(target: Path, backend: str) -> tuple[Path, ...]:
    try:
        suffixes = _INSTALL_SUFFIXES[backend]
    except KeyError:
        raise HubLocalStateError("Hub database layout is unsupported") from None
    return (Path(f"{target}.chunkhound-install.json"),) + tuple(
        Path(f"{target}{suffix}") for suffix in suffixes
    )


def automatic_target_appears_cold(target: Path, backend: str) -> bool:
    """Probe without ownership or mutation; uncertainty suppresses prompting."""
    for path in automatic_inventory_paths(target, backend):
        try:
            path.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return False
        return False
    return True


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


def automatic_target_absent(
    target: Path,
    backend: str,
    ownership: DatabaseOwnership | None = None,
) -> bool:
    """Return true only for exhaustive absence under database ownership."""
    with ownership_scope(target, ownership):
        paths = automatic_inventory_paths(target, backend)
        primary = Path(f"{target}{_INSTALL_SUFFIXES[backend][0]}")
        try:
            primary.lstat()
        except FileNotFoundError:
            pass
        except OSError:
            raise HubLocalStateError(
                "Hub local filesystem is unavailable"
            ) from None
        else:
            return False

        # The ownership boundary rejects a surviving installation marker first.
        # Any other state without its primary is recoverable or ambiguous, not cold.
        for path in paths[1:]:
            try:
                path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                raise HubLocalStateError(
                    "Hub local filesystem is unavailable"
                ) from None
            raise HubLocalStateError(
                "Hub local database state is incomplete or requires recovery"
            )
        return True


async def _await_pull_worker(worker: asyncio.Future[bool]) -> bool:
    """Drain actual completion, preserving fatal hub outcomes over cancellation."""
    try:
        return await await_worker_completion(worker)
    except asyncio.CancelledError:
        # The shared helper has fully drained this private executor Future.
        # Keep its cancellation policy except when recovery/fatal reporting is due.
        failure = worker.exception()
        if failure is not None and isinstance(
            failure,
            (
                *FATAL_HUB_ERRORS,
                native.DatabaseInterruptedError,
                DatabaseReaderCloseError,
            ),
        ):
            raise failure from None
        raise


def automatic_pull_eligible(
    config: Config, *, mcp_mode: bool, no_embeddings: bool
) -> bool:
    """Return whether automatic cold installation may be considered."""
    return (
        not mcp_mode
        and not no_embeddings
        and not config.embeddings_disabled
        and bool(config.hub.url)
    )


async def _pull(
    config: Config,
    rel: str,
    automatic: bool,
    *,
    mcp_mode: bool,
    no_embeddings: bool,
    source: ValidatedHubSource | None = None,
) -> bool:
    if automatic and not automatic_pull_eligible(
        config, mcp_mode=mcp_mode, no_embeddings=no_embeddings
    ):
        return False
    url = config.hub.url
    if not url:
        raise HubUnavailable("No hub manifest URL configured")
    checkout = config.target_dir or Path.cwd()
    try:
        authorized_checkout_digest = checkout_digest(checkout)
        authorized_url_digest = manifest_url_digest(url)
    except OSError:
        raise HubLocalStateError("Hub checkout identity is unavailable") from None
    if source is not None and (
        source.checkout_digest != authorized_checkout_digest
        or source.url_digest != authorized_url_digest
    ):
        raise HubUnavailable("Validated hub source no longer matches configuration")
    if automatic and source is None:
        try:
            if not is_trusted(checkout, url):
                return False
        except HubTrustError:
            raise HubAuthorizationError(
                "Hub source authorization could not be verified; automatic hub "
                "use was not started"
            ) from None
    target = installation_target(config)
    key = configured_key(config)  # Offline; do not let drift pinning run first.
    if automatic:
        # Keep actual executor completion private: shutdown cancels Tasks, not threads.
        worker = asyncio.get_running_loop().run_in_executor(
            None,
            contextvars.copy_context().run,
            automatic_target_absent,
            target,
            key.db_provider,
        )
        if not await _await_pull_worker(worker):
            return False
    if source is not None:
        try:
            current_checkout_digest = checkout_digest(
                config.target_dir or Path.cwd()
            )
        except OSError:
            raise HubLocalStateError("Hub checkout identity is unavailable") from None
        if (
            current_checkout_digest != authorized_checkout_digest
            or config.hub.url != url
        ):
            return False
        validated = source
    elif automatic:
        try:
            current_checkout_digest = checkout_digest(
                config.target_dir or Path.cwd()
            )
        except OSError:
            raise HubLocalStateError("Hub checkout identity is unavailable") from None
        if (
            current_checkout_digest != authorized_checkout_digest
            or config.hub.url != url
        ):
            return False
        try:
            if not is_trusted(config.target_dir or Path.cwd(), url):
                return False
        except HubTrustError:
            raise HubAuthorizationError(
                "Hub source authorization could not be verified; automatic hub "
                "use was not started"
            ) from None
        validated = await _fetch_bound_manifest(
            url,
            authorized_checkout_digest,
            authorized_url_digest,
        )
        try:
            current_checkout_digest = checkout_digest(
                config.target_dir or Path.cwd()
            )
        except OSError:
            raise HubLocalStateError("Hub checkout identity is unavailable") from None
        if (
            current_checkout_digest != authorized_checkout_digest
            or config.hub.url != url
        ):
            return False
    else:
        validated = await fetch_validated_manifest(config)
    selected = select_manifest(validated.manifest, key, rel)
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
        download_timeout_seconds=config.hub.download_timeout_seconds,
    ) as candidate:

        def commit() -> bool:
            return install_candidate(
                candidate,
                target,
                key.db_provider,
                state,
                should_install=(
                    lambda ownership: automatic_target_absent(
                        target,
                        key.db_provider,
                        ownership=ownership,
                    )
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


async def pull(
    config: Config,
    rel: str = "latest",
    automatic: bool = False,
    *,
    mcp_mode: bool = False,
    no_embeddings: bool = False,
    source: ValidatedHubSource | None = None,
) -> bool:
    try:
        try:
            return await _pull(
                config,
                rel,
                automatic,
                mcp_mode=mcp_mode,
                no_embeddings=no_embeddings,
                source=source,
            )
        except DatabaseReaderCloseError as exc:
            raise HubReaderCloseError(exc) from None
        except _NATIVE_ERRORS as exc:
            # Initial ordinary inspection can fail before install.py is entered.
            raise installation_failure(exc) from None
        except OSError:
            raise HubLocalStateError("Hub local filesystem is unavailable") from None
    except HubAuthorizationError as exc:
        logger.bind(hub_client=True).error("{}", str(exc))
        raise
    except HubLocalStateError:
        logger.bind(hub_client=True).error(
            "Hub local installation state is unsafe; local indexing was not started"
        )
        raise
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
            "Hub pull unavailable: {}",
            str(exc),
        )
        return False
