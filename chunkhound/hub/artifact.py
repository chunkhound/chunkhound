"""Safely materialize and verify disposable hub candidates, without installation."""

import gzip
import ntpath
import sys
import tarfile
import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from loguru import logger

from chunkhound.core.embedding_model_drift import resolve_indexed_model
from chunkhound.hub.errors import HubDatabaseMismatch, HubUnavailable
from chunkhound.hub.inspect import inspect_database
from chunkhound.hub.manifest import Selected
from chunkhound.hub.transport import CHUNK, download, require_space
from chunkhound.providers.database.ownership import DatabaseReaderCloseError

_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{i}" for prefix in ("COM", "LPT") for i in "123456789¹²³"
}


def extract(archive: Path, stage: Path, backend: str) -> Path:
    """Stream only portable regular files/directories into a caller-owned stage.

    The stage must be a private directory; this is not cross-process exclusion
    against another writer changing it during extraction. Permissions, ownership
    and timestamps from the publisher are never applied. On failure, the caller
    owns removal of any partially extracted files (``prepare`` does this).
    """
    if backend not in {"duckdb", "lancedb"}:
        raise HubDatabaseMismatch("Hub archive backend is unsupported")
    expected = "chunks.db" if backend == "duckdb" else "lancedb.lancedb"
    seen: set[str] = set()
    # Include implicit parents: A/x and a/y collide on portable filesystems too.
    paths: dict[str, tuple[str, bool]] = {}
    metadata_remaining = CHUNK

    class BoundedTarInfo(tarfile.TarInfo):
        @classmethod
        def fromtarfile(cls, tar: tarfile.TarFile) -> "BoundedTarInfo":
            try:
                return super().fromtarfile(tar)
            # Typeshed omits these concrete stdlib HeaderError subclasses.
            except (
                tarfile.InvalidHeaderError,  # type: ignore[attr-defined]
                tarfile.TruncatedHeaderError,  # type: ignore[attr-defined]
            ):
                # TarFile.next silently treats these as EOF after a member.
                # Keep genuine zero-block/clean boundary EOF semantics, but do
                # not let broken headers (including PAX records) end iteration.
                raise HubUnavailable("Hub archive contains a corrupt header") from None

        def _proc_member(self, tar: tarfile.TarFile) -> tarfile.TarInfo:
            # tarfile's extension hook runs BEFORE eager metadata/sparse parsing.
            # Bound aggregate metadata for each yielded member, including chains.
            nonlocal metadata_remaining
            # Global PAX state survives yielded members and is copied by the
            # parser. Check it on recursive entry, after a global update but
            # BEFORE any subsequent metadata/member copies. Each update itself
            # is bounded below; repeated replacement is not a lifetime quota.
            global_headers = tar.pax_headers or {}
            retained = sys.getsizeof(global_headers) + sum(
                sys.getsizeof(key) + sys.getsizeof(value)
                for key, value in global_headers.items()
            )
            if retained > CHUNK:
                raise HubUnavailable("Hub archive metadata exceeds size limit")
            if self.type == tarfile.GNUTYPE_SPARSE:
                raise HubUnavailable("Hub archive contains an unsafe member")
            if self.type in {
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.SOLARIS_XHDTYPE,
                tarfile.GNUTYPE_LONGNAME,
                tarfile.GNUTYPE_LONGLINK,
            }:
                metadata_remaining -= max(512, self.size)
                if self.size < 0 or metadata_remaining < 0:
                    raise HubUnavailable("Hub archive metadata exceeds size limit")
            # CPython documents this extension hook, but typeshed omits it.
            return super()._proc_member(tar)  # type: ignore[misc,no-any-return]

        def _proc_gnusparse_10(
            self,
            next: tarfile.TarInfo,
            pax_headers: dict[str, str],
            tar: tarfile.TarFile,
        ) -> None:
            # GNU sparse 1.0 maps live outside the bounded PAX metadata payload.
            raise HubUnavailable("Hub archive contains an unsafe member")

    try:
        if stage.is_symlink() or not stage.is_dir():
            raise HubUnavailable("Hub extraction stage is not a private directory")
        # gzip validates CRC/length, unlike tarfile's r|gz decompression wrapper.
        with gzip.open(archive, "rb") as compressed:
            with tarfile.open(
                fileobj=compressed, mode="r|", tarinfo=BoundedTarInfo, errors="strict"
            ) as tar:
                for member in tar:
                    name = member.name
                    path = PurePosixPath(name)
                    parts = path.parts
                    raw_parts = name.removesuffix("/").split("/")
                    if (
                        not parts
                        or path.is_absolute()
                        or ".." in parts
                        or "\\" in name
                        or ":" in name
                        or any(ord(c) < 32 or c in '<>"|?*' for c in name)
                        or any(ntpath.isabs(p) for p in parts)
                        or any(p in {"", "."} for p in raw_parts)
                        or any(p.rstrip(" .") != p for p in parts)
                        or any(
                            p.split(".")[0].rstrip(" ").upper() in _DEVICES
                            for p in parts
                        )
                        or member.mode & 0o7111
                        or not (member.isfile() or member.isdir())
                        or member.issparse()
                        or member.size < 0
                    ):
                        raise HubUnavailable("Hub archive contains an unsafe member")
                    if parts[0] != expected:
                        raise HubDatabaseMismatch(
                            "Hub archive database entry differs from declared backend"
                        )
                    if backend == "duckdb" and (len(parts) != 1 or not member.isfile()):
                        raise HubDatabaseMismatch(
                            "Hub archive backend shape differs from manifest"
                        )
                    if backend == "lancedb" and len(parts) == 1 and not member.isdir():
                        raise HubDatabaseMismatch(
                            "Hub archive backend shape differs from manifest"
                        )
                    portable_name = str(path).casefold()
                    if portable_name in seen:
                        raise HubUnavailable(
                            "Hub archive contains duplicate portable paths"
                        )
                    seen.add(portable_name)
                    for length in range(1, len(parts) + 1):
                        prefix = "/".join(parts[:length])
                        directory = length < len(parts) or member.isdir()
                        previous = paths.get(prefix.casefold())
                        if previous is not None and previous != (prefix, directory):
                            raise HubUnavailable(
                                "Hub archive contains conflicting portable paths"
                            )
                        paths[prefix.casefold()] = (prefix, directory)
                        if stage.joinpath(*parts[:length]).is_symlink():
                            raise HubUnavailable(
                                "Hub extraction path contains a symlink"
                            )
                    destination = stage.joinpath(*parts)
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        require_space(stage, member.size)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source = tar.extractfile(member)
                        if source is None:
                            raise HubUnavailable("Hub archive member cannot be read")
                        with source, destination.open("xb") as output:
                            remaining = member.size
                            while remaining:
                                block = source.read(min(CHUNK, remaining))
                                if not block:
                                    raise HubUnavailable(
                                        "Hub archive member is truncated"
                                    )
                                require_space(stage, len(block))
                                output.write(block)
                                remaining -= len(block)
                    # Streaming mode on Python <3.13 still caches TarInfo objects.
                    tar.members.clear()  # type: ignore[attr-defined]
                    metadata_remaining = CHUNK
            # Consume trailing padding and the gzip footer with bounded memory.
            while compressed.read(CHUNK):
                pass
        if not seen:
            raise HubDatabaseMismatch("Hub archive lacks its declared database entry")
        return stage / expected
    except (HubUnavailable, HubDatabaseMismatch):
        raise
    except (
        tarfile.TarError,
        OSError,
        ValueError,
        EOFError,
        zlib.error,
        RecursionError,
    ):
        raise HubUnavailable("Hub archive could not be unpacked") from None


def verify_candidate(path: Path, selected: Selected) -> None:
    """Check actual read-only inventory, not publisher metadata or sidecars."""
    key = selected.variant.key()
    try:
        inventory = inspect_database(path, key.db_provider)
        active = resolve_indexed_model(inventory.groups, None)
        if inventory.format_version != key.index_format_version:
            raise ValueError("format")
        if active is None or (active.provider, active.model, active.dims) != (
            key.provider,
            key.model,
            key.dims,
        ):
            raise ValueError("identity")
    except DatabaseReaderCloseError:
        # This is shutdown uncertainty, not evidence of a manifest contradiction.
        raise
    except Exception:
        raise HubDatabaseMismatch(
            "Downloaded database open, format, or embedding identity "
            "differs from manifest"
        ) from None


@dataclass
class PreparationState:
    """Set by the completed install worker, never inferred from candidate existence."""

    committed: bool = False


@asynccontextmanager
async def prepare(
    selected: Selected,
    work_parent: Path,
    *,
    state: PreparationState | None = None,
) -> AsyncIterator[Path]:
    """Yield one verified candidate, closing readers first and always removing work.

    Caller exceptions are not hub failures and retain their original category.
    No path is rewritten or installed by this context manager.
    """
    try:
        workspace = TemporaryDirectory(prefix=".chunkhound-pull-", dir=work_parent)
    except OSError:
        raise HubUnavailable("Hub staging workspace is unavailable") from None
    failed = False
    try:
        try:
            directory = Path(workspace.name)
            archive = await download(
                selected.version.artifact, directory / "artifact.part"
            )
            stage = directory / "stage"
            stage.mkdir()
            candidate = extract(archive, stage, selected.variant.db_provider)
            verify_candidate(candidate, selected)
        except OSError:
            raise HubUnavailable("Hub staging workspace is unavailable") from None
        yield candidate
    except BaseException:
        failed = True
        raise
    finally:
        try:
            workspace.cleanup()
        except OSError:
            if state is not None and state.committed:
                logger.bind(hub_client=True).warning(
                    "Hub base installed; staging cleanup could not be completed"
                )
            elif failed:
                # Preserve a caller failure/fatal mismatch/cancellation category.
                logger.bind(hub_client=True).error(
                    "Hub staging workspace could not be removed"
                )
            else:
                raise HubUnavailable(
                    "Hub staging workspace could not be removed"
                ) from None
