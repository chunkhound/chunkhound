"""User-owned grants for trusting hub manifests declared by repositories."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

from chunkhound.utils.atomic_write import write_json_atomically

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_STORE_VERSION = 1
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_READ_ERROR = "Hub trust store cannot be read safely"
_UPDATE_ERROR = "Hub trust store cannot be updated safely"


class HubTrustError(RuntimeError):
    """The user-owned hub trust store could not be used safely."""


def trust_store_path() -> Path:
    """Return the fixed user-owned trust-store path."""
    return Path.home() / ".config" / "chunkhound" / "hub-trust.json"


def checkout_digest(checkout: Path) -> str:
    """Hash the canonical checkout identity without persisting its path."""
    resolved = os.path.normcase(str(checkout.expanduser().resolve()))
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def manifest_url_digest(manifest_url: str) -> str:
    """Hash an exact manifest URL so credentials never reach the store."""
    return hashlib.sha256(manifest_url.encode("utf-8")).hexdigest()


def _empty_store() -> dict[str, object]:
    return {"version": _STORE_VERSION, "grants": {}}


def _validate_store(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("trust store must be an object")
    version = value.get("version")
    if type(version) is not int or version != _STORE_VERSION:
        raise ValueError("unsupported trust store version")
    grants = value.get("grants")
    if not isinstance(grants, dict):
        raise ValueError("trust store grants must be an object")
    for checkout, digests in grants.items():
        if (
            not isinstance(checkout, str)
            or _DIGEST_PATTERN.fullmatch(checkout) is None
            or not isinstance(digests, list)
        ):
            raise ValueError("invalid trust store grant")
        if any(
            not isinstance(digest, str) or _DIGEST_PATTERN.fullmatch(digest) is None
            for digest in digests
        ):
            raise ValueError("invalid trust store digest")
    return cast(dict[str, object], value)


def _read_store() -> dict[str, object]:
    try:
        path = trust_store_path()
        with path.open(encoding="utf-8") as handle:
            return _validate_store(json.load(handle))
    except FileNotFoundError:
        return _empty_store()
    except HubTrustError:
        raise
    except Exception as exc:
        raise HubTrustError(_READ_ERROR) from exc


def _open_lock(path: Path) -> BinaryIO:
    flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "a+b")
    except BaseException:
        os.close(descriptor)
        raise
    return handle


def _acquire_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_LOCK,  # type: ignore[attr-defined]
            1,
        )
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(  # type: ignore[attr-defined]
            handle.fileno(),
            msvcrt.LK_UNLCK,  # type: ignore[attr-defined]
            1,
        )
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked_store() -> Generator[None, None, None]:
    handle: BinaryIO | None = None
    acquired = False
    try:
        path = trust_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = _open_lock(path.with_name("hub-trust.lock"))
        _acquire_lock(handle)
        acquired = True
        yield
    except HubTrustError:
        raise
    except Exception as exc:
        raise HubTrustError(_UPDATE_ERROR) from exc
    finally:
        if handle is not None:
            try:
                if acquired:
                    _release_lock(handle)
                handle.close()
            except Exception as exc:
                raise HubTrustError(_UPDATE_ERROR) from exc


def _write_store(store: dict[str, object]) -> None:
    try:
        path = trust_store_path()
        write_json_atomically(
            path,
            store,
            private=True,
            indent=2,
            sort_keys=True,
            trailing_newline=True,
        )
        if os.name != "nt":
            os.chmod(path, 0o600)
    except Exception as exc:
        raise HubTrustError(_UPDATE_ERROR) from exc


def is_trusted(checkout: Path, manifest_url: str) -> bool:
    """Return whether the exact URL is granted for the canonical checkout."""
    try:
        store = _read_store()
        grants = cast(dict[str, list[str]], store["grants"])
        return manifest_url_digest(manifest_url) in grants.get(
            checkout_digest(checkout), []
        )
    except HubTrustError:
        raise
    except Exception as exc:
        raise HubTrustError(_READ_ERROR) from exc


def grant_trust(checkout: Path, manifest_url: str) -> None:
    """Persist an exact manifest URL grant for a canonical checkout."""
    with _locked_store():
        store = _read_store()
        grants = cast(dict[str, list[str]], store["grants"])
        key = checkout_digest(checkout)
        values = set(grants.get(key, []))
        values.add(manifest_url_digest(manifest_url))
        grants[key] = sorted(values)
        _write_store(store)


def forget_checkout_trust(checkout: Path) -> bool:
    """Forget all grants for a checkout, returning whether one existed."""
    with _locked_store():
        store = _read_store()
        grants = cast(dict[str, list[str]], store["grants"])
        removed = grants.pop(checkout_digest(checkout), None) is not None
        if removed:
            _write_store(store)
        return removed
