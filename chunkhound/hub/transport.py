"""Bounded, secret-safe HTTP transport for full opaque hub URLs."""

import asyncio
import hashlib
import logging
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx

from chunkhound.hub.errors import HubUnavailable
from chunkhound.hub.manifest import Artifact

MANIFEST_LIMIT = 4 * 1024 * 1024
CHUNK = 1024 * 1024
RESERVE = 16 * 1024 * 1024

# Suppress HTTP request URLs even when hosted as a library rather than by the CLI.
for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


def url_scheme_ok(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme == "https":
        return True
    if parts.scheme != "http" or parts.hostname is None:
        return False
    try:
        return ip_address(parts.hostname).is_loopback
    except ValueError:
        return False


def require_space(directory: Path, needed: int) -> None:
    if shutil.disk_usage(directory).free < needed + RESERVE:
        raise HubUnavailable("Insufficient disk space for hub candidate")


@asynccontextmanager
async def response(
    client: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> AsyncIterator[httpx.Response]:
    current = url
    for hop in range(6):
        try:
            parts = urlsplit(current)
            valid = (
                bool(parts.hostname)
                and url_scheme_ok(current)
                and parts.username is None
            )
            # Accessing port also rejects malformed or out-of-range ports.
            _ = parts.port
        except ValueError:
            valid = False
        if not valid:
            raise HubUnavailable(
                "Hub URL must use HTTPS or literal loopback HTTP without userinfo"
            )
        async with client.stream("GET", current, headers=headers) as result:
            if result.status_code not in {301, 302, 303, 307, 308}:
                yield result
                return
            location = result.headers.get("location")
            if hop == 5 or not location:
                raise HubUnavailable("Hub redirect chain is invalid")
            # Only HTTP redirects resolve relative references, never artifact URLs.
            current = urljoin(current, location)
    raise HubUnavailable("Hub redirect limit exceeded")


async def fetch_manifest(url: str) -> bytes:
    async def fetch() -> bytes:
        async with httpx.AsyncClient(
            timeout=10, follow_redirects=False, trust_env=False
        ) as client:
            async with response(client, url, {"Accept-Encoding": "identity"}) as result:
                if result.status_code != 200:
                    raise HubUnavailable("Hub manifest is unavailable")
                if result.headers.get("content-encoding", "identity") != "identity":
                    raise HubUnavailable(
                        "Hub manifest uses unsupported HTTP content encoding"
                    )
                body = bytearray()
                async for block in result.aiter_raw(CHUNK):
                    body.extend(block)
                    if len(body) > MANIFEST_LIMIT:
                        raise HubUnavailable("Hub manifest exceeds size limit")
                return bytes(body)

    try:
        return await asyncio.wait_for(fetch(), timeout=10)
    except HubUnavailable:
        raise
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        OSError,
        ValueError,
        asyncio.TimeoutError,
    ):
        raise HubUnavailable("Hub manifest request failed") from None


async def download(artifact: Artifact, part: Path) -> Path:
    """Verify a caller-owned part, retaining contiguous bytes across three attempts.

    The caller owns the temporary workspace and its cleanup. Resuming between
    process invocations is not promised; every returned file is size/hash checked.
    """

    async def transfer() -> Path:
        async with httpx.AsyncClient(
            timeout=60, follow_redirects=False, trust_env=False
        ) as client:
            for attempt in range(3):
                offset = part.stat().st_size if part.exists() else 0
                if offset > artifact.bytes:
                    part.unlink()
                    offset = 0
                if offset == artifact.bytes:
                    break
                # Three compressed sizes is headroom, not an extraction guarantee.
                require_space(part.parent, artifact.bytes - offset + 3 * artifact.bytes)
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                try:
                    async with response(client, artifact.url, headers) as result:
                        if result.status_code == 416:
                            part.unlink(missing_ok=True)
                            continue
                        if result.status_code not in {200, 206}:
                            raise HubUnavailable("Hub artifact request failed")
                        if (
                            result.headers.get("content-encoding", "identity")
                            != "identity"
                        ):
                            raise HubUnavailable(
                                "Hub artifact uses unsupported HTTP content encoding"
                            )
                        if result.status_code == 206:
                            match = re.fullmatch(
                                r"bytes (\d+)-(\d+)/(\d+)",
                                result.headers.get("content-range", ""),
                            )
                            if not match or tuple(map(int, match.groups())) != (
                                offset,
                                artifact.bytes - 1,
                                artifact.bytes,
                            ):
                                part.unlink(missing_ok=True)
                                raise HubUnavailable(
                                    "Hub returned an invalid byte range"
                                )
                            mode = "ab"
                        else:
                            # A server may ignore Range. Never splice a full response.
                            mode = "wb"
                        with part.open(mode) as output:
                            async for block in result.aiter_raw(CHUNK):
                                if output.tell() + len(block) > artifact.bytes:
                                    raise HubUnavailable(
                                        "Hub artifact size differs from manifest"
                                    )
                                require_space(part.parent, len(block))
                                output.write(block)
                        if part.stat().st_size == artifact.bytes:
                            break
                except httpx.HTTPError:
                    if attempt == 2:
                        raise HubUnavailable(
                            "Hub artifact download interrupted"
                        ) from None
            if not part.exists() or part.stat().st_size != artifact.bytes:
                raise HubUnavailable("Hub artifact size differs from manifest")
            digest = hashlib.sha256()
            with part.open("rb") as downloaded:
                while block := downloaded.read(CHUNK):
                    digest.update(block)
            if digest.hexdigest() != artifact.sha256.lower():
                raise HubUnavailable("Hub artifact checksum differs from manifest")
            return part

    try:
        return await asyncio.wait_for(transfer(), timeout=3600)
    except HubUnavailable:
        raise
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        OSError,
        ValueError,
        asyncio.TimeoutError,
    ):
        raise HubUnavailable("Hub artifact download failed") from None
