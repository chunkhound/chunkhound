"""Bounded HTTP(S) transport with sanitized hub-owned failures."""

import asyncio
import hashlib
import logging
import re
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from chunkhound.hub.errors import HubUnavailable
from chunkhound.hub.manifest import Artifact
from chunkhound.utils.http_redirects import RedirectChain, RedirectError

MANIFEST_LIMIT = 4 * 1024 * 1024
CHUNK = 1024 * 1024
RESERVE = 16 * 1024 * 1024

# Suppress default HTTP diagnostics in library use. Explicitly DEBUG-enabled
# child loggers can still expose response headers, including redirect URLs.
for name in ("httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


def url_scheme_ok(url: str) -> bool:
    try:
        parts = urlsplit(url)
        return parts.scheme in {"http", "https"} and parts.hostname is not None
    except ValueError:
        return False


def require_space(directory: Path, needed: int) -> None:
    if shutil.disk_usage(directory).free < needed + RESERVE:
        raise HubUnavailable("Insufficient disk space for hub candidate")


def _validate_hub_destination(url: str, previous: str | None) -> None:
    try:
        parts = urlsplit(url)
        valid = (
            parts.scheme in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
        )
        _ = parts.port
    except ValueError:
        valid = False
    if not valid:
        raise RedirectError("Hub URL must use HTTP or HTTPS without userinfo")
    if previous is not None:
        if urlsplit(previous).scheme == "https" and parts.scheme == "http":
            raise RedirectError("Hub redirect cannot downgrade HTTPS to HTTP")


@asynccontextmanager
async def response(
    client: httpx.AsyncClient, url: str, headers: dict[str, str]
) -> AsyncIterator[httpx.Response]:
    try:
        chain = RedirectChain(
            url, headers, validate=_validate_hub_destination, max_redirects=5
        )
        while True:
            async with client.stream(
                "GET", chain.current_url, headers=chain.headers
            ) as result:
                if not chain.advance(
                    result.status_code, result.headers.get("location")
                ):
                    yield result
                    return
    except RedirectError as exc:
        raise HubUnavailable(str(exc)) from None


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


async def download(
    artifact: Artifact, part: Path, *, timeout_seconds: float = 3600.0
) -> Path:
    """Verify a caller-owned part, retaining contiguous bytes across three attempts.

    The caller owns the temporary workspace and its cleanup. Resuming between
    process invocations is not promised; every returned file is size/hash checked.
    One cooperative total budget covers all attempts and verification. Network
    inactivity limits are independent; synchronous disk work cannot be forcibly
    interrupted by this budget.
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
        return await asyncio.wait_for(transfer(), timeout=timeout_seconds)
    except HubUnavailable:
        raise
    except asyncio.TimeoutError:
        raise HubUnavailable(
            f"Hub artifact download exceeded {timeout_seconds:g} seconds; increase "
            "hub.download_timeout_seconds or "
            "CHUNKHOUND_HUB__DOWNLOAD_TIMEOUT_SECONDS"
        ) from None
    except (httpx.HTTPError, httpx.InvalidURL, OSError, ValueError):
        raise HubUnavailable("Hub artifact download failed") from None
