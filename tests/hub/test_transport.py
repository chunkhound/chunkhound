"""Full-URL downloads tested against a real local HTTP(S) publisher."""

import errno
import hashlib
import logging
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.hub.errors import HubUnavailable
from chunkhound.hub.manifest import Artifact
from chunkhound.hub.transport import download, fetch_manifest

OBJECT = "/opaque%2Fobject?signature=url-secret&x=a%2Fb&x=2"
BODY = bytes(range(256)) * 12288


def artifact_at(hub_http, body=BODY, **overrides):
    hub_http.resources[OBJECT] = body
    fields = {
        "url": hub_http.url + OBJECT,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    fields.update(overrides)
    return Artifact(**fields)


def assert_secret_free(caught, caplog):
    diagnostic = "".join(traceback.format_exception(caught.value)) + caplog.text
    for secret in ["url-secret", "response-body-secret", "exception-secret", OBJECT]:
        assert secret not in diagnostic
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("ignore_range", [False, True])
async def test_resumes_or_restarts_without_splicing(hub_http, tmp_path, ignore_range):
    artifact = artifact_at(hub_http)
    hub_http.ignore_range = ignore_range
    part = tmp_path / "download.part"
    part.write_bytes(BODY[:137])
    assert await download(artifact, part) == part
    assert part.read_bytes() == BODY
    assert hub_http.requests == [(OBJECT, "bytes=137-")]
    assert hub_http.encodings == ["identity"]


@pytest.mark.parametrize("ignore_range", [False, True])
async def test_interrupted_download_retries_contiguous_bytes(
    hub_http, tmp_path, ignore_range
):
    artifact = artifact_at(hub_http)
    hub_http.ignore_range = ignore_range
    # More than one streaming block reaches disk before the wire is cut short.
    hub_http.interrupts = [1024 * 1024 + 137]
    part = tmp_path / "download.part"
    assert (await download(artifact, part)).read_bytes() == BODY
    assert len(hub_http.requests) == 2
    assert hub_http.requests[0] == (OBJECT, None)
    resumed_at = int(hub_http.requests[1][1][6:-1])
    assert 0 < resumed_at <= 1024 * 1024 + 137


async def test_interrupted_retries_stop_after_three_attempts(
    hub_http, tmp_path, caplog
):
    artifact = artifact_at(hub_http)
    hub_http.interrupts = [1024 * 1024, 1024 * 1024, 1]
    part = tmp_path / "download.part"
    with pytest.raises(HubUnavailable) as caught:
        await download(artifact, part)
    assert len(hub_http.requests) == 3
    assert part.read_bytes() == BODY[: part.stat().st_size]
    assert part.stat().st_size < len(BODY)
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("persistent", [False, True])
async def test_range_416_discards_partial_and_restarts_boundedly(
    hub_http, tmp_path, persistent, caplog
):
    artifact = artifact_at(hub_http, b"payload")
    part = tmp_path / "download.part"
    part.write_bytes(b"pay")
    hub_http.statuses = [416] * (3 if persistent else 1)
    if persistent:
        with pytest.raises(HubUnavailable) as caught:
            await download(artifact, part)
        assert_secret_free(caught, caplog)
        assert len(hub_http.requests) == 3
    else:
        assert (await download(artifact, part)).read_bytes() == b"payload"
        assert len(hub_http.requests) == 2
    assert hub_http.requests[0] == (OBJECT, "bytes=3-")
    assert all(request == (OBJECT, None) for request in hub_http.requests[1:])


async def test_exact_manifest_url_and_relative_redirect(hub_http, caplog):
    caplog.set_level(logging.DEBUG)
    body = b'{"manifest_version":2}'
    hub_http.resources[OBJECT] = body
    assert await fetch_manifest(hub_http.url + OBJECT) == body
    hub_http.redirects["/redirect"] = OBJECT
    assert await fetch_manifest(hub_http.url + "/redirect") == body
    assert hub_http.requests == [(OBJECT, None), ("/redirect", None), (OBJECT, None)]
    assert hub_http.encodings == ["identity"] * 3
    assert "url-secret" not in caplog.text
    assert OBJECT not in caplog.text


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("redirect", [False, True])
@pytest.mark.parametrize("host", ["localhost", "hub.internal", "192.0.2.1", "127.1"])
async def test_http_urls_allow_localhost_and_non_loopback_hosts(
    hub_http, hub_network, tmp_path, operation, redirect, host
):
    artifact = artifact_at(hub_http, b"payload")
    url = artifact.url.replace("127.0.0.1", host)
    if redirect:
        hub_http.redirects["/redirect"] = url
        url = hub_http.url + "/redirect"
    if operation == "manifest":
        assert await fetch_manifest(url) == b"payload"
    else:
        artifact = artifact_at(hub_http, b"payload", url=url)
        assert (
            await download(artifact, tmp_path / "download.part")
        ).read_bytes() == b"payload"
    assert hub_http.requests == ([("/redirect", None)] if redirect else []) + [
        (OBJECT, None)
    ]
    port = hub_http.url.rsplit(":", 1)[1]
    assert hub_http.hosts == ([f"127.0.0.1:{port}"] if redirect else []) + [
        f"{host}:{port}"
    ]


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
async def test_https_accepts_a_valid_trusted_certificate(
    hub_https, hub_trusted_certificate, tmp_path, operation
):
    artifact = artifact_at(hub_https, b"payload")
    if operation == "manifest":
        assert await fetch_manifest(artifact.url) == b"payload"
    else:
        assert (
            await download(artifact, tmp_path / "download.part")
        ).read_bytes() == b"payload"
    assert hub_https.requests == [(OBJECT, None)]


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("fault", ["untrusted", "hostname"])
async def test_https_rejects_invalid_certificates_without_sending_http(
    hub_https, hub_network, tmp_path, caplog, request, operation, fault
):
    caplog.set_level(logging.DEBUG)
    url = hub_https.url + OBJECT
    if fault == "hostname":
        request.getfixturevalue("hub_trusted_certificate")
        url = url.replace("127.0.0.1", "hub.internal")
    with pytest.raises(HubUnavailable) as caught:
        if operation == "manifest":
            await fetch_manifest(url)
        else:
            await download(artifact_at(hub_https, url=url), tmp_path / "download.part")
    assert hub_https.requests == []
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("transition", ["http-http", "http-https", "https-https"])
async def test_http_and_https_redirects_allow_non_downgrade_transitions(
    hub_http, hub_https, hub_trusted_certificate, tmp_path, operation, transition
):
    source_scheme, target_scheme = transition.split("-")
    servers = {"http": hub_http, "https": hub_https}
    source, target = servers[source_scheme], servers[target_scheme]
    artifact_at(target, b"payload")
    source.redirects["/redirect"] = target.url + OBJECT
    url = source.url + "/redirect"
    if operation == "manifest":
        assert await fetch_manifest(url) == b"payload"
    else:
        artifact = artifact_at(target, b"payload", url=url)
        assert (
            await download(artifact, tmp_path / "download.part")
        ).read_bytes() == b"payload"
    if source is target:
        assert source.requests == [("/redirect", None), (OBJECT, None)]
    else:
        assert source.requests == [("/redirect", None)]
        assert target.requests == [(OBJECT, None)]


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("start_http", [False, True])
async def test_https_redirect_downgrades_are_rejected_before_the_http_request(
    hub_http,
    hub_https,
    hub_trusted_certificate,
    tmp_path,
    caplog,
    operation,
    status,
    start_http,
):
    caplog.set_level(logging.DEBUG)
    artifact_at(hub_http, b"payload")
    hub_https.redirect_status = status
    hub_https.redirects["/secure"] = hub_http.url + OBJECT
    url = hub_https.url + "/secure"
    if start_http:
        hub_http.redirects["/upgrade"] = url
        url = hub_http.url + "/upgrade"
    with pytest.raises(HubUnavailable) as caught:
        if operation == "manifest":
            await fetch_manifest(url)
        else:
            await download(
                artifact_at(hub_http, b"payload", url=url), tmp_path / "download.part"
            )
    assert hub_https.requests == [("/secure", None)]
    assert hub_http.requests == ([("/upgrade", None)] if start_http else [])
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("redirect", [False, True])
@pytest.mark.parametrize(
    "unsafe",
    [
        "http://user:url-secret@127.0.0.1:{port}/",
        "https://user:url-secret@example.invalid/",
        "ftp://127.0.0.1/?token=url-secret",
        "http://127.0.0.1:bad/?token=url-secret",
        "http://[broken/?token=url-secret",
    ],
)
async def test_initial_and_redirect_urls_are_validated_without_secret_diagnostics(
    hub_http, tmp_path, caplog, operation, redirect, unsafe
):
    caplog.set_level(logging.DEBUG)
    url = unsafe.format(port=hub_http.url.rsplit(":", 1)[1])
    if redirect:
        hub_http.redirects["/redirect"] = url
        url = hub_http.url + "/redirect"
    with pytest.raises(HubUnavailable) as caught:
        if operation == "manifest":
            await fetch_manifest(url)
        else:
            await download(artifact_at(hub_http, url=url), tmp_path / "download.part")
    if redirect:
        # httpx may reject Location while constructing next_request even with
        # follow_redirects=False. That protocol error may use the retry budget,
        # but an unsafe destination must never receive a request.
        assert 1 <= len(hub_http.requests) <= (3 if operation == "artifact" else 1)
        assert all(request == ("/redirect", None) for request in hub_http.requests)
    else:
        assert hub_http.requests == []
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("location", [None, "/redirect"])
async def test_missing_or_looping_redirect_is_bounded(
    hub_http, tmp_path, caplog, operation, location
):
    hub_http.redirects["/redirect"] = location
    url = hub_http.url + "/redirect"
    with pytest.raises(HubUnavailable) as caught:
        if operation == "manifest":
            await fetch_manifest(url)
        else:
            await download(artifact_at(hub_http, url=url), tmp_path / "download.part")
    assert len(hub_http.requests) == (1 if location is None else 6)
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("fault", ["checksum", "range", "short", "long", "status"])
async def test_bad_artifact_response_is_soft_and_secret_free(
    hub_http, tmp_path, caplog, fault
):
    caplog.set_level(logging.DEBUG)
    body = b"payload" * 100
    overrides = {}
    if fault in {"short", "long"}:
        overrides["bytes"] = len(body) + (1 if fault == "short" else -1)
    if fault == "checksum":
        overrides["sha256"] = "0" * 64
    artifact = artifact_at(hub_http, body, **overrides)
    part = tmp_path / "download.part"
    if fault == "range":
        part.write_bytes(body[:5])
        hub_http.bad_range = True
    if fault == "status":
        hub_http.status = 503
    with pytest.raises(HubUnavailable) as caught:
        await download(artifact, part)
    assert len(hub_http.requests) <= 3
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize(
    "content_range", ["", "bytes 3-5/*", "bytes 3-4/7", "bytes 3-6/8"]
)
async def test_range_requires_exact_start_end_and_total(
    hub_http, tmp_path, content_range
):
    artifact = artifact_at(hub_http, b"payload")
    part = tmp_path / "download.part"
    part.write_bytes(b"pay")
    hub_http.content_range = content_range
    with pytest.raises(HubUnavailable, match="invalid byte range"):
        await download(artifact, part)
    assert hub_http.requests == [(OBJECT, "bytes=3-")]


@pytest.mark.parametrize("operation", ["manifest", "artifact"])
@pytest.mark.parametrize("fault", ["status", "encoding", "interrupted"])
async def test_http_failures_never_expose_response_body_or_url(
    hub_http, tmp_path, caplog, operation, fault
):
    caplog.set_level(logging.DEBUG)
    artifact = artifact_at(hub_http, b"payload")
    if fault == "status":
        hub_http.status = 503
    elif fault == "encoding":
        hub_http.encoding = "gzip"
    else:
        hub_http.interrupts = [1, 1, 1]
    with pytest.raises(HubUnavailable) as caught:
        if operation == "manifest":
            await fetch_manifest(artifact.url)
        else:
            await download(artifact, tmp_path / "download.part")
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("extra", [0, 1])
async def test_manifest_stream_enforces_four_mib_limit_without_content_length(
    hub_http, extra
):
    body = b"x" * (4 * 1024 * 1024 + extra)
    hub_http.resources[OBJECT] = body
    hub_http.omit_length = True
    if extra:
        with pytest.raises(HubUnavailable, match="size limit"):
            await fetch_manifest(hub_http.url + OBJECT)
    else:
        assert await fetch_manifest(hub_http.url + OBJECT) == body


@pytest.mark.parametrize("fault", ["headroom", "live-space", "enospc"])
async def test_disk_failures_are_soft_without_secret_exception_text(
    hub_http, tmp_path, monkeypatch, caplog, fault
):
    artifact = artifact_at(hub_http)
    part = tmp_path / "download.part"
    reserve = 16 * 1024 * 1024

    def disk_usage(directory):
        if fault == "headroom":
            return SimpleNamespace(free=reserve + 4 * len(BODY) - 1)
        # Permit the first write, then exhaust space: checking only at request
        # start (or before the first write) must not be sufficient.
        already_written = part.exists() and part.stat().st_size > 0
        return SimpleNamespace(free=0 if already_written else 2**40)

    if fault != "enospc":
        monkeypatch.setattr(shutil, "disk_usage", disk_usage)
    else:
        original_open = Path.open

        def open_file(path, mode="r", *args, **kwargs):
            if path == part and mode in {"ab", "wb"}:
                raise OSError(errno.ENOSPC, "exception-secret")
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(HubUnavailable) as caught:
        await download(artifact, part)
    assert len(hub_http.requests) == (0 if fault == "headroom" else 1)
    if fault == "live-space":
        assert 0 < part.stat().st_size < len(BODY)
        assert part.read_bytes() == BODY[: part.stat().st_size]
    else:
        assert not part.exists() or part.stat().st_size == 0
    assert_secret_free(caught, caplog)


@pytest.mark.parametrize("existing", ["oversized", "complete", "corrupt"])
async def test_existing_part_is_never_returned_without_size_and_hash_verification(
    hub_http, tmp_path, existing
):
    artifact = artifact_at(
        hub_http, b"payload", sha256=hashlib.sha256(b"payload").hexdigest().upper()
    )
    part = tmp_path / "download.part"
    part.write_bytes(
        {
            "oversized": b"invalid-too-long",
            "complete": b"payload",
            "corrupt": b"invalid",
        }[existing]
    )
    if existing == "corrupt":
        with pytest.raises(HubUnavailable, match="checksum"):
            await download(artifact, part)
    else:
        assert (await download(artifact, part)).read_bytes() == b"payload"
    assert hub_http.requests == ([(OBJECT, None)] if existing == "oversized" else [])
