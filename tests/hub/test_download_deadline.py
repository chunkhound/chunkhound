"""Total download budget and public fallback/cleanup contracts."""

import asyncio
import hashlib
import json
import threading
import traceback

import httpx
import pytest
from loguru import logger

from chunkhound.hub.client import pull
from chunkhound.hub.errors import HubUnavailable
from chunkhound.hub.manifest import Artifact
from chunkhound.hub.transport import download
from tests.hub.activation_support import cli, config_for, publish, write_config
from tests.hub.artifacts import build_database


async def test_artifact_deadline_is_shared_across_retries(tmp_path, monkeypatch):
    body = b"payload"
    requests = []
    closed = []

    class DelayedStream(httpx.AsyncByteStream):
        def __init__(self, attempt):
            self.attempt = attempt

        async def __aiter__(self):
            await asyncio.sleep(0.3)
            if self.attempt == 1:
                raise httpx.ReadError("transport-secret")
            yield body

        async def aclose(self):
            closed.append(self.attempt)

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=DelayedStream(len(requests)))

    real_client = httpx.AsyncClient

    def make_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    artifact = Artifact(
        url="https://hub.test/object?secret=url-secret", bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )
    part = tmp_path / "artifact.part"
    with pytest.raises(HubUnavailable) as caught:
        await asyncio.wait_for(download(artifact, part, timeout_seconds=0.5), 2)
    diagnostic = "".join(traceback.format_exception(caught.value))
    assert "0.5" in str(caught.value)
    assert "hub.download_timeout_seconds" in str(caught.value)
    assert "CHUNKHOUND_HUB__DOWNLOAD_TIMEOUT_SECONDS" in str(caught.value)
    assert "url-secret" not in diagnostic
    assert "transport-secret" not in diagnostic
    assert len(requests) == 2
    assert closed == [1, 2]
    assert not part.exists() or part.stat().st_size < len(body)


@pytest.mark.parametrize("automatic", [False, True])
async def test_pull_timeout_uses_config_and_cleans_workspace(
    tmp_path, hub_http, automatic
):
    config = config_for(tmp_path / "checkout", hub_http)
    config.hub.download_timeout_seconds = 0.05
    _, route = publish(hub_http, build_database(tmp_path / "published"))
    release = threading.Event()

    def stall_artifact(path):
        if path == route:
            release.wait(2)

    hub_http.before_get = stall_artifact
    messages = []
    sink = logger.add(
        lambda message: messages.append(str(message)),
        filter=lambda record: record["extra"].get("hub_client", False),
    )
    try:
        assert not await asyncio.wait_for(pull(config, automatic=automatic), 1)
    finally:
        release.set()
        logger.remove(sink)
    assert not config.database.get_db_path().exists()
    assert list(config.database.path.glob(".chunkhound-pull-*")) == []
    text = "\n".join(messages)
    assert "0.05" in text and "hub.download_timeout_seconds" in text
    assert ("continuing local indexing" in text) is automatic
    assert "token=secret" not in text


async def test_cancelled_download_cleans_workspace_without_install(tmp_path, hub_http):
    config = config_for(tmp_path / "checkout", hub_http)
    _, route = publish(hub_http, build_database(tmp_path / "published"))
    started = threading.Event()
    release = threading.Event()

    def stall_artifact(path):
        if path == route:
            started.set()
            release.wait(2)

    hub_http.before_get = stall_artifact
    task = asyncio.create_task(pull(config))
    try:
        assert await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not config.database.get_db_path().exists()
    assert list(config.database.path.glob(".chunkhound-pull-*")) == []


@pytest.mark.parametrize("command", ["pull", "index"])
def test_cli_download_timeout_reports_failure_or_indexes_locally(
    tmp_path, hub_http, embedding_http, command
):
    project = tmp_path / "checkout"
    data = write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    data["hub"]["download_timeout_seconds"] = 0.05
    (project / ".chunkhound.json").write_text(json.dumps(data))
    (project / "local.py").write_text("def deadline_local_fallback():\n    return 42\n")
    _, route = publish(hub_http, build_database(tmp_path / "published"))
    release = threading.Event()

    def stall_artifact(path):
        if path == route:
            release.wait(2)

    hub_http.before_get = stall_artifact
    try:
        result = cli(project, command, project)
    finally:
        release.set()
    assert result.returncode == (1 if command == "pull" else 0), (
        result.stdout + result.stderr
    )
    text = result.stdout + result.stderr
    assert "hub.download_timeout_seconds" in text
    assert "token=secret" not in text
    assert list((project / ".chunkhound" / "db").glob(".chunkhound-pull-*")) == []
    if command == "index":
        searched = cli(project, "search", "deadline_local_fallback", project, "--regex")
        assert searched.returncode == 0, searched.stdout + searched.stderr
        assert "local.py" in searched.stdout + searched.stderr
