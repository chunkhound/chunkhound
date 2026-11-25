import argparse
import asyncio
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger

import chunkhound.hub.client as hub_client
import chunkhound_native as native
from chunkhound.core.config.hub_config import HubConfig
from chunkhound.hub.client import ValidatedHubSource
from chunkhound.hub.errors import (
    HUB_PULL_FAILURE_GUIDANCE,
    HubLocalStateError,
    HubUnavailable,
)
from chunkhound.hub.inspect import inspect_database
from chunkhound.hub.manifest import parse_manifest
from chunkhound.hub.trust import HubTrustError, checkout_digest, manifest_url_digest
from tests.hub.activation_support import (
    ROOT,
    cli,
    cli_env,
    cli_is_trusted,
    config_for,
    publish,
    write_config,
)
from tests.hub.artifacts import build_database
from tests.hub.conftest import hub_server
from tests.hub.test_index_format import snapshot


def valid_source(checkout, hub_url):
    artifact = {"url": "http://127.0.0.1:1/base.tar.gz", "bytes": 1, "sha256": "0" * 64}
    variant = {
        "embedding_id": "test-base",
        "db_provider": "duckdb",
        "provider": "openai",
        "model": "hub-test",
        "dims": 8,
        "index_format_version": 1,
        "versions": [{"rel": r, "artifact": artifact} for r in ("latest", "-1", "-2")],
    }
    manifest = parse_manifest(
        json.dumps(
            {
                "manifest_version": 2,
                "project_id": "fixture-project",
                "embeddings": [variant],
            }
        ).encode()
    )
    return ValidatedHubSource(
        manifest=manifest,
        checkout_digest=checkout_digest(checkout),
        url_digest=manifest_url_digest(hub_url),
    )


def grant_cli_trust(project, checkout, hub_url):
    path = project.parent / "home" / ".config" / "chunkhound" / "hub-trust.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "grants": {
                    checkout_digest(checkout): [manifest_url_digest(hub_url)],
                },
            }
        ),
        encoding="utf-8",
    )


async def test_untrusted_plain_pull_rejects_before_hub_boundaries(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    config.hub.url = "https://user:secret@hub.invalid/manifest"
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    fetch = AsyncMock()
    artifact_pull = AsyncMock(return_value=False)
    monkeypatch.setattr(
        pull_module, "is_trusted", Mock(return_value=False), raising=False
    )
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    fetch.assert_not_awaited()
    artifact_pull.assert_not_awaited()
    text = "\n".join(messages)
    assert "pull --trust" in text
    assert "user:secret" not in text
    assert HUB_PULL_FAILURE_GUIDANCE not in text


async def test_plain_pull_trust_read_failure_is_sanitized_and_offline(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    fetch = AsyncMock()
    artifact_pull = AsyncMock(return_value=False)
    monkeypatch.setattr(
        pull_module,
        "is_trusted",
        Mock(side_effect=HubTrustError("private trust path")),
        raising=False,
    )
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    fetch.assert_not_awaited()
    artifact_pull.assert_not_awaited()
    assert messages == ["Hub source authorization could not be verified"]
    assert "private trust path" not in "\n".join(messages)


async def test_trusted_plain_pull_uses_explicit_replacement_path(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    trust_check = Mock(return_value=True)
    artifact_pull = AsyncMock(return_value=True)
    source = valid_source(checkout, config.hub.url)
    fetch = AsyncMock(return_value=source)
    monkeypatch.setattr(pull_module, "is_trusted", trust_check, raising=False)
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="-1")

    await pull_module.pull_command(args, config)

    trust_check.assert_called_once_with(checkout.resolve(), config.hub.url)
    fetch.assert_awaited_once_with(config)
    artifact_pull.assert_awaited_once_with(config, rel="-1", source=source)


@pytest.mark.parametrize(
    "outcome",
    [
        HubUnavailable("sanitized unavailable"),
        HubLocalStateError("unsafe local state"),
        False,
    ],
)
async def test_attempted_explicit_pull_failure_stops_with_local_retry_guidance(
    tmp_path, hub_http, monkeypatch, outcome
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    artifact_pull = (
        AsyncMock(return_value=outcome)
        if outcome is False
        else AsyncMock(side_effect=outcome)
    )
    monkeypatch.setattr(
        pull_module,
        "fetch_validated_manifest",
        AsyncMock(return_value=valid_source(checkout, config.hub.url)),
    )
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert HUB_PULL_FAILURE_GUIDANCE in messages
    assert config.hub.url not in "\n".join(messages)


async def test_attempted_explicit_pull_cancellation_exits_with_exact_guidance(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    monkeypatch.setattr(
        pull_module,
        "fetch_validated_manifest",
        AsyncMock(return_value=valid_source(checkout, config.hub.url)),
    )
    monkeypatch.setattr(
        pull_module, "pull", AsyncMock(side_effect=asyncio.CancelledError())
    )
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert messages == [
        "The attempted hub pull was cancelled before completion.",
        HUB_PULL_FAILURE_GUIDANCE,
    ]
    assert config.hub.url not in "\n".join(messages)


async def test_trusted_plain_pull_fetch_local_state_error_is_logged_once(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    monkeypatch.setattr(
        pull_module,
        "fetch_validated_manifest",
        AsyncMock(side_effect=HubLocalStateError("unsafe local state")),
    )
    artifact_pull = AsyncMock(return_value=True)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert messages.count("unsafe local state") == 1
    assert HUB_PULL_FAILURE_GUIDANCE in messages
    artifact_pull.assert_not_awaited()


async def test_trusted_plain_pull_manifest_fetch_cancellation_exits_with_exact_guidance(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    fetch = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch)
    artifact_pull = AsyncMock(return_value=True)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    fetch.assert_awaited_once_with(config)
    artifact_pull.assert_not_awaited()
    assert messages == [
        "The attempted hub pull was cancelled before completion.",
        HUB_PULL_FAILURE_GUIDANCE,
    ]


async def test_trusted_plain_pull_fetch_unavailable_emits_retry_guidance(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    monkeypatch.setattr(
        pull_module,
        "fetch_validated_manifest",
        AsyncMock(side_effect=HubUnavailable("sanitized unavailable")),
    )
    monkeypatch.setattr(pull_module, "pull", AsyncMock(return_value=True))
    args = argparse.Namespace(trust=False, path=checkout, version="latest")
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]))
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert "Hub pull unavailable: sanitized unavailable" in messages
    assert HUB_PULL_FAILURE_GUIDANCE in messages


async def test_pull_trust_revalidates_even_when_source_is_already_trusted(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    config = config_for(checkout, hub_http)
    source = valid_source(checkout, config.hub.url)
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    trust_check = Mock(return_value=True)
    fetch = AsyncMock(return_value=source)
    grant = Mock()
    artifact_pull = AsyncMock(return_value=True)
    monkeypatch.setattr(pull_module, "is_trusted", trust_check, raising=False)
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch)
    monkeypatch.setattr(pull_module, "grant_trust", grant)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=True, path=checkout, version="latest")

    await pull_module.pull_command(args, config)

    trust_check.assert_not_called()
    fetch.assert_awaited_once_with(config)
    grant.assert_called_once_with(checkout.resolve(), config.hub.url)
    artifact_pull.assert_awaited_once_with(config, rel="latest", source=source)


def test_pull_consent_shows_only_origin_and_selected_embedding(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    publish(hub_http, build_database(tmp_path / "published"))
    result = cli(project, "pull", project, "--trust")
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"Hub source origin={hub_http.url} " in output
    assert 'embedding provider="openai" model="hub-test" dimensions=8' in output
    assert "/catalogue" not in output and "token=secret" not in output
    assert hub_http.requests.count(("/catalogue?token=secret", None)) == 1


def test_pull_rejects_userinfo_before_any_request(tmp_path, hub_http, embedding_http):
    project = tmp_path / "checkout"
    host = hub_http.url.removeprefix("http://")
    write_config(
        project,
        "duckdb",
        f"http://user:pw@{host}/catalogue?token=secret",
        embedding_http.url,
    )
    result = cli(project, "pull", project, "--trust")
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert hub_http.requests == []
    assert "user:pw" not in output and "token=secret" not in output


async def test_consent_origin_brackets_ipv6_literal(tmp_path, hub_http, monkeypatch):
    checkout = tmp_path / "checkout"
    url = "http://[::1]:8080/catalogue?token=secret"
    config = config_for(checkout, hub_http).model_copy(
        update={"hub": HubConfig(url=url)}
    )
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    monkeypatch.setattr(pull_module, "is_trusted", Mock(return_value=True))
    monkeypatch.setattr(
        pull_module,
        "fetch_validated_manifest",
        AsyncMock(return_value=valid_source(checkout, url)),
    )
    monkeypatch.setattr(pull_module, "pull", AsyncMock(return_value=True))
    messages = []
    sink = logger.add(
        lambda m: messages.append(m.record["message"]),
        filter=lambda r: r["level"].name == "INFO",
    )
    try:
        await pull_module.pull_command(
            argparse.Namespace(trust=False, path=checkout, version="latest"), config
        )
    finally:
        logger.remove(sink)
    assert any("Hub source origin=http://[::1]:8080 " in m for m in messages), messages
    assert not any("token=secret" in m for m in messages)


def test_untrusted_plain_pull_makes_zero_requests_and_explains_trust(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    write_config(
        project,
        "duckdb",
        hub_http.url + "/catalogue?token=secret",
        embedding_http.url,
    )

    result = cli(project, "pull", project)

    assert result.returncode == 1
    assert hub_http.requests == []
    assert "pull --trust" in result.stderr
    assert "token=secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rel", ["latest", "-1", "-2"])
def test_explicit_relative_version_is_installed_and_searchable(
    tmp_path, hub_http, embedding_http, backend, rel
):
    project = tmp_path / "checkout"
    write_config(
        project, backend, hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    names = {"latest": "newest", "-1": "previous", "-2": "oldest"}
    versions = []
    for label, symbol in names.items():
        source = build_database(tmp_path / symbol, backend, symbol_prefix=symbol)
        manifest, _ = publish(hub_http, source, rel=label, prefix=symbol)
        versions.extend(manifest["embeddings"][0]["versions"])
    manifest["embeddings"][0]["versions"] = versions
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    prefix = names[rel]
    result = cli(project, "pull", project, "--version", rel)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Hub base installed; run index to overlay local changes" in (
        result.stdout + result.stderr
    )
    assert embedding_http.requests == []  # Explicit pull installs only.
    search = cli(project, "search", prefix + "_0", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "source.py" in search.stdout + search.stderr
    assert "No results found" not in search.stdout + search.stderr
    assert hub_http.requests == [
        ("/catalogue?token=secret", None),
        (f"/{prefix}.tar.gz?token=secret", None),
    ]
    assert "token=secret" not in result.stdout + result.stderr
    assert "/catalogue?" not in result.stdout + result.stderr
    assert cli_is_trusted(project, project, hub_http.url + "/catalogue?token=secret")


def test_trusted_plain_pull_explicitly_replaces_prior_database(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    publish(
        hub_http,
        build_database(tmp_path / "first", symbol_prefix="first"),
        prefix="first",
    )
    authorized = cli(project, "pull", project, "--trust")
    assert authorized.returncode == 0, authorized.stdout + authorized.stderr
    target = project / ".chunkhound" / "db" / "chunks.db"
    before = snapshot(target.parent)
    publish(
        hub_http,
        build_database(tmp_path / "replacement", symbol_prefix="replacement"),
        prefix="replacement",
    )
    hub_http.requests.clear()

    replaced = cli(project, "pull", project)

    assert replaced.returncode == 0, replaced.stdout + replaced.stderr
    assert snapshot(target.parent) != before
    assert hub_http.requests == [
        ("/catalogue?token=secret", None),
        ("/replacement.tar.gz?token=secret", None),
    ]
    search = cli(project, "search", "replacement_0", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "source.py" in search.stdout + search.stderr
    assert "token=secret" not in replaced.stdout + replaced.stderr


def test_pull_trust_is_saved_after_manifest_before_compatibility_selection(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    publish(
        hub_http,
        build_database(tmp_path / "published", model="wrong-model"),
        model="wrong-model",
    )

    result = cli(project, "pull", project, "--trust")

    assert result.returncode == 1
    assert cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [("/catalogue?token=secret", None)]


async def test_pull_trust_checkout_identity_failure_is_sanitized_once(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    raw = "raw checkout path?token=secret"
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")

    def fail_digest(_checkout):
        raise OSError(raw)

    monkeypatch.setattr(pull_module, "checkout_digest", fail_digest)
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", AsyncMock())
    args = argparse.Namespace(trust=True, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert messages == ["Hub checkout identity is unavailable"]
    assert raw not in "\n".join(messages)
    assert HUB_PULL_FAILURE_GUIDANCE not in messages
    pull_module.fetch_validated_manifest.assert_not_awaited()


async def test_pull_trust_manifest_binding_failure_is_sanitized_once(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    raw = "raw checkout path?token=secret"
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")

    async def fail_fetch(_config):
        try:
            raise OSError(raw)
        except OSError as cause:
            raise HubLocalStateError("Hub checkout identity is unavailable") from cause

    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fail_fetch)
    args = argparse.Namespace(trust=True, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert messages == ["Hub checkout identity is unavailable"]
    assert raw not in "\n".join(messages)
    assert HUB_PULL_FAILURE_GUIDANCE not in messages


async def test_pull_trust_post_fetch_identity_failure_is_sanitized_once(
    tmp_path, hub_http, monkeypatch
):
    checkout = tmp_path / "checkout"
    config = config_for(checkout, hub_http)
    captured_url = config.hub.url
    source = valid_source(checkout, captured_url)
    raw = "raw post-fetch checkout path?token=secret"
    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")

    def fail_revalidation(_current_checkout):
        raise OSError(raw)

    monkeypatch.setattr(
        pull_module, "fetch_validated_manifest", AsyncMock(return_value=source)
    )
    # The re-check after the fetch is shared client code; fail it there.
    monkeypatch.setattr(hub_client, "checkout_digest", fail_revalidation)
    monkeypatch.setattr(pull_module, "grant_trust", Mock())
    args = argparse.Namespace(trust=True, path=checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert messages == ["Hub checkout identity is unavailable"]
    assert raw not in "\n".join(messages)
    pull_module.grant_trust.assert_not_called()
    assert HUB_PULL_FAILURE_GUIDANCE not in messages


@pytest.mark.parametrize("change", ["checkout", "url"])
async def test_pull_trust_revalidates_captured_identity_before_grant_or_artifact(
    tmp_path, hub_http, monkeypatch, change
):
    first_checkout = tmp_path / "first-checkout"
    second_checkout = tmp_path / "second-checkout"
    first_checkout.mkdir()
    second_checkout.mkdir()
    config = config_for(first_checkout, hub_http)
    captured_url = config.hub.url
    source = valid_source(first_checkout, captured_url)

    async def fetch_and_mutate(_config):
        if change == "checkout":
            config.target_dir = second_checkout
        else:
            config.hub.url += "&changed=true"
        return source

    pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
    grant = Mock()
    artifact_pull = AsyncMock(return_value=True)
    monkeypatch.setattr(pull_module, "fetch_validated_manifest", fetch_and_mutate)
    monkeypatch.setattr(pull_module, "grant_trust", grant)
    monkeypatch.setattr(pull_module, "pull", artifact_pull)
    args = argparse.Namespace(trust=True, path=first_checkout, version="latest")
    messages = []
    sink = logger.add(
        lambda message: messages.append(message.record["message"]),
        filter=lambda record: record["level"].name == "ERROR",
    )
    try:
        with pytest.raises(SystemExit) as stopped:
            await pull_module.pull_command(args, config)
    finally:
        logger.remove(sink)

    assert stopped.value.code == 1
    assert grant.call_count == 0
    artifact_pull.assert_not_awaited()
    assert messages == [
        "Hub pull unavailable: Validated hub source no longer matches authorized "
        "configuration"
    ]
    assert captured_url not in "\n".join(messages)


@pytest.mark.parametrize(
    "artifact_url",
    [
        "index.tar.gz",
        "file:///tmp/index.tar.gz",
        "https://user:secret@cdn.invalid/index.tar.gz",
        "https://cdn.invalid:bad/index.tar.gz",
    ],
)
def test_pull_trust_unsupported_artifact_destination_saves_nothing(
    tmp_path, hub_http, embedding_http, artifact_url
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    manifest, _ = publish(hub_http, build_database(tmp_path / "published"))
    manifest["embeddings"][0]["versions"][0]["artifact"]["url"] = artifact_url
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()

    result = cli(project, "pull", project, "--trust")

    text = result.stdout + result.stderr
    assert result.returncode == 1
    assert "invalid or unsupported" in text
    assert not cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [("/catalogue?token=secret", None)]
    assert "Traceback" not in text


def test_pull_trust_invalid_manifest_saves_nothing(tmp_path, hub_http, embedding_http):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    hub_http.resources["/catalogue?token=secret"] = b"{}"

    result = cli(project, "pull", project, "--trust")

    assert result.returncode == 1
    assert not cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [("/catalogue?token=secret", None)]


def test_pull_trust_survives_artifact_failure(tmp_path, hub_http, embedding_http):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    _, route = publish(hub_http, build_database(tmp_path / "published"))
    hub_http.resources[route] = b"corrupt artifact"

    result = cli(project, "pull", project, "--trust")

    assert result.returncode == 1
    assert cli_is_trusted(project, project, hub_url)
    assert hub_http.requests[0] == ("/catalogue?token=secret", None)
    assert any(request[0] == route for request in hub_http.requests[1:])


def test_pull_trust_save_failure_warns_but_current_pull_proceeds(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published"))
    script = r"""
import importlib
import sys
sys.path.insert(0, sys.argv.pop(1))
from chunkhound.api.cli.main import main
from chunkhound.hub.trust import HubTrustError
pull_module = importlib.import_module("chunkhound.api.cli.commands.pull")
def fail_grant(checkout, manifest_url):
    raise HubTrustError("raw trust failure secret")
pull_module.grant_trust = fail_grant
sys.argv = ["chunkhound", *sys.argv[1:]]
main()
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), "pull", str(project), "--trust"],
        cwd=project,
        env=cli_env(project),
        capture_output=True,
        text=True,
        timeout=180,
    )

    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "future automatic hub use was not enabled" in text
    assert "raw trust failure secret" not in text
    assert not cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [
        ("/catalogue?token=secret", None),
        ("/base.tar.gz?token=secret", None),
    ]


def test_pull_trust_flags_are_mutually_exclusive(tmp_path):
    result = cli(tmp_path, "pull", "--trust", "--forget-trust")
    assert result.returncode == 2
    assert "not allowed with argument --trust" in result.stderr


def test_forget_trust_rejects_relative_version_without_config(tmp_path):
    result = cli(tmp_path, "pull", "--forget-trust", "--version", "-1")
    assert result.returncode == 1
    assert "--version cannot be used with --forget-trust" in result.stderr


def test_forget_trust_is_offline_idempotent_and_preserves_database(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published"))
    trusted = cli(project, "pull", project, "--trust")
    assert trusted.returncode == 0, trusted.stdout + trusted.stderr
    assert cli_is_trusted(project, project, hub_url)
    target = project / ".chunkhound" / "db" / "chunks.db"
    before = snapshot(target.parent)
    (project / ".chunkhound.json").write_text("not valid json", encoding="utf-8")
    sentinel = tmp_path / "remote-config-called"
    script = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
import chunkhound.api.cli.utils.config_factory as config_factory
from chunkhound.api.cli.main import main
sentinel = Path(sys.argv.pop(1))
async def fail_if_called(args, command):
    sentinel.write_text("called")
    raise AssertionError("remote config fetch must not run")
config_factory.run_remote_config_fetch = fail_if_called
sys.argv = ["chunkhound", *sys.argv[1:]]
main()
"""
    hub_http.requests.clear()

    first = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            str(sentinel),
            "pull",
            str(project),
            "--forget-trust",
        ],
        cwd=project,
        env=cli_env(project),
        capture_output=True,
        text=True,
        timeout=180,
    )
    second = cli(project, "pull", project, "--forget-trust")

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    assert not sentinel.exists()
    assert not cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == []
    assert snapshot(target.parent) == before


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("rust", ["0", "1"])
@pytest.mark.parametrize("initial", ["absent", "misaligned", "aligned"])
def test_trusted_cold_index_installs_but_existing_database_never_requests_hub(
    tmp_path, hub_http, embedding_http, backend, rust, initial
):
    project = tmp_path / "checkout"
    project.mkdir()
    hub_url = hub_http.url + "/catalogue?token=secret"
    (project / "source.py").write_text(
        "\n".join(f"def local_{i}(): return 'local_marker_{i}'" for i in range(100)),
        encoding="utf-8",
    )
    expected_model = "hub-test"
    if initial != "absent":
        expected_model = "hub-test" if initial == "aligned" else "old-model"
        write_config(project, backend, None, embedding_http.url, expected_model)
        old = cli(project, "index", project, rust=rust)
        assert old.returncode == 0, old.stdout + old.stderr
    write_config(project, backend, hub_url, embedding_http.url)
    publish(hub_http, build_database(tmp_path / "published", backend))
    grant_cli_trust(project, project, hub_url)
    embedding_http.requests.clear()
    result = cli(project, "index", project, "--verbose", rust=rust)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(hub_http.requests) == (2 if initial == "absent" else 0)
    target = (
        project
        / ".chunkhound"
        / "db"
        / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    )
    assert {g.model for g in inspect_database(target, backend).groups} == {
        expected_model
    }
    assert all(
        request["model"] == expected_model for request in embedding_http.requests
    )
    if initial == "absent":
        assert embedding_http.requests  # Normal overlay followed base install.
    assert "token=secret" not in result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["duckdb", "lancedb"])
@pytest.mark.parametrize("initial", ["misaligned", "aligned"])
def test_index_trust_existing_database_saves_source_without_artifact_request(
    tmp_path, hub_http, embedding_http, backend, initial
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    model = "hub-test" if initial == "aligned" else "old-model"
    write_config(project, backend, None, embedding_http.url, model)
    (project / "source.py").write_text("def retained_local(): return 1\n")
    baseline = cli(project, "index", project)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    target = (
        project
        / ".chunkhound"
        / "db"
        / ("chunks.db" if backend == "duckdb" else "lancedb.lancedb")
    )
    write_config(project, backend, hub_url, embedding_http.url)
    _, route = publish(hub_http, build_database(tmp_path / "published", backend))
    hub_http.requests.clear()
    preparation_snapshots = tmp_path / "preparation-snapshots.json"
    script = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
report = Path(sys.argv.pop(1))
target = Path(sys.argv.pop(1))
from chunkhound.api.cli.commands import run
from chunkhound.api.cli.main import main
from tests.hub.test_index_format import snapshot
original = run._prepare_hub_index
async def snapshot_preparation(*args, **kwargs):
    before = snapshot(target)
    decision = await original(*args, **kwargs)
    after = snapshot(target)
    report.write_text(json.dumps({"before": before, "after": after}))
    return decision
run._prepare_hub_index = snapshot_preparation
sys.argv = ["chunkhound", *sys.argv[1:]]
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            str(preparation_snapshots),
            str(target),
            "index",
            str(project),
            "--trust",
        ],
        cwd=project,
        env=cli_env(project),
        text=True,
        capture_output=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    snapshots = json.loads(preparation_snapshots.read_text())
    assert snapshots["after"] == snapshots["before"]
    assert cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [("/catalogue?token=secret", None)]
    assert all(request[0] != route for request in hub_http.requests)
    assert {g.model for g in inspect_database(target, backend).groups} == {model}


def test_index_trust_without_source_stops_before_local_index(tmp_path, embedding_http):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", None, embedding_http.url)
    (project / "source.py").write_text("def must_not_index(): return 1\n")

    result = cli(project, "index", project, "--trust")

    assert result.returncode == 1
    assert "Configure hub.url" in result.stdout + result.stderr
    assert embedding_http.requests == []
    assert not (project / ".chunkhound" / "db" / "chunks.db").exists()


def test_index_trust_persists_and_installs_cold_base(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    (project / "local.py").write_text("def local_overlay(): return 1\n")
    publish(hub_http, build_database(tmp_path / "published"))

    result = cli(project, "index", project, "--trust")

    assert result.returncode == 0, result.stdout + result.stderr
    assert cli_is_trusted(project, project, hub_url)
    assert hub_http.requests == [
        ("/catalogue?token=secret", None),
        ("/base.tar.gz?token=secret", None),
    ]
    assert embedding_http.requests
    search = cli(project, "search", "local_overlay", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "local.py" in search.stdout + search.stderr


@pytest.mark.parametrize("source", ["environment", "explicit", "project", "global"])
def test_trust_store_path_declaration_cannot_authorize_or_redirect(
    tmp_path, hub_http, embedding_http, source
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    alternate = tmp_path / "attacker" / "grants.json"
    malicious = {"url": hub_url, "trust_store_path": str(alternate)}
    data = write_config(project, "duckdb", None, embedding_http.url)
    data.pop("hub")
    (project / "source.py").write_text("def must_not_index(): return 1\n")
    env, args = cli_env(project), ["index", project]

    if source == "environment":
        env["CHUNKHOUND_HUB__URL"] = hub_url
        env["CHUNKHOUND_HUB__TRUST_STORE_PATH"] = str(alternate)
    elif source == "explicit":
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({"hub": malicious}))
        args.extend(("--config", explicit))
    elif source == "project":
        data["hub"] = malicious
    else:
        Path(env["CHUNKHOUND_GLOBAL_CONFIG_FILE"]).write_text(
            json.dumps({"hub": malicious})
        )
    (project / ".chunkhound.json").write_text(json.dumps(data))

    result = subprocess.run(
        [sys.executable, "-m", "chunkhound.api.cli.main", *map(str, args)],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    # tmp_path embeds the test name, so scrub it before checking diagnostics.
    output = (result.stdout + result.stderr).replace(str(tmp_path), "<tmp>")
    central_trust = Path(env["HOME"]) / ".config" / "chunkhound" / "hub-trust.json"
    assert hub_http.requests == []
    assert not alternate.exists()
    assert not central_trust.exists()
    if source == "environment":
        assert result.returncode == 0, output
        assert "trust_store_path" not in output
        assert embedding_http.requests
        assert (project / ".chunkhound" / "db" / "chunks.db").exists()
    else:
        assert result.returncode == 1, output
        assert "trust_store_path" in output
        assert embedding_http.requests == []
        assert not (project / ".chunkhound" / "db" / "chunks.db").exists()


def test_trusted_manifest_delegates_artifact_redirect_chain(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    source = build_database(tmp_path / "published")
    manifest, origin_route = publish(hub_http, source)
    archive = hub_http.resources.pop(origin_route)

    with hub_server() as artifact_http:
        delegated_route = "/delegated.tar.gz?artifact-token=secret"
        final_route = "/objects/base.tar.gz?artifact-token=secret"
        artifact_http.redirects[delegated_route] = artifact_http.url + final_route
        artifact_http.resources[final_route] = archive
        artifact = manifest["embeddings"][0]["versions"][0]["artifact"]
        artifact["url"] = artifact_http.url + delegated_route
        hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()

        result = cli(project, "index", project, "--trust")

    assert result.returncode == 0, result.stdout + result.stderr
    assert hub_http.requests == [("/catalogue?token=secret", None)]
    assert artifact_http.requests == [
        (delegated_route, None),
        (final_route, None),
    ]
    assert len(archive) == artifact["bytes"]
    assert hashlib.sha256(archive).hexdigest() == artifact["sha256"]
    assert cli_is_trusted(project, project, hub_url)
    assert {
        group.model
        for group in inspect_database(
            project / ".chunkhound" / "db" / "chunks.db", "duckdb"
        ).groups
    } == {"hub-test"}
    assert "token=secret" not in result.stdout + result.stderr


def test_untrusted_no_prompt_index_builds_searchable_local_database_offline(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    (project / "source.py").write_text("def local_only(): return 1\n")
    publish(hub_http, build_database(tmp_path / "published"))

    result = cli(project, "index", project)

    assert result.returncode == 0, result.stdout + result.stderr
    assert hub_http.requests == []
    assert embedding_http.requests
    assert not cli_is_trusted(project, project, hub_url)
    search = cli(project, "search", "local_only", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "source.py" in search.stdout + search.stderr
    output = result.stdout + result.stderr
    assert output.count("Configured hub source is not trusted") == 1
    assert "token=secret" not in output


def test_untrusted_trust_read_failure_warns_once_and_indexes_offline(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "source.py").write_text("def local_only(): return 1\n")
    store = project.parent / "home" / ".config" / "chunkhound" / "hub-trust.json"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text("{not json", encoding="utf-8")  # unreadable trust store
    result = cli(project, "index", project)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert hub_http.requests == []
    assert output.count("Hub trust settings could not be read") == 1
    assert output.count("Configured hub source is not trusted") == 0


def test_interactive_local_choice_allows_full_index_without_hub_request(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "source.py").write_text("def chosen_local(): return 1\n")
    prompt_report = tmp_path / "prompt-report.json"
    script = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
report = Path(sys.argv.pop(1))
from chunkhound.api.cli.commands import autodoc_prompts
from chunkhound.api.cli.main import main
autodoc_prompts.is_interactive = lambda: True
def prompt(question, *, default=None):
    report.write_text(json.dumps({"question": question, "default": default}))
    return "local"
autodoc_prompts.prompt_text = prompt
sys.argv = ["chunkhound", *sys.argv[1:]]
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            str(prompt_report),
            "index",
            str(project),
        ],
        cwd=project,
        env={**cli_env(project), "CHUNKHOUND_NO_PROMPTS": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(prompt_report.read_text()) == {
        "question": "Continue with local indexing or cancel? (local/cancel)",
        "default": "cancel",
    }
    assert "index --trust" in result.stdout + result.stderr
    assert hub_http.requests == []
    assert embedding_http.requests


def test_interactive_cancel_stops_without_hub_embedding_database_or_trust(
    tmp_path, hub_http, embedding_http
):
    project = tmp_path / "checkout"
    hub_url = hub_http.url + "/catalogue?token=secret"
    write_config(project, "duckdb", hub_url, embedding_http.url)
    (project / "source.py").write_text("def cancelled_local(): return 1\n")
    publish(hub_http, build_database(tmp_path / "published"))
    prompt_report = tmp_path / "prompt-report.json"
    script = r"""
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
report = Path(sys.argv.pop(1))
from chunkhound.api.cli.commands import autodoc_prompts
from chunkhound.api.cli.main import main
autodoc_prompts.is_interactive = lambda: True
def prompt(question, *, default=None):
    report.write_text(json.dumps({"question": question, "default": default}))
    return "cancel"
autodoc_prompts.prompt_text = prompt
sys.argv = ["chunkhound", *sys.argv[1:]]
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            str(prompt_report),
            "index",
            str(project),
        ],
        cwd=project,
        env={**cli_env(project), "CHUNKHOUND_NO_PROMPTS": "0"},
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 1
    assert json.loads(prompt_report.read_text()) == {
        "question": "Continue with local indexing or cancel? (local/cancel)",
        "default": "cancel",
    }
    assert hub_http.requests == []
    assert embedding_http.requests == []
    assert not (project / ".chunkhound" / "db" / "chunks.db").exists()
    assert not cli_is_trusted(project, project, hub_url)


@pytest.mark.parametrize("mode", ["--no-embeddings", "--simulate", "--check-ignores"])
def test_index_subtools_do_not_request_hub(tmp_path, hub_http, embedding_http, mode):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "source.py").write_text("def local_only(): return 1\n")
    result = cli(project, "index", project, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert hub_http.requests == []
    assert embedding_http.requests == []


def test_trusted_plain_pull_unknown_dimensions_keeps_guidance_and_stays_offline(
    tmp_path, hub_http
):
    project = tmp_path / "checkout"
    data = write_config(
        project,
        "duckdb",
        hub_http.url + "/catalogue?token=secret",
        "http://127.0.0.1:1/v1",
        model="unknown-offline-model",
    )
    del data["embedding"]["output_dims"]
    (project / ".chunkhound.json").write_text(json.dumps(data))
    grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")

    result = cli(project, "pull", project)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert hub_http.requests == []
    assert "known embedding dimensions" in output
    assert HUB_PULL_FAILURE_GUIDANCE in output


def test_unknown_dimensions_do_not_probe_hub_or_embedding_service(tmp_path, hub_http):
    project = tmp_path / "checkout"
    data = write_config(
        project,
        "duckdb",
        hub_http.url + "/catalogue?token=secret",
        "http://127.0.0.1:1/v1",
        model="unknown-offline-model",
    )
    del data["embedding"]["output_dims"]
    (project / ".chunkhound.json").write_text(json.dumps(data))
    grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    result = cli(project, "pull", project)
    assert result.returncode == 1
    assert hub_http.requests == []
    assert "known embedding dimensions" in result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
@pytest.mark.parametrize(
    "failure", ["http", "checksum", "archive", "space", "mismatch"]
)
def test_cli_failure_taxonomy_is_real_not_a_mocked_registry_assertion(
    tmp_path, hub_http, embedding_http, mode, failure
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "local.py").write_text("def local_fallback(): return 7\n")
    source = build_database(
        tmp_path / "published",
        model="wrong-model" if failure == "mismatch" else "hub-test",
    )
    manifest, route = publish(hub_http, source)
    if failure == "http":
        hub_http.status = 503
    elif failure == "checksum":
        manifest["embeddings"][0]["versions"][0]["artifact"]["sha256"] = "0" * 64
    elif failure == "archive":
        body = b"not an archive"
        hub_http.resources[route] = body
        manifest["embeddings"][0]["versions"][0]["artifact"].update(
            bytes=len(body), sha256=hashlib.sha256(body).hexdigest()
        )
    hub_http.resources["/catalogue?token=secret"] = json.dumps(manifest).encode()
    if mode == "pull":
        grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    if failure == "space":
        script = """
import shutil
import sys
from pathlib import Path
from chunkhound.api.cli.main import main
original = shutil.disk_usage
def disk_usage(path):
    usage = original(path)
    return (
        usage._replace(free=0)
        if Path(path).name.startswith('.chunkhound-pull-') else usage
    )
shutil.disk_usage = disk_usage
sys.argv = ['chunkhound', *sys.argv[1:]]
main()
"""
        command_args = [mode, str(project), "--verbose"]
        if mode == "index":
            command_args.append("--trust")
        result = subprocess.run(
            [sys.executable, "-c", script, *command_args],
            cwd=project,
            env=cli_env(project),
            capture_output=True,
            text=True,
            timeout=180,
        )
    else:
        command_args = [mode, project, "--verbose"]
        if mode == "index":
            command_args.append("--trust")
        result = cli(project, *command_args)
    fatal = mode in {"pull", "index"}
    assert result.returncode == (1 if fatal else 0), result.stdout + result.stderr
    text = result.stdout + result.stderr
    assert "token=secret" not in text and "response-body-secret" not in text
    assert "Traceback" not in text
    target = project / ".chunkhound" / "db" / "chunks.db"
    assert target.exists() is (not fatal)
    if not fatal:
        search = cli(project, "search", "local_fallback", project, "--regex")
        assert "local.py" in search.stdout + search.stderr


@pytest.mark.parametrize("mode", ["pull"])
def test_candidate_mismatch_exit_preserves_local_database_sidecars(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(project, "duckdb", None, embedding_http.url, "old-model")
    (project / "local.py").write_text("def retained_local_work(): return 7\n")
    baseline = cli(project, "index", project)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    target = project / ".chunkhound" / "db" / "chunks.db"
    before = snapshot(target.parent)
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    publish(hub_http, build_database(tmp_path / "published", model="contradiction"))
    grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    result = cli(project, mode, project, "--verbose")
    assert result.returncode == 1
    assert snapshot(target.parent) == before
    assert "Traceback" not in result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["pull", "index"])
def test_preinstall_busy_has_fixed_fatal_diagnostic(
    tmp_path, hub_http, embedding_http, mode
):
    project = tmp_path / "checkout"
    write_config(
        project, "duckdb", hub_http.url + "/catalogue?token=secret", embedding_http.url
    )
    (project / "local.py").write_text("def local_busy_fallback(): return 7\n")
    target = project / ".chunkhound" / "db" / "chunks.db"
    target.parent.mkdir(parents=True)
    publish(hub_http, build_database(tmp_path / "published"))
    if mode == "pull":
        grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    # This owner excludes installation but owns no backend writer connection.
    held = native.acquire_database_lease(str(target))
    try:
        command_args = [mode, project, "--verbose"]
        if mode == "index":
            command_args.append("--trust")
        result = cli(project, *command_args)
    finally:
        held.close()
    assert result.returncode == 1
    assert (
        "Hub local installation state is unsafe; local indexing was not started"
        in result.stderr
    )
    assert "Traceback" not in result.stdout + result.stderr
    assert not target.exists()


def test_zero_is_not_a_relative_version(tmp_path):
    assert cli(tmp_path, "pull", "--version", "0").returncode == 2


@pytest.mark.parametrize(
    "boundary,mode",
    [("candidate", "pull")],
)
def test_cli_reader_failure_has_sanitized_exit_without_fallback(
    tmp_path, hub_http, embedding_http, boundary, mode
):
    project = tmp_path / "checkout"
    write_config(
        project,
        "duckdb",
        hub_http.url + "/catalogue?token=secret",
        embedding_http.url,
    )
    (project / "local.py").write_text("def must_not_index(): return 42\n")
    target = project / ".chunkhound" / "db" / "chunks.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    before = []
    if boundary != "recheck":
        build_database(target.parent, model="old-model")
        before.append(snapshot(target))
    competitor = build_database(tmp_path / "competitor", model="old-model")
    publish(hub_http, build_database(tmp_path / "published"))
    grant_cli_trust(project, project, hub_http.url + "/catalogue?token=secret")
    sidecar = target.with_name(target.name + ".root.json")
    retained_root = json.dumps({"version": 1, "indexed_root_path": project.as_posix()})
    sidecar.write_text(retained_root)
    if boundary == "recheck":

        def install_race_winner(route):
            if route.startswith("/base.tar.gz"):
                lease = native.acquire_install_lease(str(target), "duckdb")
                try:
                    lease.install(str(competitor))
                finally:
                    lease.close()
                sidecar.write_text(retained_root)
                before.append(snapshot(target))

        hub_http.before_get = install_race_winner
    report = tmp_path / "cli-reader-report.json"
    script = r"""
import json
import sys
from pathlib import Path
root, boundary, target_text, report_text, mode, project = sys.argv[1:]
sys.path.insert(0, root)
import pytest
import chunkhound_native as native
from chunkhound.api.cli.main import main
from chunkhound.hub.errors import HubReaderCloseError
from tests.hub.test_activation_client import reader_close_fault, retained_reader_error

target, report = Path(target_text), Path(report_text)
def wants_path(path):
    if boundary == "candidate":
        return any(p.name.startswith(".chunkhound-pull-") for p in path.parents)
    return path.resolve() == target.resolve()
sys.argv = ["chunkhound", mode, project, "--verbose"]
original = None
failure = None
outcome = None
held_workspaces = []
try:
    with reader_close_fault(pytest.MonkeyPatch(), wants_path) as fault_state:
        readers, held_workspaces = fault_state
        try:
            main()
        except SystemExit as error:
            outcome = error
        assert outcome is not None and outcome.code == 1
        original = retained_reader_error(outcome)
        failure = outcome.__context__
        assert isinstance(failure, HubReaderCloseError)
        assert failure.reader_error is original
        assert failure.ownership is original.ownership
        held_target = readers[0][1]
        original.ownership.assert_covers(held_target)
        assert readers[0][0]().execute("SELECT 42").fetchone() == (42,)
        with pytest.raises(native.DatabaseBusyError):
            native.acquire_install_lease(str(held_target), "duckdb")
    failure.retry_close()
    failure.retry_close()
    if boundary == "candidate":
        assert not held_target.parent.exists()
    else:
        native.acquire_install_lease(str(held_target), "duckdb").close()
    assert not list(target.parent.glob(".chunkhound-pull-*"))
    report.write_text(json.dumps({"actual_exit": outcome.code,
                                  "retained_then_retried": True}))
finally:
    if failure is not None:
        failure.retry_close()
    elif original is not None:
        original.retry_close()
    for workspace in held_workspaces:
        workspace.cleanup()
raise outcome
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(ROOT),
            boundary,
            str(target),
            str(report),
            mode,
            str(project),
        ],
        cwd=project,
        env=cli_env(project),
        capture_output=True,
        text=True,
        timeout=180,
    )
    text = result.stdout + result.stderr
    assert report.exists(), text
    assert json.loads(report.read_text()) == {
        "actual_exit": 1,
        "retained_then_retried": True,
    }
    assert result.returncode == 1, text
    assert text.count("Cannot close hub database inspection reader") == 1
    for forbidden in (
        "REVIEW_SECRET",
        "raw-close-detail",
        "raw-reader-detail",
        "Traceback",
        "Full error details",
        "continuing local indexing",
        "identity is unreadable",
        "token=secret",
    ):
        assert forbidden not in text
    assert snapshot(target) == before[0]
    assert sidecar.read_text() == retained_root
    assert embedding_http.requests == []
    assert len(hub_http.requests) == (0 if boundary == "initial" else 2)
    search = cli(project, "search", "must_not_index", project, "--regex")
    assert search.returncode == 0, search.stdout + search.stderr
    assert "No results found" in search.stdout + search.stderr
