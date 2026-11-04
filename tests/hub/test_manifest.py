"""Public contracts for offline hub selection and storage compatibility."""

import copy
import json
import socket
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.index_format import INDEX_FORMAT_VERSION
from chunkhound.hub.errors import HubUnavailable
from chunkhound.hub.manifest import MatchKey, configured_key, select_manifest


@pytest.fixture(autouse=True)
def offline_environment(clean_environment: None, monkeypatch: pytest.MonkeyPatch):
    def refuse_network(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Manifest selection must not probe embedding endpoints")

    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse_network)


def catalogue() -> dict[str, Any]:
    return {
        "manifest_version": 2,
        "project_id": "team",
        "index_root": {"layout": "single"},
        "embeddings": [
            {
                "embedding_id": "opaque",
                "db_provider": "duckdb",
                "provider": "openai",
                "model": "hub-test",
                "dims": 8,
                "index_format_version": 1,
                "versions": [
                    {
                        "rel": "latest",
                        "build_id": "b",
                        "revision": "abc",
                        "created_at": "2026-10-11T00:00:00Z",
                        "artifact": {
                            "url": "https://cdn.invalid/object?token=secret",
                            "bytes": 99,
                            "sha256": "a" * 64,
                        },
                    }
                ],
            }
        ],
    }


def test_selects_exact_key_and_keeps_artifact_url_opaque():
    raw = json.dumps(catalogue()).encode()
    key = MatchKey("duckdb", "openai", "hub-test", 8, 1)
    selected = select_manifest(raw, key, "latest")
    assert selected.version.artifact.url == "https://cdn.invalid/object?token=secret"
    assert "token=secret" not in repr(selected)
    for wrong in [
        MatchKey("lancedb", "openai", "hub-test", 8, 1),
        MatchKey("duckdb", "azure_openai", "hub-test", 8, 1),
        MatchKey("duckdb", "openai", "hub-test", 9, 1),
        MatchKey("duckdb", "openai", "different", 8, 1),
        MatchKey("duckdb", "openai", "hub-test", 8, 2),
    ]:
        with pytest.raises(HubUnavailable):
            select_manifest(raw, wrong, "latest")
    for rel in ["-1", "-2", "0", "build-id"]:
        with pytest.raises(HubUnavailable):
            select_manifest(raw, key, rel)


@pytest.mark.parametrize("rel", ["latest", "-1", "-2"])
def test_selects_requested_retained_version_without_walking(rel):
    raw = catalogue()
    template = raw["embeddings"][0]["versions"][0]
    versions = []
    for label in ("latest", "-1", "-2"):
        version = copy.deepcopy(template)
        version["rel"] = label
        version["revision"] = label
        versions.append(version)
    raw["embeddings"][0]["versions"] = versions
    selected = select_manifest(
        json.dumps(raw).encode(), MatchKey("duckdb", "openai", "hub-test", 8, 1), rel
    )
    assert selected.version.revision == rel


@pytest.mark.parametrize(
    "fault",
    ["version", "duplicate", "missing-rel", "relative-url", "duplicate-rel", "four"],
)
def test_manifest_declines_unknown_ambiguous_or_nonopaque_contract(fault):
    raw = catalogue()
    versions = raw["embeddings"][0]["versions"]
    if fault == "version":
        raw["manifest_version"] = 3
    elif fault == "duplicate":
        raw["embeddings"].append(raw["embeddings"][0].copy())
    elif fault == "missing-rel":
        raw["embeddings"][0]["versions"] = []
    elif fault == "relative-url":
        versions[0]["artifact"] = {
            "path": "index.tar.gz",
            "bytes": 99,
            "sha256": "a" * 64,
        }
    elif fault == "duplicate-rel":
        versions.append(copy.deepcopy(versions[0]))
    else:
        versions.extend(copy.deepcopy(versions[0]) for _ in range(3))
    with pytest.raises(HubUnavailable):
        select_manifest(
            json.dumps(raw).encode(),
            MatchKey("duckdb", "openai", "hub-test", 8, 1),
            "latest",
        )


@pytest.mark.parametrize(
    ("level", "field", "value"),
    [
        ("variant", "dims", "8"),
        ("variant", "dims", True),
        ("variant", "dims", 0),
        ("variant", "index_format_version", 0),
        ("variant", "model", 123),
        ("version", "rel", "0"),
        ("version", "rel", "build-id"),
        ("artifact", "bytes", "99"),
        ("artifact", "bytes", 0),
        ("artifact", "sha256", "not-a-digest"),
    ],
)
def test_invalid_wire_values_are_not_coerced(level, field, value):
    raw = catalogue()
    variant = raw["embeddings"][0]
    version = variant["versions"][0]
    target = {"variant": variant, "version": version, "artifact": version["artifact"]}
    target[level][field] = value
    with pytest.raises(HubUnavailable, match="invalid or unsupported"):
        select_manifest(
            json.dumps(raw).encode(),
            MatchKey("duckdb", "openai", "hub-test", 8, 1),
            "latest",
        )


def test_diagnostic_metadata_does_not_choose_the_variant():
    raw = catalogue()
    raw["project_id"] = "other-team"
    raw["index_root"] = ["diagnostic-only"]
    raw["future_metadata"] = {"ignored": True}
    other = copy.deepcopy(raw["embeddings"][0])
    other["embedding_id"] = "same-opaque-id"
    other["model"] = "different"
    raw["embeddings"].insert(0, other)
    selected = select_manifest(
        json.dumps(raw).encode(),
        MatchKey("duckdb", "openai", "hub-test", 8, 1),
        "latest",
    )
    assert selected.variant.embedding_id == "opaque"


@pytest.mark.parametrize("malformed", [False, True])
def test_manifest_diagnostics_do_not_expose_urls_or_query_secrets(malformed):
    raw = catalogue()
    raw["embeddings"][0]["model"] = "https://private.invalid/model?key=model-secret"
    payload = json.dumps(raw).encode()
    if malformed:
        payload = payload[:-1]
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        with pytest.raises(HubUnavailable) as error:
            select_manifest(
                payload, MatchKey("duckdb", "openai", "hub-test", 8, 1), "latest"
            )
    finally:
        logger.remove(sink)
    diagnostic = str(error.value) + "".join(messages)
    for secret in ("cdn.invalid", "token=secret", "private.invalid", "model-secret"):
        assert secret not in diagnostic
    assert error.value.__suppress_context__ or error.value.__context__ is None


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("openai", "private-unknown", "http://127.0.0.1:1/v1"),
        ("openai", "text-embedding-3-small", "http://127.0.0.1:1/v1"),
        ("voyageai", "private-unknown", None),
    ],
)
def test_unknown_dimensions_decline_without_guessing(
    tmp_path, provider, model, base_url
):
    config = Config(
        target_dir=tmp_path,
        embedding={
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": "test",
        },
    )
    with pytest.raises(HubUnavailable, match="dimensions"):
        configured_key(config)


@pytest.mark.parametrize(
    ("provider", "model", "base_url", "output_dims", "expected"),
    [
        ("openai", "text-embedding-3-small", None, None, 1536),
        ("openai", "text-embedding-3-small", None, 512, 512),
        ("openai", "private-unknown", "http://127.0.0.1:1/v1", 8, 8),
        ("voyageai", "voyage-code-3", None, None, 1024),
        ("voyageai", "voyage-code-3", None, 512, 512),
        ("voyageai", "private-unknown", None, 8, 8),
    ],
)
def test_known_or_explicit_dimensions_resolve_offline_without_mutation(
    tmp_path: Path, provider, model, base_url, output_dims, expected
):
    config = Config(
        target_dir=tmp_path,
        embedding={
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": "test",
            "output_dims": output_dims,
        },
    )
    before = config.model_dump()
    files = list(tmp_path.rglob("*"))
    assert configured_key(config) == MatchKey(
        "duckdb", provider, model, expected, INDEX_FORMAT_VERSION
    )
    assert config.model_dump() == before
    assert list(tmp_path.rglob("*")) == files


@pytest.mark.parametrize(
    ("provider", "output_dims", "azure", "failure"),
    [
        ("openai", 8, False, None),
        ("voyageai", 8, False, None),
        ("openai", None, False, "dimensions"),
        ("voyageai", None, False, "dimensions"),
        ("openai", 8, True, "interoperability"),
    ],
)
def test_configured_key_debug_diagnostics_do_not_expose_configuration_secrets(
    tmp_path, provider, output_dims, azure, failure
):
    model = "https://model-secret.invalid/private?token=model-query-secret"
    embedding = {
        "provider": provider,
        "model": model,
        "api_key": "api-key-secret",
        "output_dims": output_dims,
        "rerank_model": "reranker?token=rerank-model-secret",
        "rerank_url": "https://rerank-secret.invalid/rerank?token=rerank-query-secret",
        "rerank_format": "cohere",
    }
    if azure:
        embedding.update(
            azure_endpoint="https://endpoint-secret.openai.azure.com",
            api_version="2024-02-01-versionsecret",
            azure_deployment="deployment?token=deployment-query-secret",
        )
    else:
        embedding["base_url"] = (
            "https://endpoint-user:endpoint-password@endpoint-secret.invalid/v1"
            "?token=endpoint-query-secret"
        )
    config = Config(target_dir=tmp_path, embedding=embedding)
    messages: list[tuple[str, str]] = []
    sink = logger.add(
        lambda message: messages.append((message.record["level"].name, str(message))),
        level="DEBUG",
    )
    diagnostic = ""
    try:
        if failure:
            with pytest.raises(HubUnavailable, match=failure) as error:
                configured_key(config)
            diagnostic = str(error.value)
        else:
            assert configured_key(config) == MatchKey(
                "duckdb", provider, model, output_dims, INDEX_FORMAT_VERSION
            )
    finally:
        logger.remove(sink)
    assert any(level == "DEBUG" for level, _ in messages)
    diagnostic += "".join(message for _, message in messages)
    secrets = (
        "model-secret.invalid",
        "model-query-secret",
        "api-key-secret",
        "rerank-model-secret",
        "rerank-secret.invalid",
        "rerank-query-secret",
        "endpoint-user",
        "endpoint-password",
        "endpoint-secret",
        "endpoint-query-secret",
        "versionsecret",
        "deployment-query-secret",
    )
    assert not [secret for secret in secrets if secret in diagnostic]


def test_azure_selector_does_not_alias_storage_identity(tmp_path):
    config = Config(
        target_dir=tmp_path,
        embedding={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test",
            "azure_endpoint": "https://example.openai.azure.com",
            "api_version": "2024-02-01",
            "azure_deployment": "deployment",
        },
    )
    with pytest.raises(HubUnavailable, match="interoperability"):
        configured_key(config)


@pytest.mark.parametrize("disabled", [False, True])
def test_missing_or_disabled_embeddings_decline(tmp_path, disabled):
    config = Config(
        target_dir=tmp_path,
        embedding={
            "provider": "openai",
            "model": "text-embedding-3-small",
            "api_key": "test",
        },
    )
    if disabled:
        config.embeddings_disabled = True
    else:
        config.embedding = None
    with pytest.raises(HubUnavailable, match="embedding configuration"):
        configured_key(config)


def test_provider_configuration_failure_is_a_sanitized_soft_failure(tmp_path):
    config = Config(
        target_dir=tmp_path,
        embedding={"provider": "openai", "model": "text-embedding-3-small"},
    )
    with pytest.raises(
        HubUnavailable, match="Cannot resolve configured embedding identity offline"
    ) as error:
        configured_key(config)
    assert error.value.__suppress_context__
