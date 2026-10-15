"""Tests for EmbeddingConfig.is_provider_configured() contract and provider creation."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.providers.embeddings import openai_provider as openai_provider_module
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.providers.embeddings.voyageai_provider import VoyageAIEmbeddingProvider


class TestIsProviderConfigured:
    """Verify is_provider_configured() across provider/endpoint/key combinations."""

    def test_openai_official_without_key(self):
        cfg = EmbeddingConfig(provider="openai", api_key=None)
        assert cfg.is_provider_configured() is False

    def test_openai_official_with_key(self):
        cfg = EmbeddingConfig(provider="openai", api_key="sk-test")
        assert cfg.is_provider_configured() is True

    @pytest.mark.parametrize(
        "url",
        [
            "http://localhost:11434",
            "http://192.168.1.100:8080",
            "https://my-server.local/v1",
        ],
    )
    def test_openai_custom_endpoint_without_key(self, url: str):
        cfg = EmbeddingConfig(
            provider="openai",
            base_url=url,
            api_key=None,
        )
        assert cfg.is_provider_configured() is True

    def test_azure_openai_with_key_and_version(self):
        cfg = EmbeddingConfig(
            provider="openai",
            azure_endpoint="https://my.openai.azure.com",
            api_key="az-key",
            api_version="2024-02-01",
        )
        assert cfg.is_provider_configured() is True

    def test_azure_openai_without_key(self):
        cfg = EmbeddingConfig(
            provider="openai",
            azure_endpoint="https://my.openai.azure.com",
            api_key=None,
            api_version="2024-02-01",
        )
        assert cfg.is_provider_configured() is False

    def test_openai_explicit_official_url_without_key(self):
        cfg = EmbeddingConfig(provider="openai", base_url="https://api.openai.com/v1", api_key=None)
        assert cfg.is_provider_configured() is False

    def test_voyageai_official_without_key(self):
        cfg = EmbeddingConfig(provider="voyageai", api_key=None)
        assert cfg.is_provider_configured() is False

    def test_voyageai_explicit_official_url_without_key(self):
        cfg = EmbeddingConfig(
            provider="voyageai",
            base_url="https://api.voyageai.com/v1",
            api_key=None,
        )
        assert cfg.is_provider_configured() is False

    def test_voyageai_custom_endpoint_without_key(self):
        cfg = EmbeddingConfig(
            provider="voyageai",
            base_url="http://localhost:8080",
            api_key=None,
        )
        assert cfg.is_provider_configured() is True


class TestFactoryRejectsUnconfigured:
    """Verify factory raises when provider is not configured."""

    def test_factory_rejects_unconfigured_provider(self):
        cfg = EmbeddingConfig(provider="openai", api_key=None)
        assert cfg.is_provider_configured() is False
        with pytest.raises(ValueError, match="Incomplete configuration"):
            EmbeddingProviderFactory.create_provider(cfg)


class TestFactoryCreatesProvider:
    """Verify EmbeddingProviderFactory.create_provider() returns real provider instances."""

    def test_factory_creates_provider_with_custom_endpoint_no_key(self):
        cfg = EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:11434",
            api_key=None,
            ssl_verify=False,
        )
        provider = EmbeddingProviderFactory.create_provider(cfg)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider._ssl_verify is False

    def test_factory_creates_provider_with_voyageai_custom_endpoint(self):
        cfg = EmbeddingConfig(
            provider="voyageai",
            base_url="http://localhost:8080",
            api_key=None,
        )
        provider = EmbeddingProviderFactory.create_provider(cfg)
        assert isinstance(provider, VoyageAIEmbeddingProvider)


@pytest.mark.asyncio
async def test_openai_provider_initializes_custom_endpoint_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """Provider client init should accept keyless custom OpenAI-compatible endpoints."""

    captured: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(
        openai_provider_module,
        "openai",
        SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI, AsyncAzureOpenAI=_FakeAsyncOpenAI),
    )

    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
    )

    await provider._ensure_client()

    assert provider.is_available() is True
    assert captured["api_key"] == "not-required"
    assert captured["base_url"] == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    ("provider_kwargs", "expected"),
    [
        ({"model": "text-embedding-3-small", "api_key": None}, False),
        (
            {
                "base_url": "http://localhost:11434/v1",
                "model": "nomic-embed-text",
                "api_key": None,
            },
            True,
        ),
        (
            {
                "azure_endpoint": "https://my-resource.openai.azure.com",
                "api_key": None,
                "api_version": "2024-02-01",
                "model": "text-embedding-3-small",
            },
            False,
        ),
        (
            {
                "azure_endpoint": "https://my-resource.openai.azure.com",
                "api_key": "az-key",
                "api_version": None,
                "model": "text-embedding-3-small",
            },
            False,
        ),
    ],
)
def test_openai_provider_is_available_auth_matrix(
    monkeypatch: pytest.MonkeyPatch,
    provider_kwargs: dict[str, object],
    expected: bool,
):
    """Availability should follow the endpoint auth contract."""
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)

    provider = OpenAIEmbeddingProvider(**provider_kwargs)

    assert provider.is_available() is expected


@pytest.mark.asyncio
async def test_openai_provider_health_check_reports_keyless_custom_endpoint_as_configured(
    monkeypatch: pytest.MonkeyPatch,
):
    """Health check should report auth as satisfied without inventing an API key."""
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)

    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key=None,
    )
    provider.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
    provider._discovered_native_dims = 3

    result = await provider.health_check()

    assert result["available"] is True
    assert result["authentication_required"] is False
    assert result["authentication_configured"] is True
    assert result["api_key_configured"] is False
    assert result["errors"] == []
    assert result["connectivity"] == "ok"


@pytest.mark.asyncio
async def test_openai_provider_health_check_reports_missing_official_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """Official OpenAI health check should fail fast when the API key is missing."""
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)

    provider = OpenAIEmbeddingProvider(
        model="text-embedding-3-small",
        api_key=None,
    )

    result = await provider.health_check()

    assert result["available"] is False
    assert result["authentication_required"] is True
    assert result["authentication_configured"] is False
    assert result["api_key_configured"] is False
    assert "API key not configured" in result["errors"]


@pytest.mark.asyncio
async def test_openai_provider_health_check_reports_missing_azure_api_version(
    monkeypatch: pytest.MonkeyPatch,
):
    """Azure health check should require API version in addition to the API key."""
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)

    provider = OpenAIEmbeddingProvider(
        azure_endpoint="https://my-resource.openai.azure.com",
        api_key="az-key",
        api_version=None,
        model="text-embedding-3-small",
    )

    result = await provider.health_check()

    assert result["available"] is False
    assert result["authentication_required"] is True
    assert result["authentication_configured"] is False
    assert result["api_key_configured"] is True
    assert "API version not configured" in result["errors"]


@pytest.mark.asyncio
async def test_openai_provider_health_check_reports_missing_azure_api_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """Azure health check should report missing API key explicitly."""
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)

    provider = OpenAIEmbeddingProvider(
        azure_endpoint="https://my-resource.openai.azure.com",
        api_key=None,
        api_version="2024-02-01",
        model="text-embedding-3-small",
    )

    result = await provider.health_check()

    assert result["available"] is False
    assert result["authentication_required"] is True
    assert result["authentication_configured"] is False
    assert result["api_key_configured"] is False
    assert "API key not configured" in result["errors"]


@pytest.mark.asyncio
async def test_openai_provider_applies_explicit_ssl_verify_false(
    monkeypatch: pytest.MonkeyPatch,
):
    """Explicit ssl_verify=false should create an insecure custom transport."""

    captured: dict[str, object] = {}
    httpx_calls: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self) -> None:
            return None

    class _FakeHTTPClient:
        async def aclose(self) -> None:
            return None

    def _fake_async_client(**kwargs):
        httpx_calls.update(kwargs)
        return _FakeHTTPClient()

    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(
        openai_provider_module,
        "openai",
        SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI, AsyncAzureOpenAI=_FakeAsyncOpenAI),
    )
    monkeypatch.setattr(openai_provider_module.httpx, "AsyncClient", _fake_async_client)

    provider = OpenAIEmbeddingProvider(
        base_url="https://localhost:11434/v1",
        model="nomic-embed-text",
        ssl_verify=False,
    )

    await provider._ensure_client()

    assert captured["base_url"] == "https://localhost:11434/v1"
    assert "http_client" in captured
    assert httpx_calls["verify"] is False


@pytest.mark.asyncio
async def test_openai_provider_ignores_ssl_verify_without_base_url(
    monkeypatch: pytest.MonkeyPatch,
):
    """ssl_verify must not affect the default endpoint path when base_url is unset."""

    captured: dict[str, object] = {}
    httpx_calls: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def close(self) -> None:
            return None

    class _FakeHTTPClient:
        async def aclose(self) -> None:
            return None

    def _fake_async_client(**kwargs):
        httpx_calls.update(kwargs)
        return _FakeHTTPClient()

    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(
        openai_provider_module,
        "openai",
        SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI, AsyncAzureOpenAI=_FakeAsyncOpenAI),
    )
    monkeypatch.setattr(openai_provider_module.httpx, "AsyncClient", _fake_async_client)

    provider = OpenAIEmbeddingProvider(api_key="sk-test", ssl_verify=False)
    await provider._ensure_client()

    # http_client is always present now (it bounds the connection pool to
    # guard against FD exhaustion), but its TLS verification must stay on —
    # ssl_verify is only meant to apply to custom/self-hosted base_urls.
    assert "http_client" in captured
    assert httpx_calls["verify"] is True


@dataclass
class _PatchedOpenAIClient:
    """Handle returned by `patched_openai_httpx`, exposing what the fakes captured."""

    limits_calls: dict[str, object]
    client_kwargs: dict[str, object]


@pytest.fixture
def patched_openai_httpx(monkeypatch: pytest.MonkeyPatch) -> _PatchedOpenAIClient:
    """Stub openai.AsyncOpenAI/AsyncAzureOpenAI and httpx.AsyncClient so
    OpenAIEmbeddingProvider._ensure_client() (both the standard and Azure
    branches route through the same fake class here) runs with no real
    network client, and expose what each captured:
      - limits_calls["limits"]: the httpx.Limits passed to httpx.AsyncClient(...)
      - client_kwargs: kwargs the openai SDK client class itself received
        (used by the Azure tests to assert an http_client= was actually passed).
    """
    limits_calls: dict[str, object] = {}
    client_kwargs: dict[str, object] = {}

    class _FakeOpenAIClient:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)

        async def close(self) -> None:
            return None

    class _FakeHTTPClient:
        async def aclose(self) -> None:
            return None

    def _fake_async_client(**kwargs):
        limits_calls["limits"] = kwargs["limits"]
        return _FakeHTTPClient()

    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(
        openai_provider_module,
        "openai",
        SimpleNamespace(
            AsyncOpenAI=_FakeOpenAIClient, AsyncAzureOpenAI=_FakeOpenAIClient
        ),
    )
    monkeypatch.setattr(openai_provider_module.httpx, "AsyncClient", _fake_async_client)
    return _PatchedOpenAIClient(limits_calls=limits_calls, client_kwargs=client_kwargs)


@pytest.mark.asyncio
async def test_openai_provider_connection_pool_defaults_to_ten(
    patched_openai_httpx: _PatchedOpenAIClient,
):
    """Without an explicit concurrency hint, the pool stays at its historical size."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test")
    await provider._ensure_client()

    limits = patched_openai_httpx.limits_calls["limits"]
    assert limits.max_connections == 10
    assert limits.max_keepalive_connections == 5


@pytest.mark.asyncio
async def test_openai_provider_connection_pool_scales_with_configured_concurrency(
    patched_openai_httpx: _PatchedOpenAIClient,
):
    """A high explicit max_concurrent_batches must not be silently throttled.

    Regression test: EmbeddingService gates concurrent embed_batch() calls on
    one shared provider instance with asyncio.Semaphore(max_concurrent_batches),
    which is user-configurable with no upper bound. A flat connection-pool cap
    lower than that value would serialize requests below the user's configured
    concurrency with no error or warning.
    """
    provider = OpenAIEmbeddingProvider(api_key="sk-test", max_concurrent_batches=50)
    await provider._ensure_client()

    limits = patched_openai_httpx.limits_calls["limits"]
    assert limits.max_connections >= 50
    assert limits.max_keepalive_connections >= 5


@pytest.mark.asyncio
async def test_azure_openai_provider_connection_pool_defaults_to_ten(
    patched_openai_httpx: _PatchedOpenAIClient,
):
    """The Azure client path must also bound its pool, not just the standard one.

    Regression test: _ensure_azure_client() used to build AsyncAzureOpenAI
    with no http_client= at all, silently falling back to the openai SDK's
    own internal httpx defaults — reintroducing the FD-exhaustion risk the
    standard client's pool-sizing fix was meant to close, for every
    Azure-backed Rust embed thread.
    """
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        azure_endpoint="https://foo.openai.azure.com",
        api_version="2024-02-01",
    )
    await provider._ensure_client()

    assert "http_client" in patched_openai_httpx.client_kwargs, (
        "AsyncAzureOpenAI must receive a pool-bounded http_client, not fall "
        "back to the SDK's own internal httpx defaults"
    )
    limits = patched_openai_httpx.limits_calls["limits"]
    assert limits.max_connections == 10
    assert limits.max_keepalive_connections == 5


@pytest.mark.asyncio
async def test_azure_openai_provider_connection_pool_scales_with_configured_concurrency(
    patched_openai_httpx: _PatchedOpenAIClient,
):
    """A high explicit max_concurrent_batches must not be silently throttled
    on the Azure client path either — mirrors the standard-client regression
    test above."""
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        azure_endpoint="https://foo.openai.azure.com",
        api_version="2024-02-01",
        max_concurrent_batches=50,
    )
    await provider._ensure_client()

    assert "http_client" in patched_openai_httpx.client_kwargs
    limits = patched_openai_httpx.limits_calls["limits"]
    assert limits.max_connections >= 50
    assert limits.max_keepalive_connections >= 5


@pytest.mark.asyncio
async def test_openai_rerank_ssl_override_applies_without_embedding_base_url(
    monkeypatch: pytest.MonkeyPatch,
):
    """rerank_ssl_verify must work for explicit rerank_url even on official embeddings."""

    captured_verify: dict[str, object] = {}

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            return None

        async def close(self) -> None:
            return None

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _FakeHTTPClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *_args, **_kwargs):
            return _FakeResponse()

    def _fake_async_client(**kwargs):
        captured_verify.update(kwargs)
        return _FakeHTTPClient()

    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True)
    monkeypatch.setattr(
        openai_provider_module,
        "openai",
        SimpleNamespace(AsyncOpenAI=_FakeAsyncOpenAI, AsyncAzureOpenAI=_FakeAsyncOpenAI),
    )
    monkeypatch.setattr(openai_provider_module.httpx, "AsyncClient", _fake_async_client)

    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        model="text-embedding-3-small",
        rerank_model="local-ranker",
        rerank_url="https://localhost:8001/rerank",
        rerank_ssl_verify=False,
    )

    results = await provider.rerank("query", ["doc one"])

    assert len(results) == 1
    assert captured_verify["verify"] is False
