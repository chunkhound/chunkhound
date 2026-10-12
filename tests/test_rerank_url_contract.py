"""Backend URL contracts independent of the site and Node runtime."""

import pytest

from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from chunkhound.providers.embeddings.voyageai_provider import VoyageAIEmbeddingProvider
from tests.fixtures.rerank_server_manager import RerankServerManager
from tests.rerank_server import MockRerankResult, MockRerankScenario


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("scheme", ["http", "HTTP", "hTtPs"])
def test_active_rerank_url_normalizes_scheme_only(provider, scheme):
    endpoint = "Reranker.EXAMPLE/Case%2FPath?q=KeepCase#Fragment"
    config = EmbeddingConfig(
        provider=provider, rerank_format="tei", rerank_url=f"{scheme}://{endpoint}"
    )
    assert config.rerank_url == f"{scheme.lower()}://{endpoint}"


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/rerank",
        "http:/host",
        "https://",
        "//host/rerank",
        " https://host/rerank",
    ],
)
@pytest.mark.parametrize("active", [False, True])
def test_only_active_rerank_endpoints_are_validated(provider, url, active):
    kwargs = dict(provider=provider, rerank_url=url, rerank_format="auto")
    if active:
        kwargs["rerank_model"] = "test-model"
        with pytest.raises(ValueError, match="complete HTTP"):
            EmbeddingConfig(**kwargs)
    else:
        assert EmbeddingConfig(**kwargs).rerank_url == url


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
def test_unused_endpoint_does_not_prevent_provider_creation(provider):
    config = EmbeddingConfig(
        provider=provider,
        api_key="local-test-key",
        model="local-model",
        base_url="http://localhost:9999",
        rerank_url="ftp://unused/rerank",
        rerank_format="auto",
        rerank_model=None,
    )
    EmbeddingProviderFactory.create_provider(config)


@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("base_url", [None, "https://embedding.example/v1"])
def test_active_relative_rerank_url_requires_base_url(provider, base_url):
    kwargs = dict(
        provider=provider, base_url=base_url, rerank_format="tei", rerank_url="/rerank"
    )
    if base_url:
        assert EmbeddingConfig(**kwargs).rerank_url == "/rerank"
    else:
        with pytest.raises(ValueError, match="requires base_url"):
            EmbeddingConfig(**kwargs)


def _provider(provider, entry, base_url, rerank_url):
    config = dict(
        api_key="local-test-key",
        model="local-embedding-model",
        base_url=base_url,
        rerank_url=rerank_url,
        rerank_format="tei",
        rerank_model=None,
    )
    if entry == "factory":
        return EmbeddingProviderFactory.create_provider(
            EmbeddingConfig(provider=provider, **config)
        )
    cls = OpenAIEmbeddingProvider if provider == "openai" else VoyageAIEmbeddingProvider
    return cls(**config)


@pytest.fixture
async def reranker():
    scenario = MockRerankScenario(
        name="url-contract",
        query="query",
        documents=["document"],
        results=[MockRerankResult(index=0, score=0.9)],
        response_format="tei",
    )
    async with RerankServerManager(scenarios=[scenario]) as manager:
        yield manager


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "voyageai"])
@pytest.mark.parametrize("entry", ["direct", "factory"])
@pytest.mark.parametrize("url_kind", ["uppercase", "mixedcase", "relative"])
async def test_rerank_requests_reach_the_configured_endpoint(
    reranker, provider, entry, url_kind
):
    url = f"{reranker.base_url}/rerank"
    if url_kind == "relative":
        url = "/rerank"
    else:
        url = url.replace("http:", "HTTP:" if url_kind == "uppercase" else "hTtP:")
    client = _provider(provider, entry, reranker.base_url, url)
    results = await client.rerank("query", ["document"])
    assert [(result.index, result.score) for result in results] == [(0, 0.9)]
    assert len(reranker.requests) == 1
