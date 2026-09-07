"""Deterministic embedding and rerank provider contract tests."""

import asyncio
import sys
from pathlib import Path

import pytest

from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
from tests.fixtures.rerank_server_manager import RerankServerManager
from tests.rerank_server import MockRerankResult, MockRerankScenario

# Add parent directory to path to import chunkhound modules.
sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.fast
@pytest.mark.asyncio
async def test_official_openai_validation() -> None:
    """Official OpenAI must require an API key."""
    provider = OpenAIEmbeddingProvider(api_key="sk-fake-key")
    assert provider.api_key == "sk-fake-key"

    provider = OpenAIEmbeddingProvider()
    with pytest.raises(
        ValueError, match="OpenAI API key is required for official OpenAI API"
    ):
        await provider._ensure_client()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_custom_endpoint_validation() -> None:
    """Custom endpoints may omit an API key."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )
    assert provider.base_url == "http://localhost:11434"

    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:1234", api_key="custom-key"
    )
    assert provider.api_key == "custom-key"


def test_url_detection_logic() -> None:
    """Only the canonical OpenAI host should be treated as official."""
    official_urls = [
        None,
        "https://api.openai.com",
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/",
    ]
    for url in official_urls:
        provider = OpenAIEmbeddingProvider(base_url=url)
        is_official = not provider._base_url or (
            provider._base_url.startswith("https://api.openai.com")
            and (
                provider._base_url == "https://api.openai.com"
                or provider._base_url.startswith("https://api.openai.com/")
            )
        )
        assert is_official

    custom_urls = [
        "http://localhost:11434",
        "https://api.example.com/v1/embeddings",
        "https://api.openai.com.evil.com/v1",
        "http://api.openai.com/v1",
    ]
    for url in custom_urls:
        provider = OpenAIEmbeddingProvider(base_url=url)
        is_official = not provider._base_url or (
            provider._base_url.startswith("https://api.openai.com")
            and (
                provider._base_url == "https://api.openai.com"
                or provider._base_url.startswith("https://api.openai.com/")
            )
        )
        assert not is_official


@pytest.mark.fast
@pytest.mark.asyncio
async def test_custom_endpoint_mock_behavior() -> None:
    """Client setup for custom endpoints must not fail on missing API key."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )

    try:
        await provider._ensure_client()
    except Exception as exc:  # pragma: no cover - network is irrelevant here
        assert "API key" not in str(exc)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_ollama_style_rerank_configuration_uses_mock_server() -> None:
    """OpenAI provider should rerank against a separate deterministic HTTP service."""
    query = "python function definition"
    documents = [
        "def calculate_sum(a, b): return a + b",
        "import numpy as np",
        "class Calculator: pass",
        "function add(x, y) { return x + y; }",
    ]
    scenario = MockRerankScenario(
        name="cohere-provider-contract",
        query=query,
        documents=documents,
        results=[
            MockRerankResult(index=0, score=0.97),
            MockRerankResult(index=2, score=0.66),
            MockRerankResult(index=3, score=0.61),
            MockRerankResult(index=1, score=0.11),
        ],
        response_format="cohere",
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="dummy-key-for-custom-endpoint",
            rerank_model="test-reranker",
            rerank_url=f"{manager.base_url}/rerank",
        )

        results = await provider.rerank(query, documents, top_k=3)

        assert [(result.index, result.score) for result in results] == [
            (0, 0.97),
            (2, 0.66),
            (3, 0.61),
        ]
        assert manager.requests == [
            {
                "model": "test-reranker",
                "query": query,
                "documents": documents,
                "top_n": 3,
            }
        ]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_tei_reranking_format_with_model_uses_texts_payload() -> None:
    """TEI format should send texts and parse score fields."""
    # This contract test uses the real provider + mock HTTP server so request shape
    # and response parsing stay covered together across refactors.
    query = "python programming"
    documents = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "def calculate_sum(a, b): return a + b",
    ]
    scenario = MockRerankScenario(
        name="tei-with-model",
        query=query,
        documents=documents,
        results=[
            MockRerankResult(index=2, score=0.91),
            MockRerankResult(index=0, score=0.74),
            MockRerankResult(index=1, score=0.09),
        ],
        response_format="tei",
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="dummy-key",
            rerank_model="BAAI/bge-reranker-base",
            rerank_url=f"{manager.base_url}/rerank",
            rerank_format="tei",
        )

        results = await provider.rerank(query, documents, top_k=2)

        assert [(result.index, result.score) for result in results] == [
            (2, 0.91),
            (0, 0.74),
        ]
        assert manager.requests == [{"query": query, "texts": documents}]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_tei_reranking_format_without_model_is_supported() -> None:
    """TEI deployments may fix the model server-side and omit it from config."""
    query = "server side model"
    documents = ["doc1", "doc2"]
    scenario = MockRerankScenario(
        name="tei-without-model",
        query=query,
        documents=documents,
        results=[MockRerankResult(index=1, score=0.88)],
        response_format="tei",
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="dummy-key",
            rerank_url=f"{manager.base_url}/rerank",
            rerank_format="tei",
        )

        assert provider.supports_reranking() is True
        results = await provider.rerank(query, documents)

        assert [(result.index, result.score) for result in results] == [(1, 0.88)]
        assert manager.requests == [{"query": query, "texts": documents}]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_tei_bare_array_response_format() -> None:
    """Bare-array TEI responses should normalize into ChunkHound rerank results."""
    query = "bare format"
    documents = ["doc1", "doc2"]
    scenario = MockRerankScenario(
        name="tei-bare-array",
        query=query,
        documents=documents,
        results=[
            MockRerankResult(index=0, score=0.95),
            MockRerankResult(index=1, score=0.42),
        ],
        response_format="tei-bare",
    )

    async with RerankServerManager(scenarios=[scenario]) as manager:
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            base_url="http://localhost:8080",
            model="text-embedding-3-small",
            rerank_url=f"{manager.base_url}/rerank",
            rerank_format="tei",
        )

        results = await provider.rerank(query, documents)

        assert [(result.index, result.score) for result in results] == [
            (0, 0.95),
            (1, 0.42),
        ]
        # Verify the provider sent the correct TEI-shaped request, not Cohere-shaped.
        assert manager.requests == [{"query": query, "texts": documents}]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_auto_format_detection_caches_format() -> None:
    """Auto mode should detect response shape once and reuse that format."""
    first_documents = ["doc1", "doc2", "doc3"]
    second_documents = ["docA", "docB", "docC"]
    scenarios = [
        MockRerankScenario(
            name="auto-detect-first",
            query="query one",
            documents=first_documents,
            results=[MockRerankResult(index=1, score=0.8)],
            response_format="tei",
        ),
        MockRerankScenario(
            name="auto-detect-second",
            query="query two",
            documents=second_documents,
            results=[MockRerankResult(index=2, score=0.7)],
            response_format="tei",
        ),
    ]

    async with RerankServerManager(scenarios=scenarios) as manager:
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="test-key",
            rerank_url=f"{manager.base_url}/rerank",
            rerank_format="auto",
        )

        assert provider._detected_rerank_format is None
        first = await provider.rerank("query one", first_documents)
        assert [(result.index, result.score) for result in first] == [(1, 0.8)]
        assert provider._detected_rerank_format == "tei"

        second = await provider.rerank("query two", second_documents)
        assert [(result.index, result.score) for result in second] == [(2, 0.7)]
        assert provider._detected_rerank_format == "tei"
        assert manager.requests == [
            {"query": "query one", "texts": first_documents},
            {"query": "query two", "texts": second_documents},
        ]


@pytest.mark.fast
@pytest.mark.asyncio
async def test_concurrent_rerank_calls_share_auto_detected_format_without_races() -> (
    None
):
    """Concurrent auto-mode calls should all succeed with one stable detected format."""
    documents = ["doc1", "doc2", "doc3"]
    scenarios = [
        MockRerankScenario(
            name=f"concurrent-{index}",
            query=f"query {index}",
            documents=documents,
            results=[MockRerankResult(index=index % 3, score=0.9 - (index * 0.01))],
            response_format="tei",
        )
        for index in range(10)
    ]

    async with RerankServerManager(scenarios=scenarios) as manager:
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="test-key",
            rerank_url=f"{manager.base_url}/rerank",
            rerank_format="auto",
        )

        results_list = await asyncio.gather(
            *[provider.rerank(f"query {index}", documents) for index in range(10)]
        )

        assert [results[0].index for results in results_list] == [
            index % 3 for index in range(10)
        ]
        assert provider._detected_rerank_format == "tei"
        assert len(manager.requests) == 10
        assert {request["query"] for request in manager.requests} == {
            f"query {index}" for index in range(10)
        }
        assert all(request["texts"] == documents for request in manager.requests)


@pytest.mark.fast
@pytest.mark.asyncio
async def test_malformed_rerank_response() -> None:
    """Malformed responses should fail loudly or skip invalid rows."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="test-key",
        rerank_format="tei",
    )

    with pytest.raises(ValueError, match="missing 'results' field"):
        await provider._parse_rerank_response({"status": "ok"}, "tei", num_documents=3)

    with pytest.raises(ValueError, match="'results' must be a list"):
        await provider._parse_rerank_response(
            {"results": "not a list"}, "tei", num_documents=3
        )

    with pytest.raises(ValueError, match="must have 'index' field"):
        await provider._parse_rerank_response(
            {"results": [{"score": 0.5}]}, "tei", num_documents=3
        )

    with pytest.raises(
        ValueError, match="must have 'relevance_score' or 'score' field"
    ):
        await provider._parse_rerank_response(
            {"results": [{"index": 0}]}, "tei", num_documents=3
        )

    assert (
        await provider._parse_rerank_response({"results": []}, "tei", num_documents=3)
        == []
    )

    mixed_results = {
        "results": [
            {"index": 0, "score": 0.9},
            {"index": "invalid", "score": 0.8},
            {"index": 2, "score": "invalid"},
            {"index": 3, "score": 0.7},
        ]
    }
    parsed = await provider._parse_rerank_response(
        mixed_results, "tei", num_documents=4
    )
    assert [(result.index, result.score) for result in parsed] == [(0, 0.9), (3, 0.7)]

    out_of_bounds_results = {
        "results": [
            {"index": 0, "score": 0.9},
            {"index": 5, "score": 0.8},
            {"index": 2, "score": 0.7},
        ]
    }
    parsed = await provider._parse_rerank_response(
        out_of_bounds_results, "tei", num_documents=3
    )
    assert [(result.index, result.score) for result in parsed] == [(0, 0.9), (2, 0.7)]

    negative_index_results = {
        "results": [
            {"index": -1, "score": 0.9},
            {"index": 0, "score": 0.8},
        ]
    }
    parsed = await provider._parse_rerank_response(
        negative_index_results, "tei", num_documents=3
    )
    assert [(result.index, result.score) for result in parsed] == [(0, 0.8)]


def test_cohere_format_requires_model() -> None:
    """Cohere format should fail fast without rerank_model."""
    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="cohere",
        )


def test_rerank_format_propagates_through_config() -> None:
    """rerank_format should flow from config into the provider instance."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    config = EmbeddingConfig(
        provider="openai",
        base_url="http://localhost:8001",
        model="text-embedding-3-small",
        rerank_url="/rerank",
        rerank_format="tei",
    )

    provider = EmbeddingProviderFactory.create_provider(config)
    assert provider._rerank_format == "tei"


def test_cohere_format_validation_requires_model() -> None:
    """EmbeddingConfig should enforce the Cohere model requirement."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:8001",
            model="text-embedding-3-small",
            rerank_format="cohere",
        )


def test_tei_format_validation_without_model() -> None:
    """TEI format should validate without rerank_model."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    config = EmbeddingConfig(
        provider="openai",
        base_url="http://localhost:8001",
        model="text-embedding-3-small",
        rerank_url="/rerank",
        rerank_format="tei",
    )
    assert config.rerank_format == "tei"


def test_voyage_format_propagates_through_config() -> None:
    """rerank_format='voyage' should be accepted and flow into the provider."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig
    from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

    config = EmbeddingConfig(
        provider="voyageai",
        base_url="https://ai.mongodb.com/v1",
        model="voyage-3.5",
        rerank_model="rerank-2.5",
        rerank_format="voyage",
    )
    assert config.rerank_format == "voyage"

    provider = EmbeddingProviderFactory.create_provider(config)
    assert provider._rerank_format == "voyage"


def test_voyage_format_requires_model() -> None:
    """Voyage-native format, like Cohere, needs an explicit rerank_model."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    with pytest.raises(ValueError, match="rerank_model is required"):
        EmbeddingConfig(
            provider="voyageai",
            base_url="https://ai.mongodb.com/v1",
            model="voyage-3.5",
            rerank_url="/rerank",
            rerank_format="voyage",
        )


def test_official_voyage_config_stays_on_sdk_rerank() -> None:
    """Adding the 'voyage' format must not shift the official-API default:
    official VoyageAI (no base_url) keeps rerank_format='auto' and no
    rerank_url, so reranking stays on the SDK path, never the HTTP path."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    config = EmbeddingConfig(
        provider="voyageai",
        model="voyage-3.5",
        rerank_model="rerank-2.5",
    )
    assert config.rerank_format == "auto"
    assert config.rerank_url is None


def test_supports_reranking_with_incomplete_cohere_config() -> None:
    """Fail-fast validation should reject incomplete Cohere rerank config."""
    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        OpenAIEmbeddingProvider(
            base_url="http://localhost:8001",
            model="text-embedding-3-small",
            rerank_url="/rerank",
            rerank_format="cohere",
        )


def test_supports_reranking_with_tei_config() -> None:
    """TEI config only needs a rerank URL."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:8001",
        model="text-embedding-3-small",
        rerank_url="/rerank",
        rerank_format="tei",
    )
    assert provider.supports_reranking()


def test_embedding_manager() -> None:
    """EmbeddingManager should register and return providers."""
    manager = EmbeddingManager()
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key-for-testing", model="text-embedding-3-small"
    )

    manager.register_provider(provider, set_default=True)

    retrieved = manager.get_provider()
    assert retrieved.name == "openai"
    assert retrieved.model == "text-embedding-3-small"
    assert "openai" in manager.list_providers()


@pytest.mark.fast
@pytest.mark.asyncio
async def test_mock_embedding_generation() -> None:
    """embed([]) should short-circuit before any network call."""
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key-for-testing", model="text-embedding-3-small"
    )
    assert await provider.embed([]) == []


def test_provider_integration() -> None:
    """EmbeddingManager should expose registered providers by name."""
    manager = EmbeddingManager()
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key", model="text-embedding-3-small"
    )
    manager.register_provider(provider)

    assert {"openai"}.issubset(set(manager.list_providers()))
    assert manager.get_provider("openai").name == "openai"


def test_relative_rerank_url_requires_base_url() -> None:
    """Relative rerank URLs need base_url; absolute ones do not."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    with pytest.raises(ValueError, match="requires base_url or explicit rerank_url"):
        EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            rerank_format="tei",
        )

    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        base_url="http://localhost:11434/v1",
        rerank_format="tei",
    )
    assert config.rerank_url == "/rerank"
    assert config.base_url == "http://localhost:11434/v1"

    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        rerank_url="http://localhost:8080/rerank",
        rerank_format="tei",
    )
    assert config.rerank_url == "http://localhost:8080/rerank"


@pytest.mark.parametrize(
    "rerank_format, rerank_model",
    [("cohere", "rerank-v3"), ("auto", "rerank-v3"), (None, "some-model")],
    ids=["cohere", "auto", "default-None"],
)
def test_rerank_model_with_base_url_auto_sets_rerank_url(
    rerank_format: str | None, rerank_model: str
) -> None:
    """rerank_model + base_url should imply the default rerank path."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    kwargs = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "base_url": "http://localhost:11434/v1",
        "rerank_model": rerank_model,
    }
    if rerank_format is not None:
        kwargs["rerank_format"] = rerank_format

    config = EmbeddingConfig(**kwargs)
    assert config.rerank_url == "/rerank"
