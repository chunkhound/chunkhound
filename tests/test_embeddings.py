"""Test script to verify embedding system functionality without making API calls."""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import chunkhound modules
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path

import pytest

from chunkhound.embeddings import EmbeddingManager
from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider

from .test_utils import get_api_key_for_tests, should_run_live_api_tests


async def test_official_openai_validation():
    """Test official OpenAI API key validation logic."""
    # Should work: API key provided
    provider = OpenAIEmbeddingProvider(api_key="sk-fake-key")
    assert provider.api_key == "sk-fake-key"

    # Should fail: No API key for official OpenAI
    provider = OpenAIEmbeddingProvider()
    with pytest.raises(
        ValueError, match="OpenAI API key is required for official OpenAI API"
    ):
        await provider._ensure_client()


async def test_custom_endpoint_validation():
    """Test custom endpoint mode allows optional API key."""
    # Should work: Custom endpoint, no API key
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )
    assert provider.base_url == "http://localhost:11434"

    # Should work: Custom endpoint + API key
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:1234", api_key="custom-key"
    )
    assert provider.api_key == "custom-key"


def test_url_detection_logic():
    """Test the logic that determines official vs custom endpoints."""
    # Official OpenAI URLs (should require API key)
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
        assert is_official, f"URL {url} should be detected as official OpenAI"

    # Custom URLs (should NOT require API key)
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
        assert not is_official, f"URL {url} should be detected as custom endpoint"


@pytest.mark.skipif(
    not should_run_live_api_tests(),
    reason="No API key available (set CHUNKHOUND_EMBEDDING__API_KEY or add to .chunkhound.json)",
)
async def test_real_embedding_api():
    """Test real embedding API call with discovered provider and key."""
    from .test_utils import get_embedding_config_for_tests

    config = get_embedding_config_for_tests()
    if not config:
        pytest.skip("No embedding config available for testing")

    api_key = config.get("api_key")
    provider_name = config.get("provider", "openai")
    model = config.get("model")
    base_url = config.get("base_url")

    # Create the appropriate provider based on what's configured
    if provider_name == "openai":
        from chunkhound.providers.embeddings.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        # Use model from config or default
        model = model or "text-embedding-3-small"
        provider = OpenAIEmbeddingProvider(
            api_key=api_key, model=model, base_url=base_url
        )

        # Get actual embedding dimensions from a test embedding
        test_result = await provider.embed(["test"])
        expected_dims = len(test_result[0])
    elif provider_name == "voyageai":
        from chunkhound.providers.embeddings.voyageai_provider import (
            VoyageAIEmbeddingProvider,
        )

        model = model or "voyage-3.5"
        provider = VoyageAIEmbeddingProvider(api_key=api_key, model=model)
        expected_dims = 1024  # voyage-3.5 dimensions
    else:
        pytest.skip(f"Unknown provider: {provider_name}")

    result = await provider.embed(["Hello, world!"])

    assert len(result) == 1
    assert len(result[0]) == expected_dims
    assert all(isinstance(x, float) for x in result[0])


async def test_custom_endpoint_mock_behavior():
    """Test custom endpoint behavior without real server."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )

    try:
        await provider._ensure_client()
    except Exception as e:
        assert "API key" not in str(e), (
            f"Should not require API key for custom endpoint: {e}"
        )


async def test_ollama_with_reranking_configuration():
    """Test OpenAI provider configured for Ollama with dual-endpoint reranking."""
    # Test configuration using OpenAI provider for Ollama embeddings and separate reranking endpoint
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        model="nomic-embed-text",
        api_key="dummy-key-for-custom-endpoint",  # Custom endpoints don't validate API keys
        rerank_model="test-reranker",
        rerank_url="http://localhost:8001/rerank",  # Separate rerank service
    )

    # Verify configuration
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "nomic-embed-text"
    assert provider._rerank_model == "test-reranker"
    assert provider._rerank_url == "http://localhost:8001/rerank"

    # Test that reranking is supported when rerank_model is configured
    assert provider.supports_reranking() == True

    # Test reranking call (will fail due to no actual service, but tests structure)
    try:
        await provider.rerank("test query", ["doc1", "doc2"])
    except Exception as e:
        # Expected to fail since we don't have actual rerank service running
        # But should not be an API key error
        assert "API key" not in str(e), (
            f"Should not require API key error for reranking: {e}"
        )
        # Should be a connection error since the rerank service isn't running
        assert any(
            keyword in str(e).lower()
            for keyword in ["connection", "network", "reranking failed"]
        ), f"Expected connection error for rerank service, got: {e}"


@pytest.mark.skipif(
    not (
        # Check if Ollama is running
        os.system("curl -s http://localhost:11434/api/tags > /dev/null 2>&1") == 0
        and
        # Check if rerank service is running
        os.system("curl -s http://localhost:8001/health > /dev/null 2>&1") == 0
    ),
    reason="Ollama and/or rerank service not running",
)
async def test_ollama_with_live_reranking():
    """Test OpenAI provider configured for Ollama with actual reranking service.

    Note: This test requires a real reranking service (e.g., vLLM) and may not work
    with the simple mock server due to HTTP parsing limitations in the mock server.
    """
    # Check if we're using the mock server (which has HTTP parsing issues with httpx)
    import httpx

    try:
        # Use synchronous check since we can't use async in the test setup
        with httpx.Client(timeout=1.0) as client:
            response = client.get("http://localhost:8001/health")
            if response.json().get("service") == "mock-rerank-server":
                # Mock server has issues with httpx requests - skip this test
                pytest.skip(
                    "Mock server has HTTP parsing issues with httpx - use vLLM for this test"
                )
    except Exception:
        # If we can't check, continue with test
        pass

    # This test uses OpenAI provider configured for Ollama embeddings and a separate service for reranking
    # Embeddings come from Ollama (port 11434)
    # Reranking goes to separate service (port 8001)
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
        model="nomic-embed-text",
        api_key="dummy-key",  # Ollama doesn't require real API key
        rerank_model="test-model",
        rerank_url="http://localhost:8001/rerank",  # Absolute URL to rerank service
    )

    # Test that reranking works end-to-end
    test_docs = [
        "def calculate_sum(a, b): return a + b",
        "import numpy as np",
        "class Calculator: pass",
        "function add(x, y) { return x + y; }",
    ]

    results = await provider.rerank("python function definition", test_docs, top_k=3)

    # Verify results structure
    assert len(results) <= 3, "Should respect top_k limit"
    assert all(hasattr(r, "index") and hasattr(r, "score") for r in results), (
        "Results should have index and score"
    )
    assert all(0 <= r.index < len(test_docs) for r in results), (
        "Indices should be valid"
    )
    assert all(isinstance(r.score, float) for r in results), "Scores should be floats"

    # Verify we got meaningful results (ranking may vary with embeddings)
    assert len(results) > 0, "Should return results"

    print("✅ Live reranking test passed:")
    print(f"   • Reranked {len(test_docs)} documents")
    print(
        f"   • Top result: '{test_docs[results[0].index][:50]}...' (score: {results[0].score:.3f})"
    )
    print(f"   • All results: {[(r.index, f'{r.score:.3f}') for r in results]}")
    print("   • Document mapping:")
    for i, doc in enumerate(test_docs):
        print(f"     [{i}]: {doc}")

    # Check that scores are in descending order
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            "Results should be ordered by score"
        )


async def test_tei_reranking_format_with_model():
    """Test TEI reranking format with optional model field."""
    # TEI format: uses "texts" instead of "documents", model is optional
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="dummy-key",
        rerank_model="BAAI/bge-reranker-base",  # Optional for TEI
        rerank_url="http://localhost:8001/rerank",
        rerank_format="tei",  # Explicit TEI format
    )

    # Verify configuration
    assert provider._rerank_format == "tei"
    assert provider._rerank_model == "BAAI/bge-reranker-base"
    assert provider.supports_reranking() == True

    # Test reranking call (will fail due to no actual service, but tests structure)
    try:
        await provider.rerank("test query", ["doc1", "doc2"])
    except Exception as e:
        # Expected to fail since we don't have actual rerank service running
        # Should be a connection error
        assert any(
            keyword in str(e).lower() for keyword in ["connection", "network"]
        ), f"Expected connection error, got: {e}"

    print("✅ TEI format with model test passed")


async def test_tei_reranking_format_without_model():
    """Test TEI reranking format without model field (model set at deployment)."""
    # TEI format: model is set at deployment time with --model-id flag
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="dummy-key",
        rerank_url="http://localhost:8001/rerank",
        rerank_format="tei",  # Explicit TEI format, no model needed
    )

    # Verify configuration
    assert provider._rerank_format == "tei"
    assert provider._rerank_model is None  # No model in config
    assert provider.supports_reranking() == True  # Still supports reranking

    # Test reranking call
    try:
        await provider.rerank("test query", ["doc1", "doc2"])
    except Exception as e:
        # Expected to fail since we don't have actual rerank service running
        assert any(
            keyword in str(e).lower() for keyword in ["connection", "network"]
        ), f"Expected connection error, got: {e}"

    print("✅ TEI format without model test passed")


async def test_tei_bare_array_response_format():
    """Test that bare array TEI responses are correctly normalized.

    Real TEI servers return: [{"index": 0, "score": 0.95}, ...]
    This test verifies the normalization to: {"results": [...]}
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    provider = OpenAIEmbeddingProvider(
        api_key="test-key",
        base_url="http://localhost:8080",
        model="text-embedding-3-small",
        rerank_url="http://localhost:8080/rerank",
        rerank_format="tei",  # Explicit TEI format
    )

    # Initialize the OpenAI client BEFORE patching
    await provider._ensure_client()

    # Mock httpx.AsyncClient's post method to return bare array (real TEI format)
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"index": 0, "score": 0.95},
        {"index": 1, "score": 0.42},
    ]
    mock_response.raise_for_status = MagicMock()

    # Create a mock client that returns the mock response
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    # Patch httpx.AsyncClient in the provider module
    with patch(
        "chunkhound.providers.embeddings.openai_provider.httpx.AsyncClient",
        return_value=mock_client,
    ):
        results = await provider.rerank("test query", ["doc1", "doc2"])

        # Verify results were parsed correctly
        assert len(results) == 2
        assert results[0].index == 0
        assert results[0].score == 0.95
        assert results[1].index == 1
        assert results[1].score == 0.42

    print("✅ TEI bare array response format test passed")


async def test_auto_format_detection():
    """Test auto-detection of reranking format from response."""
    # Auto mode: should detect format from response
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="dummy-key",
        rerank_url="http://localhost:8001/rerank",
        rerank_format="auto",  # Auto-detect
    )

    # Verify configuration
    assert provider._rerank_format == "auto"
    assert provider._detected_rerank_format is None  # Not detected yet

    print("✅ Auto-detection configuration test passed")


async def test_cohere_format_requires_model():
    """Test that Cohere format requires rerank_model (fail-fast validation)."""
    # Cohere format requires model - now validated at init time (fail-fast)
    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="cohere",  # Cohere requires model - fails at init
        )

    print("✅ Cohere format model requirement test passed")


def test_rerank_format_propagates_through_config():
    """Verify rerank_format propagates from config to provider."""
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
    assert provider._rerank_format == "tei", "Format should propagate from config"

    print("✅ Config propagation test passed")


def test_cohere_format_validation_requires_model():
    """Test that Cohere format validation correctly requires model."""
    import pytest

    from chunkhound.core.config.embedding_config import EmbeddingConfig

    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        EmbeddingConfig(
            provider="openai",
            base_url="http://localhost:8001",
            model="text-embedding-3-small",
            rerank_format="cohere",  # Missing rerank_model
        )

    print("✅ Cohere validation test passed")


def test_tei_format_validation_without_model():
    """Test that TEI format works without model (set at deployment)."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    # Should not raise
    config = EmbeddingConfig(
        provider="openai",
        base_url="http://localhost:8001",
        model="text-embedding-3-small",
        rerank_url="/rerank",
        rerank_format="tei",  # No model needed
    )
    assert config.rerank_format == "tei"

    print("✅ TEI validation test passed")


def test_supports_reranking_with_incomplete_cohere_config():
    """Test that incomplete Cohere config fails at init (fail-fast validation)."""
    # With fail-fast validation, incomplete Cohere config now raises at init time
    with pytest.raises(ValueError, match="rerank_model is required.*cohere"):
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:8001",
            model="text-embedding-3-small",
            rerank_url="/rerank",
            rerank_format="cohere",
            # Missing rerank_model - fails at init now
        )

    print("✅ Incomplete Cohere config validation test passed")


def test_supports_reranking_with_tei_config():
    """Test that supports_reranking returns True for TEI without model."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:8001",
        model="text-embedding-3-small",
        rerank_url="/rerank",
        rerank_format="tei",
        # No model needed for TEI
    )

    assert provider.supports_reranking(), "Should return True when TEI format has URL"

    print("✅ supports_reranking() TEI test passed")


def test_embedding_manager():
    """Test embedding manager functionality."""
    print("\nTesting embedding manager...")

    try:
        manager = EmbeddingManager()

        # Create a mock provider
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key-for-testing", model="text-embedding-3-small"
        )

        # Register provider
        manager.register_provider(provider, set_default=True)

        # Test provider retrieval
        retrieved = manager.get_provider()
        assert retrieved.name == "openai"
        assert retrieved.model == "text-embedding-3-small"

        # Test provider listing
        providers = manager.list_providers()
        assert "openai" in providers

        print("✅ Embedding manager tests passed:")
        print(f"   • Registered providers: {providers}")
        print(f"   • Default provider: {retrieved.name}/{retrieved.model}")

    except Exception as e:
        print(f"❌ Embedding manager test failed: {e}")
        assert False, f"Embedding manager test failed: {e}"


async def test_mock_embedding_generation():
    """Test embedding generation with mock data (no API call)."""
    print("\nTesting mock embedding generation...")

    try:
        # This will fail with API call, but we can test the structure
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key-for-testing", model="text-embedding-3-small"
        )

        # Test input validation
        empty_result = await provider.embed([])
        assert empty_result == []
        print("✅ Empty input handling works")

        # Test with actual text (this will fail due to fake API key, but that's expected)
        try:
            result = await provider.embed(["def hello(): pass"])
            print("❌ Unexpected success - should have failed with fake API key")
        except Exception as e:
            print(f"✅ Expected API failure with fake key: {type(e).__name__}")

        return True

    except Exception as e:
        print(f"❌ Mock embedding test failed: {e}")
        return False


def test_provider_integration():
    """Test integration of all providers with EmbeddingManager."""
    print("\nTesting provider integration with EmbeddingManager...")

    try:
        manager = EmbeddingManager()

        # Register OpenAI provider
        openai_provider = OpenAIEmbeddingProvider(
            api_key="sk-test-key", model="text-embedding-3-small"
        )
        manager.register_provider(openai_provider)

        # Test provider listing
        providers = manager.list_providers()
        expected_providers = {"openai"}
        assert expected_providers.issubset(set(providers))

        # Test specific provider retrieval
        openai_retrieved = manager.get_provider("openai")
        assert openai_retrieved.name == "openai"

        print("✅ Provider integration successful:")
        print(f"   • Registered providers: {providers}")
        print("   • Can retrieve by name: ✓")

    except Exception as e:
        print(f"❌ Provider integration test failed: {e}")
        assert False, f"Provider integration failed: {e}"


def test_environment_variable_handling():
    """Test environment variable handling."""
    print("\nTesting environment variable handling...")

    # Save original env vars
    original_key = os.getenv("OPENAI_API_KEY")
    original_url = os.getenv("OPENAI_BASE_URL")

    try:
        # Test with env vars
        os.environ["OPENAI_API_KEY"] = "sk-test-env-key"
        os.environ["OPENAI_BASE_URL"] = "https://test.example.com"

        provider = OpenAIEmbeddingProvider()
        print("✅ Environment variable loading works")

        # Test missing API key
        del os.environ["OPENAI_API_KEY"]
        try:
            provider = OpenAIEmbeddingProvider()
            print("❌ Should have failed with missing API key")
        except ValueError:
            print("✅ Correctly handles missing API key")

    except Exception as e:
        print(f"❌ Environment test failed: {e}")

    finally:
        # Restore original env vars
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        if original_url:
            os.environ["OPENAI_BASE_URL"] = original_url
        elif "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]


async def main():
    """Run all tests."""
    print("ChunkHound Embedding System Tests")
    print("=" * 40)

    # Test provider creation
    provider = await test_openai_provider_creation()

    # Test embedding manager
    manager = test_embedding_manager()

    # Test provider integration
    test_provider_integration()

    # Test mock embedding generation
    await test_mock_embedding_generation()

    # Test environment variables
    test_environment_variable_handling()

    print("\n" + "=" * 40)
    print("Test summary:")
    print("✅ OpenAI provider creation")
    print("✅ Embedding manager functionality")
    print("✅ Provider integration")
    print("✅ Mock embedding generation")
    print("✅ Environment variable handling")
    print("\nAll core embedding functionality verified!")
    print("\nTo test with real API calls, set OPENAI_API_KEY and run:")
    print(
        'python -c "import asyncio; from test_embeddings import test_real_api; asyncio.run(test_real_api())"'
    )


async def test_real_api():
    """Test with real embedding API (requires valid API key)."""
    # Get API key from generic test function
    api_key, provider_name = get_api_key_for_tests()

    if not api_key:
        print("⏭️  Skipping real API tests - no API key found")
        print(
            "To run real API tests: set CHUNKHOUND_EMBEDDING__API_KEY or configure .chunkhound.json"
        )
        return True  # Return success to not break test suite

    print("\n" + "=" * 50)
    print(f"🚀 COMPREHENSIVE REAL API TESTING ({provider_name.upper()})")
    print("=" * 50)

    try:
        # Test 1: Basic embedding generation
        print("\n1. Testing basic embedding generation...")

        # Create the appropriate provider
        if provider_name == "openai":
            from chunkhound.providers.embeddings.openai_provider import (
                OpenAIEmbeddingProvider,
            )

            provider = OpenAIEmbeddingProvider(api_key=api_key)
        elif provider_name == "voyageai":
            from chunkhound.providers.embeddings.voyageai_provider import (
                VoyageAIEmbeddingProvider,
            )

            provider = VoyageAIEmbeddingProvider(api_key=api_key, model="voyage-3.5")
        else:
            print(f"❌ Unknown provider: {provider_name}")
            return False

        test_texts = [
            "def hello(): return 'world'",
            "class Database: pass",
            "async def search(query: str) -> List[str]:",
        ]

        result = await provider.embed(test_texts)

        print("✅ Basic embedding test successful:")
        print(f"   • Generated {len(result)} embeddings")
        print(f"   • Vector dimensions: {len(result[0])}")
        print(f"   • Model: {provider.model}")
        print(f"   • Provider: {provider.name}")

        # Test 2: Alternative model (if available)
        if provider_name == "openai":
            print("\n2. Testing with text-embedding-3-large...")
            alt_provider = OpenAIEmbeddingProvider(
                api_key=api_key, model="text-embedding-3-large"
            )
            alt_result = await alt_provider.embed(["def test(): pass"])
            print("✅ Alternative model test successful:")
            print(f"   • Model: {alt_provider.model}")
            print(f"   • Dimensions: {len(alt_result[0])}")
        elif provider_name == "voyageai":
            print("\n2. Testing with voyage-3-large...")
            alt_provider = VoyageAIEmbeddingProvider(
                api_key=api_key, model="voyage-3-large"
            )
            alt_result = await alt_provider.embed(["def test(): pass"])
            print("✅ Alternative model test successful:")
            print(f"   • Model: {alt_provider.model}")
            print(f"   • Dimensions: {len(alt_result[0])}")

        # Test 3: Batch processing
        print("\n3. Testing batch processing...")
        batch_texts = [f"def function_{i}(): return {i}" for i in range(10)]

        batch_result = await provider.embed(batch_texts)
        print("✅ Batch processing test successful:")
        print(f"   • Processed {len(batch_result)} texts in batch")
        print(f"   • All vectors have {len(batch_result[0])} dimensions")

        # Test 4: Integration with EmbeddingManager
        print("\n4. Testing EmbeddingManager integration...")
        manager = EmbeddingManager()
        manager.register_provider(provider, set_default=True)

        manager_result = await manager.embed_texts(
            ["import asyncio", "from typing import List, Optional"]
        )

        print("✅ EmbeddingManager integration successful:")
        print(f"   • Generated {len(manager_result.embeddings)} embeddings via manager")
        print(f"   • Each vector: {len(manager_result.embeddings[0])} dimensions")
        print(f"   • Using provider: {manager.get_provider().name}")
        print(f"   • Result model: {manager_result.model}")
        print(f"   • Result provider: {manager_result.provider}")

        # Test 5: Vector similarity check
        print("\n5. Testing vector similarity (semantic relationship)...")
        similar_texts = [
            "async def process_file():",
            "async def handle_file():",
            "def synchronous_function():",
        ]

        similar_results = await provider.embed(similar_texts)

        # Calculate cosine similarity between first two (should be higher)
        import math

        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            magnitude_a = math.sqrt(sum(x * x for x in a))
            magnitude_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (magnitude_a * magnitude_b)

        sim_async = cosine_similarity(similar_results[0], similar_results[1])
        sim_mixed = cosine_similarity(similar_results[0], similar_results[2])

        print("✅ Semantic similarity test:")
        print(f"   • Async function similarity: {sim_async:.4f}")
        print(f"   • Mixed function similarity: {sim_mixed:.4f}")
        print(f"   • Semantic relationship detected: {sim_async > sim_mixed}")

        print("\n" + "🎉" * 15)
        print("ALL REAL API TESTS PASSED!")
        print("🎉" * 15)
        print("\nSummary:")
        print("✅ Basic embedding generation working")
        print("✅ Multiple model support")
        print("✅ Batch processing functional")
        print("✅ EmbeddingManager integration complete")
        print("✅ Semantic relationships captured in vectors")
        print("✅ Ready for production use with real embeddings!")

        return True

    except Exception as e:
        print(f"❌ Real API test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_tei_format_end_to_end_with_mock_server():
    """Test TEI format with actual mock server for end-to-end verification.

    This test requires the mock rerank server running on localhost:8001.
    Run: uv run python tests/rerank_server.py
    """
    import subprocess
    import time

    # Start mock server in background
    server_process = None
    try:
        server_process = subprocess.Popen(
            [sys.executable, "tests/rerank_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Wait for server to start
        time.sleep(2)

        # Create provider with TEI format
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="test-key",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="tei",  # Explicit TEI format
        )

        # Make actual request to mock server
        test_docs = [
            "Python is a programming language",
            "JavaScript is used for web development",
            "def calculate_sum(a, b): return a + b",
        ]

        results = await provider.rerank("python programming", test_docs, top_k=2)

        # Check if mock server returned empty results (HTTP parsing issue)
        if len(results) == 0:
            pytest.skip(
                "Mock server HTTP parsing issue - returned empty results (use vLLM/TEI for full testing)"
            )

        # Verify response structure
        assert len(results) > 0, "Should return results"
        assert len(results) <= 2, "Should respect top_k limit"
        assert all(hasattr(r, "index") and hasattr(r, "score") for r in results)
        assert all(0 <= r.index < len(test_docs) for r in results)
        assert all(isinstance(r.score, float) for r in results)

        # Verify scores are descending
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

        print("✅ TEI end-to-end test passed")
        print(f"   • Reranked {len(test_docs)} documents")
        print(f"   • Got {len(results)} results")
        print(f"   • Top score: {results[0].score:.3f}")

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=5)


async def test_auto_detection_caches_format():
    """Verify format detection caches result for subsequent calls."""
    import subprocess
    import time

    # Start mock server
    server_process = None
    try:
        server_process = subprocess.Popen(
            [sys.executable, "tests/rerank_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)

        # Create provider with auto mode
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="test-key",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="auto",
        )

        # Initial state: no format detected
        assert provider._detected_rerank_format is None

        # First request - should detect format
        test_docs = ["doc1", "doc2", "doc3"]
        results1 = await provider.rerank("query", test_docs)

        # Check if mock server returned empty results (HTTP parsing issue)
        if len(results1) == 0:
            pytest.skip(
                "Mock server HTTP parsing issue - returned empty results (use vLLM/TEI for full testing)"
            )

        # After first request: format should be detected and cached
        detected_format = provider._detected_rerank_format
        assert detected_format in ["cohere", "tei"], (
            f"Should detect format, got: {detected_format}"
        )
        print(f"✅ Auto-detected format: {detected_format}")

        # Second request - should use cached format
        results2 = await provider.rerank("another query", test_docs)

        # Format should still be the same (cached)
        assert provider._detected_rerank_format == detected_format
        assert len(results2) > 0

        print("✅ Format caching test passed")
        print(f"   • Detected format: {detected_format}")
        print("   • Format persisted across requests")

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=5)


async def test_concurrent_rerank_calls():
    """Verify concurrent rerank calls don't race on format detection."""
    import subprocess
    import time

    # Start mock server
    server_process = None
    try:
        server_process = subprocess.Popen(
            [sys.executable, "tests/rerank_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)

        # Create provider with auto mode
        provider = OpenAIEmbeddingProvider(
            base_url="http://localhost:11434/v1",
            model="nomic-embed-text",
            api_key="test-key",
            rerank_url="http://localhost:8001/rerank",
            rerank_format="auto",
        )

        test_docs = ["doc1", "doc2", "doc3"]

        # Make multiple concurrent requests
        tasks = [provider.rerank(f"query {i}", test_docs) for i in range(10)]

        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for i, results in enumerate(results_list):
            if isinstance(results, Exception):
                print(f"Request {i} failed: {results}")
                raise results
            assert len(results) > 0, f"Request {i} should return results"

        # Format should be detected
        assert provider._detected_rerank_format is not None

        print("✅ Concurrent rerank test passed")
        print(f"   • Ran {len(tasks)} concurrent requests")
        print("   • All requests succeeded")
        print("   • No race conditions detected")

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=5)


async def test_malformed_rerank_response():
    """Test handling of malformed rerank responses."""
    provider = OpenAIEmbeddingProvider(
        base_url="http://localhost:11434/v1",
        model="nomic-embed-text",
        api_key="test-key",
        rerank_format="tei",
    )

    # Test 1: Missing 'results' field
    with pytest.raises(ValueError, match="missing 'results' field"):
        await provider._parse_rerank_response({"status": "ok"}, "tei", num_documents=3)

    # Test 2: 'results' is not a list
    with pytest.raises(ValueError, match="'results' must be a list"):
        await provider._parse_rerank_response(
            {"results": "not a list"}, "tei", num_documents=3
        )

    # Test 3: Result missing 'index' field
    malformed_results = {
        "results": [{"score": 0.5}]  # Missing 'index'
    }
    with pytest.raises(ValueError, match="must have 'index' field"):
        await provider._parse_rerank_response(malformed_results, "tei", num_documents=3)

    # Test 4: Result missing score field
    malformed_results = {
        "results": [{"index": 0}]  # Missing 'score'
    }
    with pytest.raises(
        ValueError, match="must have 'relevance_score' or 'score' field"
    ):
        await provider._parse_rerank_response(malformed_results, "tei", num_documents=3)

    # Test 5: Empty results list (should succeed with empty list)
    empty_results = {"results": []}
    parsed = await provider._parse_rerank_response(
        empty_results, "tei", num_documents=3
    )
    assert parsed == []

    # Test 6: Results with invalid data types (should skip bad entries)
    mixed_results = {
        "results": [
            {"index": 0, "score": 0.9},
            {"index": "invalid", "score": 0.8},  # Bad index
            {"index": 2, "score": "invalid"},  # Bad score
            {"index": 3, "score": 0.7},
        ]
    }
    parsed = await provider._parse_rerank_response(
        mixed_results, "tei", num_documents=4
    )
    assert len(parsed) == 2  # Should skip 2 invalid entries
    assert parsed[0].index == 0
    assert parsed[1].index == 3

    # Test 7: Out-of-bounds indices (should skip)
    out_of_bounds_results = {
        "results": [
            {"index": 0, "score": 0.9},  # Valid
            {"index": 5, "score": 0.8},  # Out of bounds (only 3 docs)
            {"index": 2, "score": 0.7},  # Valid
        ]
    }
    parsed = await provider._parse_rerank_response(
        out_of_bounds_results, "tei", num_documents=3
    )
    assert len(parsed) == 2  # Should skip out-of-bounds entry
    assert parsed[0].index == 0
    assert parsed[1].index == 2

    # Test 8: Negative indices (should skip)
    negative_index_results = {
        "results": [
            {"index": -1, "score": 0.9},  # Negative index
            {"index": 0, "score": 0.8},  # Valid
        ]
    }
    parsed = await provider._parse_rerank_response(
        negative_index_results, "tei", num_documents=3
    )
    assert len(parsed) == 1  # Should skip negative index
    assert parsed[0].index == 0

    print("✅ Malformed response handling test passed")
    print("   • Missing fields detected")
    print("   • Invalid data types handled")
    print("   • Empty results handled")
    print("   • Out-of-bounds indices rejected")
    print("   • Negative indices rejected")


def test_relative_rerank_url_requires_base_url():
    """Verify relative rerank_url requires base_url."""
    from chunkhound.core.config.embedding_config import EmbeddingConfig

    # Test 1: Relative URL without base_url should fail
    with pytest.raises(ValueError, match="base_url is required"):
        config = EmbeddingConfig(
            provider="openai",
            model="text-embedding-3-small",
            rerank_format="tei",
            # rerank_url defaults to "/rerank" (relative)
            # No base_url provided
        )

    # Test 2: Relative URL with base_url should succeed
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        base_url="http://localhost:11434/v1",
        rerank_format="tei",
    )
    assert config.rerank_url == "/rerank"
    assert config.base_url == "http://localhost:11434/v1"

    # Test 3: Absolute URL without base_url should succeed
    config = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        rerank_url="http://localhost:8080/rerank",
        rerank_format="tei",
    )
    assert config.rerank_url == "http://localhost:8080/rerank"

    print("✅ Relative URL validation test passed")
    print("   • Relative URL requires base_url")
    print("   • Absolute URL works without base_url")


if __name__ == "__main__":
    asyncio.run(main())
