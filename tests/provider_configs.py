"""
Provider configuration discovery for multi-provider testing.

This module discovers available embedding providers that support reranking
and provides their configurations for parametrized testing.
"""

from typing import Any

import httpx

from .test_utils import get_api_key_for_tests


def services_available(auto_start_rerank: bool = True) -> bool:
    """
    Check if both Ollama and rerank service are available for testing.

    Args:
        auto_start_rerank: If True, will attempt to start mock rerank server if not running

    Returns:
        True if both services are available, False otherwise
    """
    # Check Ollama embeddings service
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        ollama_ok = response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        ollama_ok = False

    if not ollama_ok:
        return False  # Need Ollama for embeddings

    # Check rerank service
    try:
        response = httpx.get("http://localhost:8001/health", timeout=1.0)
        rerank_ok = response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        rerank_ok = False

    # If rerank not running and auto-start enabled, try to start mock server
    if not rerank_ok and auto_start_rerank:
        import asyncio

        from tests.fixtures.rerank_server_manager import ensure_rerank_server_running

        async def try_start():
            manager = await ensure_rerank_server_running(start_if_needed=True)
            return (
                manager is not None or await manager.is_running() if manager else False
            )

        try:
            # Try to start mock server
            rerank_ok = asyncio.run(try_start())
        except Exception:
            rerank_ok = False

    return ollama_ok and rerank_ok


def get_reranking_providers() -> list[tuple[str, type, dict[str, Any]]]:
    """
    Discover available providers that support reranking.

    Returns:
        List of (provider_name, provider_class, config_dict) tuples for
        providers that are available and support reranking functionality.
    """
    providers = []

    # Use the existing Config class to load complete configuration from .chunkhound.json
    from chunkhound.core.config.config import Config

    try:
        # This automatically loads .chunkhound.json with full config including rerank_model
        config = Config()

        if config.embedding and config.embedding.api_key:
            # Use existing get_provider_config() to get complete dict with rerank_model
            provider_config = config.embedding.get_provider_config()

            # Get the provider class based on configured provider
            if config.embedding.provider == "voyageai":
                from chunkhound.providers.embeddings.voyageai_provider import (
                    VoyageAIEmbeddingProvider,
                )

                # Filter to only parameters VoyageAI accepts
                voyageai_config = {
                    "api_key": provider_config.get("api_key"),
                    "model": provider_config.get("model"),
                    "rerank_model": provider_config.get("rerank_model"),
                    "batch_size": provider_config.get("batch_size", 100),
                    "timeout": provider_config.get("timeout", 30),
                    "retry_attempts": provider_config.get(
                        "max_retries", 3
                    ),  # Name translation
                }
                # Remove None values
                voyageai_config = {
                    k: v for k, v in voyageai_config.items() if v is not None
                }
                providers.append(
                    ("voyageai", VoyageAIEmbeddingProvider, voyageai_config)
                )
            elif config.embedding.provider == "openai" and services_available(
                auto_start_rerank=False
            ):
                from chunkhound.providers.embeddings.openai_provider import (
                    OpenAIEmbeddingProvider,
                )

                # Filter to only parameters OpenAI provider accepts
                openai_config = {
                    "api_key": provider_config.get("api_key"),
                    "base_url": provider_config.get("base_url"),
                    "model": provider_config.get("model"),
                    "rerank_model": provider_config.get("rerank_model"),
                    "rerank_url": provider_config.get("rerank_url"),
                    "batch_size": provider_config.get("batch_size", 100),
                    "timeout": provider_config.get("timeout", 30),
                    "max_retries": provider_config.get("max_retries", 3),
                }
                # Remove None values
                openai_config = {
                    k: v for k, v in openai_config.items() if v is not None
                }
                providers.append(("openai", OpenAIEmbeddingProvider, openai_config))
    except (FileNotFoundError, ImportError, SystemExit):
        # Fall back to legacy approach if config file missing or imports fail
        pass

    # Fallback: Check for providers using legacy method only if no providers found
    if not providers:
        api_key, provider_name = get_api_key_for_tests()
        if provider_name == "voyageai" and api_key:
            from chunkhound.providers.embeddings.voyageai_provider import (
                VoyageAIEmbeddingProvider,
            )

            providers.append(
                (
                    "voyageai",
                    VoyageAIEmbeddingProvider,
                    {
                        "api_key": api_key,
                        "model": "voyage-3.5",
                        "batch_size": 100,
                        "timeout": 30,
                        "retry_attempts": 3,
                    },
                )
            )

        # Check for Ollama with reranking service (only as last resort)
        if services_available(auto_start_rerank=False):
            from chunkhound.providers.embeddings.openai_provider import (
                OpenAIEmbeddingProvider,
            )

            providers.append(
                (
                    "openai",
                    OpenAIEmbeddingProvider,
                    {
                        "api_key": "dummy-key",
                        "base_url": "http://localhost:11434/v1",
                        "model": "nomic-embed-text",
                        "rerank_model": "test-model",
                        "rerank_url": "http://localhost:8001/rerank",
                        "batch_size": 100,
                        "timeout": 30,
                        "retry_attempts": 3,
                    },
                )
            )

    return providers


def get_provider_ids() -> list[str]:
    """Get list of available provider IDs for pytest parametrization."""
    return [provider_name for provider_name, _, _ in get_reranking_providers()]


def should_skip_if_no_providers() -> bool:
    """Check if tests should be skipped due to no available providers."""
    return len(get_reranking_providers()) == 0
