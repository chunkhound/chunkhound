"""
Unified embedding provider factory for ChunkHound.

This module provides a factory pattern for creating embedding providers
with consistent configuration across all ChunkHound execution modes.
The factory supports all four embedding providers with unified configuration.
"""

from typing import TYPE_CHECKING, Any

from loguru import logger

from .embedding_config import EmbeddingConfig

if TYPE_CHECKING:
    from chunkhound.embeddings import (
        EmbeddingProvider,
        OpenAIEmbeddingProvider,
    )


class EmbeddingProviderFactory:
    """
    Factory for creating embedding providers from unified configuration.

    This factory provides consistent provider creation across MCP server
    and indexing flows, supporting all four embedding providers with
    type-safe configuration validation.
    """

    @staticmethod
    def create_provider(config: EmbeddingConfig) -> "EmbeddingProvider":
        """
        Create an embedding provider from configuration.

        Args:
            config: Validated embedding configuration

        Returns:
            Configured embedding provider instance

        Raises:
            ValueError: If provider configuration is invalid or incomplete
            ImportError: If required dependencies are not available
        """
        # Validate configuration completeness
        if not config.is_provider_configured():
            missing = config.get_missing_config()
            raise ValueError(
                f"Incomplete configuration for {config.provider} provider. "
                f"Missing: {', '.join(missing)}"
            )

        # Get provider-specific configuration
        provider_config = config.get_provider_config()

        # Create provider based on type
        if config.provider == "openai":
            return EmbeddingProviderFactory._create_openai_provider(provider_config)
        elif config.provider == "voyageai":
            return EmbeddingProviderFactory._create_voyageai_provider(provider_config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    @staticmethod
    def _create_openai_provider(config: dict[str, Any]) -> "OpenAIEmbeddingProvider":
        """Create OpenAI embedding provider."""
        try:
            from chunkhound.embeddings import create_openai_provider
        except ImportError as e:
            raise ImportError(
                "Failed to import OpenAI provider. "
                "Ensure chunkhound.embeddings module is available."
            ) from e

        # Extract OpenAI-specific parameters
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("model")
        rerank_model = config.get("rerank_model")
        rerank_url = config.get("rerank_url", "/rerank")
        rerank_format = config.get("rerank_format", "auto")
        rerank_batch_size = config.get("rerank_batch_size")
        ssl_verify = config.get("ssl_verify", True)
        rerank_ssl_verify = config.get("rerank_ssl_verify")

        # Azure OpenAI parameters
        api_version = config.get("api_version")
        azure_endpoint = config.get("azure_endpoint")
        azure_deployment = config.get("azure_deployment")

        # Model should come from config, but handle None case safely
        if not model:
            raise ValueError("Model not specified in provider configuration")

        # Log Azure configuration if present
        if azure_endpoint:
            logger.debug(
                f"Creating Azure OpenAI provider: model={model}, "
                f"azure_endpoint={azure_endpoint}, api_version={api_version}, "
                f"azure_deployment={azure_deployment}, "
                f"api_key={'***' if api_key else None}, "
                f"rerank_model={rerank_model}, rerank_format={rerank_format}, "
                f"rerank_batch_size={rerank_batch_size}"
            )
        else:
            logger.debug(
                f"Creating OpenAI provider: model={model}, "
                f"base_url={base_url}, api_key={'***' if api_key else None}, "
                f"rerank_model={rerank_model}, rerank_format={rerank_format}, "
                f"rerank_batch_size={rerank_batch_size}"
            )

        try:
            return create_openai_provider(
                api_key=api_key,
                base_url=base_url,
                model=model,
                rerank_model=rerank_model,
                rerank_url=rerank_url,
                rerank_format=rerank_format,
                rerank_batch_size=rerank_batch_size,
                ssl_verify=ssl_verify,
                rerank_ssl_verify=rerank_ssl_verify,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment,
            )
        except Exception as e:
            raise ValueError(f"Failed to create OpenAI provider: {e}") from e

    @staticmethod
    def _create_voyageai_provider(config: dict[str, Any]) -> "EmbeddingProvider":
        """Create VoyageAI embedding provider."""
        try:
            from chunkhound.providers.embeddings.voyageai_provider import (
                VoyageAIEmbeddingProvider,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import VoyageAI provider. "
                "Ensure voyageai package is installed: uv pip install voyageai"
            ) from e

        # Extract VoyageAI-specific parameters
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        model = config.get("model")
        rerank_model = config.get("rerank_model")
        rerank_batch_size = config.get("rerank_batch_size")
        rerank_url = config.get("rerank_url")
        rerank_format = config.get("rerank_format", "auto")
        max_concurrent_batches = config.get("max_concurrent_batches")
        ssl_verify = config.get("ssl_verify", True)
        rerank_ssl_verify = config.get("rerank_ssl_verify")

        # Model should come from config, but handle None case safely
        if not model:
            raise ValueError("Model not specified in provider configuration")

        logger.debug(
            f"Creating VoyageAI provider: model={model}, "
            f"base_url={base_url}, api_key={'***' if api_key else None}, "
            f"rerank_model={rerank_model}, rerank_url={rerank_url}, "
            f"rerank_format={rerank_format}, rerank_batch_size={rerank_batch_size}"
        )

        try:
            # Build kwargs, only including rerank params if explicitly set
            # to allow provider constructor defaults to be used
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "model": model,
                "batch_size": config.get("batch_size", 100),
                "timeout": config.get("timeout", 30),
                "retry_attempts": config.get("max_retries", 3),
                "ssl_verify": ssl_verify,
            }
            if base_url is not None:
                kwargs["base_url"] = base_url
            if rerank_model is not None:
                kwargs["rerank_model"] = rerank_model
            if rerank_batch_size is not None:
                kwargs["rerank_batch_size"] = rerank_batch_size
            # rerank_url: resolve relative paths against base_url, then forward absolute URLs only
            if (
                rerank_url
                and base_url
                and not rerank_url.startswith(("http://", "https://"))
            ):
                from urllib.parse import urljoin

                rerank_url = urljoin(base_url.rstrip("/") + "/", rerank_url.lstrip("/"))
            if rerank_url and rerank_url.startswith(("http://", "https://")):
                kwargs["rerank_url"] = rerank_url
                kwargs["rerank_format"] = rerank_format
                if rerank_ssl_verify is not None:
                    kwargs["rerank_ssl_verify"] = rerank_ssl_verify
            if max_concurrent_batches is not None:
                kwargs["max_concurrent_batches"] = max_concurrent_batches

            return VoyageAIEmbeddingProvider(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create VoyageAI provider: {e}") from e

    @staticmethod
    def get_supported_providers() -> list[str]:
        """
        Get list of supported embedding providers.

        Returns:
            List of supported provider names
        """
        return ["openai", "voyageai", "openai_compatible"]

    @staticmethod
    def validate_provider_dependencies(provider: str) -> tuple[bool, str | None]:
        """
        Validate that dependencies for a provider are available.

        Args:
            provider: Provider name to validate

        Returns:
            Tuple of (is_available, error_message)
        """
        if provider not in EmbeddingProviderFactory.get_supported_providers():
            return False, f"Unsupported provider: {provider}"

        # Try to import the required create function
        try:
            if provider == "openai":
                from chunkhound.embeddings import create_openai_provider  # noqa: F401
            elif provider == "voyageai":
                from chunkhound.providers.embeddings.voyageai_provider import (  # noqa: F401
                    VoyageAIEmbeddingProvider,
                )

            return True, None

        except ImportError as e:
            return False, f"Missing dependencies for {provider} provider: {e}"

    @staticmethod
    def create_provider_from_legacy_args(
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ) -> "EmbeddingProvider":
        """
        Create provider from legacy CLI-style arguments.

        This method provides backward compatibility for existing code
        that uses the old argument-based provider creation.

        Args:
            provider: Provider name
            model: Model name
            api_key: API key
            base_url: Base URL
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured embedding provider

        Raises:
            ValueError: If configuration is invalid
        """
        # Create configuration from arguments
        config_dict = {
            "provider": provider,
        }

        if model:
            config_dict["model"] = model
        if api_key:
            config_dict["api_key"] = api_key
        if base_url:
            config_dict["base_url"] = base_url

        # Add any additional kwargs
        config_dict.update(kwargs)

        # Create configuration instance
        try:
            config = EmbeddingConfig(**config_dict)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}") from e

        # Create provider
        return EmbeddingProviderFactory.create_provider(config)
