"""
OpenAI embedding configuration for ChunkHound.

This module provides a type-safe, validated configuration system for OpenAI
embeddings with support for multiple configuration sources (environment
variables, config files, CLI arguments) across MCP server and indexing flows.
"""

import argparse
import os
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL

from .openai_utils import is_official_openai_endpoint


# Error message constants for consistent messaging across config and provider
RERANK_MODEL_REQUIRED_COHERE = (
    "rerank_model is required when using rerank_format='cohere'. "
    "Either provide rerank_model or use rerank_format='tei'."
)
RERANK_BASE_URL_REQUIRED = (
    "base_url is required when using reranking with relative rerank_url. "
    "Either provide base_url or use an absolute rerank_url (http://...)"
)


def validate_rerank_configuration(
    provider: str,
    rerank_format: str,
    rerank_model: str | None,
    rerank_url: str,
    base_url: str | None,
) -> None:
    """Validate rerank configuration consistency.

    Shared validation logic used by both config and provider layers.

    Args:
        provider: Embedding provider name
        rerank_format: Reranking API format ('cohere', 'tei', or 'auto')
        rerank_model: Model name for reranking (optional for TEI)
        rerank_url: Rerank endpoint URL
        base_url: Base URL for API (required for relative rerank_url)

    Raises:
        ValueError: If configuration is invalid
    """
    # VoyageAI uses SDK-based reranking, doesn't need URL configuration
    if provider == "voyageai":
        return

    # For Cohere format, rerank_model is required
    if rerank_format == "cohere" and not rerank_model:
        raise ValueError(RERANK_MODEL_REQUIRED_COHERE)

    # If using reranking (model set or TEI format with URL), validate URL config
    is_using_reranking = rerank_model or (rerank_format == "tei" and rerank_url)

    if is_using_reranking:
        # For relative URLs, we need base_url
        if not rerank_url.startswith(("http://", "https://")) and not base_url:
            raise ValueError(RERANK_BASE_URL_REQUIRED)


class EmbeddingConfig(BaseSettings):
    """
    OpenAI embedding configuration for ChunkHound.

    Configuration Sources (in order of precedence):
    1. CLI arguments
    2. Environment variables (CHUNKHOUND_EMBEDDING_*)
    3. Config files
    4. Default values

    Environment Variables:
        CHUNKHOUND_EMBEDDING_API_KEY=sk-...
        CHUNKHOUND_EMBEDDING_MODEL=text-embedding-3-small
        CHUNKHOUND_EMBEDDING_BASE_URL=https://api.openai.com/v1
    """

    model_config = SettingsConfigDict(
        env_prefix="CHUNKHOUND_EMBEDDING_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="ignore",  # Ignore unknown fields for forward compatibility
    )

    # Provider Selection
    provider: Literal["openai", "voyageai"] = Field(
        default="openai", description="Embedding provider (openai, voyageai)"
    )

    # Common Configuration
    model: str | None = Field(
        default=None,
        description="Embedding model name (uses provider default if not specified)",
    )

    api_key: SecretStr | None = Field(
        default=None, description="API key for authentication (provider-specific)"
    )

    base_url: str | None = Field(
        default=None, description="Base URL for the embedding API"
    )

    rerank_model: str | None = Field(
        default=None,
        description="Reranking model name (enables multi-hop search if specified)",
    )

    rerank_url: str = Field(
        default="/rerank",
        description="Rerank endpoint URL. Absolute URLs (http/https) used as-is for separate services. "
        "Relative paths combined with base_url for same-server reranking.",
    )

    rerank_format: Literal["cohere", "tei", "auto"] = Field(
        default="auto",
        description="Reranking API format. 'cohere' for Cohere-compatible APIs (requires model in request), "
        "'tei' for Hugging Face Text Embeddings Inference (model set at deployment), "
        "'auto' for automatic format detection from response.",
    )

    # Internal settings - not exposed to users
    batch_size: int = Field(default=100, description="Internal batch size")
    rerank_batch_size: int | None = Field(
        default=None,
        description="Max documents per rerank batch (overrides model defaults, bounded by model caps)",
    )
    timeout: int = Field(default=30, description="Internal timeout")
    max_retries: int = Field(default=3, description="Internal max retries")
    max_concurrent_batches: int | None = Field(
        default=None,
        description="Internal concurrency (auto-detected from provider if not set)",
    )
    optimization_batch_frequency: int = Field(
        default=1000,
        description="Internal optimization frequency (runs every N batches during indexing)",
    )

    @field_validator("rerank_batch_size")
    def validate_rerank_batch_size(cls, v: int | None) -> int | None:  # noqa: N805
        """Validate rerank batch size is positive."""
        if v is not None and v <= 0:
            raise ValueError("rerank_batch_size must be positive")
        return v

    @field_validator("model")
    def validate_model(cls, v: str | None) -> str | None:  # noqa: N805
        """Fix common model name typos."""
        if v is None:
            return v

        # Fix common typos
        typo_fixes = {
            "text-embedding-small": "text-embedding-3-small",
            "text-embedding-large": "text-embedding-3-large",
        }

        return typo_fixes.get(v, v)

    @field_validator("base_url")
    def validate_base_url(cls, v: str | None) -> str | None:  # noqa: N805
        """Validate and normalize base URL."""
        if v is None:
            return v

        # Remove trailing slash for consistency
        v = v.rstrip("/")

        # Basic URL validation
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("base_url must start with http:// or https://")

        return v

    @model_validator(mode="after")
    def validate_rerank_config(self) -> Self:
        """Validate rerank configuration using shared validation logic."""
        validate_rerank_configuration(
            provider=self.provider,
            rerank_format=self.rerank_format,
            rerank_model=self.rerank_model,
            rerank_url=self.rerank_url,
            base_url=self.base_url,
        )
        return self

    def get_provider_config(self) -> dict[str, Any]:
        """
        Get provider-specific configuration dictionary.

        Returns:
            Dictionary containing configuration parameters for the selected provider
        """
        base_config = {
            "provider": self.provider,
            # Always provide resolved model to factory
            "model": self.get_default_model(),
            "batch_size": self.batch_size,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

        # Add API key if available
        if self.api_key:
            base_config["api_key"] = self.api_key.get_secret_value()

        # Add base URL if available
        if self.base_url:
            base_config["base_url"] = self.base_url

        # Add rerank configuration if available
        if self.rerank_model:
            base_config["rerank_model"] = self.rerank_model
        base_config["rerank_url"] = self.rerank_url
        base_config["rerank_format"] = self.rerank_format
        if self.rerank_batch_size is not None:
            base_config["rerank_batch_size"] = self.rerank_batch_size

        return base_config

    def get_default_model(self) -> str:
        """
        Get the model name, using default if not specified.

        Returns:
            Model name or provider default
        """
        if self.model:
            return self.model

        # Provider defaults
        if self.provider == "voyageai":
            return VOYAGE_DEFAULT_MODEL
        else:  # openai
            return "text-embedding-3-small"

    def is_provider_configured(self) -> bool:
        """
        Check if the selected provider is properly configured.

        Returns:
            True if provider is properly configured
        """
        if self.provider == "openai":
            # For OpenAI provider, only require API key for official endpoints
            if is_official_openai_endpoint(self.base_url):
                return self.api_key is not None
            else:
                # Custom endpoints don't require API key
                return True
        else:
            # For other providers (voyageai, etc.), always require API key
            return self.api_key is not None

    def get_missing_config(self) -> list[str]:
        """
        Get list of missing required configuration.

        Returns:
            List of missing configuration parameter names
        """
        missing = []

        if self.provider == "openai":
            # For OpenAI provider, only require API key for official endpoints
            if is_official_openai_endpoint(self.base_url) and not self.api_key:
                missing.append("api_key (set CHUNKHOUND_EMBEDDING_API_KEY)")
        else:
            # For other providers (voyageai, etc.), always require API key
            if not self.api_key:
                missing.append("api_key (set CHUNKHOUND_EMBEDDING_API_KEY)")

        return missing

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add embedding-related CLI arguments."""
        parser.add_argument(
            "--model",
            "--embedding-model",
            help="Embedding model (default: text-embedding-3-small)",
        )

        parser.add_argument(
            "--api-key",
            "--embedding-api-key",
            help="API key for embedding provider (uses env var if not specified)",
        )

        parser.add_argument(
            "--base-url",
            "--embedding-base-url",
            help="Base URL for embedding API (uses env var if not specified)",
        )

        parser.add_argument(
            "--no-embeddings",
            action="store_true",
            help="Skip embedding generation (index code only)",
        )

    @classmethod
    def load_from_env(cls) -> dict[str, Any]:
        """Load embedding config from environment variables."""
        config = {}

        if api_key := os.getenv("CHUNKHOUND_EMBEDDING__API_KEY"):
            config["api_key"] = api_key
        if base_url := os.getenv("CHUNKHOUND_EMBEDDING__BASE_URL"):
            config["base_url"] = base_url
        if provider := os.getenv("CHUNKHOUND_EMBEDDING__PROVIDER"):
            config["provider"] = provider
        if model := os.getenv("CHUNKHOUND_EMBEDDING__MODEL"):
            config["model"] = model

        # Reranking configuration
        if rerank_model := os.getenv("CHUNKHOUND_EMBEDDING__RERANK_MODEL"):
            config["rerank_model"] = rerank_model
        if rerank_url := os.getenv("CHUNKHOUND_EMBEDDING__RERANK_URL"):
            config["rerank_url"] = rerank_url
        if rerank_format := os.getenv("CHUNKHOUND_EMBEDDING__RERANK_FORMAT"):
            config["rerank_format"] = rerank_format
        if rerank_batch_size := os.getenv("CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE"):
            try:
                config["rerank_batch_size"] = int(rerank_batch_size)
            except ValueError:
                pass

        return config

    @classmethod
    def extract_cli_overrides(cls, args: Any) -> dict[str, Any]:
        """Extract embedding config from CLI arguments."""
        overrides = {}

        # Handle model arguments (both variations)
        if hasattr(args, "model") and args.model:
            overrides["model"] = args.model
        if hasattr(args, "embedding_model") and args.embedding_model:
            overrides["model"] = args.embedding_model

        # Handle API key arguments (both variations)
        if hasattr(args, "api_key") and args.api_key:
            overrides["api_key"] = args.api_key
        if hasattr(args, "embedding_api_key") and args.embedding_api_key:
            overrides["api_key"] = args.embedding_api_key

        # Handle base URL arguments (both variations)
        if hasattr(args, "base_url") and args.base_url:
            overrides["base_url"] = args.base_url
        if hasattr(args, "embedding_base_url") and args.embedding_base_url:
            overrides["base_url"] = args.embedding_base_url

        # Handle no-embeddings flag (special case - disables embeddings)
        if hasattr(args, "no_embeddings") and args.no_embeddings:
            return {"disabled": True}  # This will be handled specially in main Config

        return overrides

    def __repr__(self) -> str:
        """String representation hiding sensitive information."""
        api_key_display = "***" if self.api_key else None
        return (
            f"EmbeddingConfig("
            f"provider={self.provider}, "
            f"model={self.get_default_model()}, "
            f"api_key={api_key_display}, "
            f"base_url={self.base_url})"
        )
