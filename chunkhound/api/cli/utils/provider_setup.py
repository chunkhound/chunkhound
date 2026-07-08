"""Shared embedding/LLM manager construction for CLI commands."""

import sys

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.exceptions.core import ConfigurationError
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager

from .rich_output import RichOutputFormatter


def setup_embedding_manager(
    formatter: RichOutputFormatter, config: Config
) -> EmbeddingManager:
    """Construct an EmbeddingManager, exiting on provider setup failure."""
    embedding_manager = EmbeddingManager()

    try:
        if config.embedding:
            provider = EmbeddingProviderFactory.create_provider(config.embedding)
            embedding_manager.register_provider(provider, set_default=True)
    except ValueError as e:
        formatter.error(f"Embedding provider setup failed: {e}")
        formatter.info(
            "Configure an embedding provider via:\n"
            "1. Create .chunkhound.json with embedding configuration, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    return embedding_manager


def setup_llm_manager(
    formatter: RichOutputFormatter, config: Config
) -> LLMManager | None:
    """Construct an LLMManager if configured, exiting on provider setup failure."""
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            return LLMManager(utility_config, synthesis_config)
    except (ValueError, ConfigurationError) as e:
        formatter.error(f"LLM provider setup failed: {e}")
        formatter.info(
            "Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        formatter.error(f"Unexpected error setting up LLM provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    return None
