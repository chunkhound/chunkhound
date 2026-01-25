"""Test utilities shared across test files."""

import json
import os
from pathlib import Path

from chunkhound.embeddings import EmbeddingManager


def _find_config_file() -> Path | None:
    """Find .chunkhound.json in current or parent directory."""
    search_paths = [
        Path(".chunkhound.json"),      # Current directory
        Path("../.chunkhound.json"),   # Parent directory
    ]

    for config_file in search_paths:
        if config_file.exists():
            return config_file
    return None


def get_api_key_for_tests() -> tuple[str | None, str | None]:
    """
    Intelligently discover API key and provider for testing.

    DEPRECATED: Use get_embedding_config_for_tests() for new code.
    This function is maintained for backwards compatibility only.

    Priority:
    1. CHUNKHOUND_EMBEDDING__API_KEY environment variable
    2. .chunkhound.json in current or parent directory
    3. Return (None, None) if not found

    Returns:
        Tuple of (api_key, provider) or (None, None) if not found
    """
    config_dict = get_embedding_config_for_tests()
    if not config_dict:
        return None, None

    api_key = config_dict.get("api_key")
    provider = config_dict.get("provider")
    return api_key, provider


def get_embedding_config_for_tests() -> dict | None:
    """
    Intelligently discover full embedding configuration for testing.

    Priority:
    1. CHUNKHOUND_EMBEDDING__* environment variables
    2. .chunkhound.json in current or parent directory

    Returns:
        Dictionary with embedding config fields or None if not found.
        Fields: api_key, provider, model, base_url,
                rerank_model, rerank_url, rerank_format, rerank_batch_size

    Usage:
        Use with build_embedding_config_from_dict() to create Config-compatible dict:

        config_dict = get_embedding_config_for_tests()
        embedding_config = build_embedding_config_from_dict(config_dict)
    """
    config: dict[str, str | int] = {}

    # Priority 1: Environment variables
    api_key = os.environ.get("CHUNKHOUND_EMBEDDING__API_KEY")
    if api_key and api_key.strip():
        config["api_key"] = api_key.strip()

    # Check for other env vars
    if provider := os.environ.get("CHUNKHOUND_EMBEDDING__PROVIDER"):
        config["provider"] = provider
    if model := os.environ.get("CHUNKHOUND_EMBEDDING__MODEL"):
        config["model"] = model
    if base_url := os.environ.get("CHUNKHOUND_EMBEDDING__BASE_URL"):
        config["base_url"] = base_url
    if rerank_model := os.environ.get("CHUNKHOUND_EMBEDDING__RERANK_MODEL"):
        config["rerank_model"] = rerank_model
    if rerank_url := os.environ.get("CHUNKHOUND_EMBEDDING__RERANK_URL"):
        config["rerank_url"] = rerank_url
    if rerank_format := os.environ.get("CHUNKHOUND_EMBEDDING__RERANK_FORMAT"):
        config["rerank_format"] = rerank_format
    if rerank_batch_size := os.environ.get("CHUNKHOUND_EMBEDDING__RERANK_BATCH_SIZE"):
        try:
            config["rerank_batch_size"] = int(rerank_batch_size)
        except ValueError:
            pass

    # Priority 2: Local .chunkhound.json file
    config_file = _find_config_file()
    if config_file:
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            embedding_config = config_data.get("embedding", {})

            # Merge with config, but don't override env vars
            if "api_key" not in config and (api_key := embedding_config.get("api_key")):
                config["api_key"] = api_key
            if "provider" not in config and (provider := embedding_config.get("provider")):
                config["provider"] = provider
            if "model" not in config and (model := embedding_config.get("model")):
                config["model"] = model
            if "base_url" not in config and (base_url := embedding_config.get("base_url")):
                config["base_url"] = base_url
            if "rerank_model" not in config and (rerank_model := embedding_config.get("rerank_model")):
                config["rerank_model"] = rerank_model
            if "rerank_url" not in config and (rerank_url := embedding_config.get("rerank_url")):
                config["rerank_url"] = rerank_url
            if "rerank_format" not in config and (rerank_format := embedding_config.get("rerank_format")):
                config["rerank_format"] = rerank_format
            if "rerank_batch_size" not in config and (rerank_batch_size := embedding_config.get("rerank_batch_size")):
                try:
                    config["rerank_batch_size"] = int(rerank_batch_size)
                except ValueError:
                    pass

        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass

    # Return None if no API key found (minimum requirement)
    if not config.get("api_key"):
        return None

    return config


def build_embedding_config_from_dict(config_dict: dict | None) -> dict | None:
    """
    Build embedding config dict suitable for Config() from discovered config.

    This helper eliminates code duplication across test files by centralizing
    the logic for propagating optional fields from discovered config to the
    embedding config dict expected by Config().

    Args:
        config_dict: Output from get_embedding_config_for_tests()

    Returns:
        Embedding config dict or None if no config provided

    Example:
        config_dict = get_embedding_config_for_tests()
        embedding_config = build_embedding_config_from_dict(config_dict)
        config = Config(
            database={"path": str(db_path), "provider": "duckdb"},
            embedding=embedding_config
        )
    """
    if not config_dict:
        return None

    embedding_config = {
        "provider": config_dict.get("provider", "openai"),
        "api_key": config_dict["api_key"],
    }

    # Optional fields - propagate only if present
    optional_fields = [
        "model", "base_url",
        "rerank_model", "rerank_url", "rerank_format", "rerank_batch_size"
    ]
    for field in optional_fields:
        if field in config_dict:
            embedding_config[field] = config_dict[field]

    return embedding_config


def create_embedding_manager_for_tests(config_dict: dict | None) -> EmbeddingManager | None:
    """
    Create EmbeddingManager from discovered config.

    Centralizes provider instantiation logic used across test files.

    Args:
        config_dict: Output from get_embedding_config_for_tests()

    Returns:
        Configured EmbeddingManager or None if no config provided

    Example:
        config_dict = get_embedding_config_for_tests()
        embedding_manager = create_embedding_manager_for_tests(config_dict)
    """
    if not config_dict:
        return None

    provider = config_dict.get("provider", "openai")
    api_key = config_dict["api_key"]
    model = config_dict.get("model")
    base_url = config_dict.get("base_url")

    manager = EmbeddingManager()
    kwargs = {"api_key": api_key}
    if model:
        kwargs["model"] = model

    if provider == "openai":
        if base_url:
            kwargs["base_url"] = base_url
        from chunkhound.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
        manager.register_provider(OpenAIEmbeddingProvider(**kwargs), set_default=True)
    elif provider == "voyageai":
        from chunkhound.providers.embeddings.voyageai_provider import VoyageAIEmbeddingProvider
        manager.register_provider(VoyageAIEmbeddingProvider(**kwargs), set_default=True)
    else:
        # Unsupported provider - return None
        return None

    return manager


def should_run_live_api_tests(expected_provider: str | None = None) -> bool:
    """Check if live API tests should run for the specified provider."""
    api_key, provider = get_api_key_for_tests()
    if not api_key:
        return False

    if expected_provider and provider != expected_provider:
        return False

    return True
