"""Validation utilities for ChunkHound CLI arguments."""

import sys
from pathlib import Path

from loguru import logger


def validate_path(
    path: Path, must_exist: bool = True, must_be_dir: bool = True
) -> bool:
    """Validate a file system path."""
    if must_exist and not path.exists():
        logger.error(f"Path does not exist: {path}")
        return False

    if must_exist and must_be_dir and not path.is_dir():
        logger.error(f"Path is not a directory: {path}")
        return False

    return True


def ensure_database_directory(db_path: Path) -> bool:
    """Ensure the database directory exists."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create database directory: {e}")
        return False


def ensure_config_directory(config_path: Path | None) -> bool:
    """Ensure configuration directory exists."""
    if config_path is None:
        return True

    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Cannot access configuration directory: {e}")
        return False


def validate_provider_args(
    provider: str | None, api_key: str | None, base_url: str | None, model: str | None
) -> bool:
    """Validate embedding provider arguments."""
    if not provider:
        logger.error(
            """
            No embedding provider configured. Choose from: openai, voyageai, mistral.
            To fix this, you can:
            1) Set via --provider flag
            2) Set CHUNKHOUND_EMBEDDING__PROVIDER environment variable
            3) Add provider to .chunkhound.json config file
            4) Use --no-embeddings to skip embeddings entirely.
            
            For more information, see: https://ofriw.github.io/chunkhound/configuration/
            """
        )
        return False

    if provider == "openai":
        # Only require API key for official OpenAI endpoints
        from chunkhound.core.config.openai_utils import is_official_openai_endpoint

        if is_official_openai_endpoint(base_url) and not api_key:
            logger.error(
                "OpenAI API key required for official OpenAI endpoints. "
                "Set CHUNKHOUND_EMBEDDING_API_KEY or use --api-key"
            )
            return False
    elif provider == "voyageai":
        if not api_key:
            logger.error(
                "VoyageAI API key required. Set CHUNKHOUND_EMBEDDING_API_KEY "
                "or use --api-key"
            )
            return False
    elif provider == "mistral":
        if not api_key:
            logger.error(
                "Mistral API key required. Set CHUNKHOUND_EMBEDDING_API_KEY "
                "or use --api-key"
            )
            return False
    else:
        logger.error(f"Unknown provider: {provider}")
        return False

    return True


def validate_config_args(
    server_type: str, base_url: str | None, model: str | None, api_key: str | None
) -> bool:
    """Validate configuration server arguments."""
    if server_type in ["openai"] and not model:
        logger.error(f"Model is required for {server_type} servers")
        return False

    if not base_url:
        logger.error("Base URL is required")
        return False

    if not (base_url.startswith("http://") or base_url.startswith("https://")):
        logger.error("Base URL must start with http:// or https://")
        return False

    return True


def validate_file_patterns(
    include_patterns: list[str] | None, exclude_patterns: list[str] | None
) -> bool:
    """Validate file inclusion and exclusion patterns."""
    if include_patterns is not None:
        if any(not pattern.strip() for pattern in include_patterns):
            logger.error("Include patterns cannot be empty")
            return False

    if exclude_patterns is not None:
        if any(not pattern.strip() for pattern in exclude_patterns):
            logger.error("Exclude patterns cannot be empty")
            return False

    return True


def validate_server_name(name: str, existing_servers: list[str]) -> bool:
    """Validate server name for uniqueness and format."""
    if not name or not name.strip():
        logger.error("Server name cannot be empty")
        return False

    if not name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        logger.error(
            "Server name can only contain letters, numbers, hyphens, "
            "underscores, and dots"
        )
        return False

    if name in existing_servers:
        logger.error(f"Server '{name}' already exists")
        return False

    return True


def validate_numeric_args(batch_size: int | None = None) -> bool:
    """Validate numeric arguments."""
    if batch_size is not None:
        if batch_size < 1:
            logger.error("Batch size must be at least 1")
            return False
        if batch_size > 1000:
            logger.error("Batch size cannot exceed 1000")
            return False

    return True


def validate_embedding_dimension(dimension: int | None) -> bool:
    """Validate embedding dimension parameter."""
    if dimension is None:
        return True

    if dimension < 1:
        logger.error("Embedding dimension must be positive")
        return False

    if dimension > 10000:  # Reasonable upper limit
        logger.error("Embedding dimension is too large (max 10,000)")
        return False

    return True


def validate_timeout_args(timeout: float | None) -> bool:
    """Validate timeout arguments."""
    if timeout is None:
        return True

    if timeout <= 0:
        logger.error("Timeout must be positive")
        return False

    if timeout > 300:  # 5 minutes max
        logger.error("Timeout cannot exceed 300 seconds")
        return False

    return True


def exit_on_validation_error(message: str) -> None:
    """Print error message and exit with error code."""
    logger.error(message)
    sys.exit(1)
