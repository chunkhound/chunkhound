"""Configuration factory for consolidated config loading and validation.

This module provides centralized functions to create and validate configuration
instances, eliminating duplication across CLI commands and MCP servers.
"""

import argparse
from pathlib import Path

from chunkhound.core.config.database_config import DatabaseConfig
from chunkhound.core.config.config import Config
from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.config.mcp_config import MCPConfig
from chunkhound.core.config.research_config import ResearchConfig


def _fallback_config(args: argparse.Namespace) -> Config:
    """Construct a minimal config object when validated loading fails."""
    target_dir = getattr(args, "path", None)
    resolved_target = None
    if target_dir is not None:
        try:
            resolved_target = Path(target_dir).resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            resolved_target = None

    return Config.model_construct(
        database=DatabaseConfig(),
        embedding=None,
        llm=None,
        mcp=MCPConfig(),
        indexing=IndexingConfig(),
        research=ResearchConfig(),
        debug=False,
        target_dir=resolved_target,
        embeddings_disabled=bool(getattr(args, "no_embeddings", False)),
    )


def create_validated_config(
    args: argparse.Namespace, command: str
) -> tuple[Config, list[str]]:
    """Create and validate config for a specific command.

    This centralizes the config loading pattern that was duplicated across
    main.py, run.py, and mcp_server.py.

    Args:
        args: Parsed command-line arguments
        command: Command name for validation ('index', 'mcp')

    Returns:
        tuple: (config_instance, validation_errors)
    """
    try:
        config = Config(args=args)
    except ValueError as exc:
        return _fallback_config(args), [str(exc)]

    validation_errors = config.validate_for_command(command, args)
    return config, validation_errors


def create_config(args: argparse.Namespace) -> Config:
    """Create config without validation.

    Use this when validation isn't needed or will be done separately.

    Args:
        args: Parsed command-line arguments

    Returns:
        Config instance
    """
    return Config(args=args)


def create_default_config() -> Config:
    """Create config with defaults (no CLI args).

    Use this for fallback scenarios where args aren't available.

    Returns:
        Config instance with defaults
    """
    return Config()
