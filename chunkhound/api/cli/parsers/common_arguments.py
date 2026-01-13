"""Common CLI argument patterns shared across parsers."""

import argparse
from pathlib import Path

from chunkhound.core.audience import parse_audience


def _parse_audience(value: str) -> str:
    try:
        return parse_audience(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to all commands.

    Args:
        parser: Argument parser to add common arguments to
    """
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )


def add_config_arguments(parser: argparse.ArgumentParser, configs: list[str]) -> None:
    """Add CLI arguments for specified config sections.

    Args:
        parser: Argument parser to add config arguments to
        configs: List of config section names to include
    """
    if "database" in configs:
        from chunkhound.core.config.database_config import DatabaseConfig

        DatabaseConfig.add_cli_arguments(parser)

    if "embedding" in configs:
        from chunkhound.core.config.embedding_config import EmbeddingConfig

        EmbeddingConfig.add_cli_arguments(parser)

    if "indexing" in configs:
        from chunkhound.core.config.indexing_config import IndexingConfig

        IndexingConfig.add_cli_arguments(parser)

    if "mcp" in configs:
        from chunkhound.core.config.mcp_config import MCPConfig

        MCPConfig.add_cli_arguments(parser)

    if "llm" in configs:
        from chunkhound.core.config.llm_config import LLMConfig

        LLMConfig.add_cli_arguments(parser)

    if "research" in configs:
        from chunkhound.core.config.research_config import ResearchConfig

        ResearchConfig.add_cli_arguments(parser)
