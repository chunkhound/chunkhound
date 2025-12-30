"""Common CLI argument patterns shared across parsers."""

import argparse
from pathlib import Path


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
    parser.add_argument(
        "--log-file",
        type=str,
        help="Enable file logging to specified path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set file logging level (default: INFO)",
    )
    parser.add_argument(
        "--performance-log",
        type=str,
        help="Enable separate performance timing log to specified path",
    )
    parser.add_argument(
        "--progress-display-log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set log level for progress display messages (default: WARNING)",
    )
    parser.add_argument(
        "--max-log-messages",
        type=int,
        help="Maximum number of log messages to buffer for progress display (default: 100)",
    )
    parser.add_argument(
        "--log-panel-ratio",
        type=float,
        help="Ratio of terminal height for log panel in progress display (0.0-1.0, default: 0.3)",
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
