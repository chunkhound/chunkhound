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


def add_git_diff_arguments(parser: argparse.ArgumentParser) -> None:
    """Add mutually-exclusive git commit-range arguments shared by search and research."""
    diff_group = parser.add_mutually_exclusive_group()
    diff_group.add_argument(
        "--commit-range",
        type=str,
        default=None,
        dest="commit_range",
        help="Git revision range (e.g. 'HEAD~10..HEAD', 'v1.0..v2.0').",
    )
    diff_group.add_argument(
        "--commit-hash",
        type=str,
        default=None,
        dest="commit_hash",
        help="Single commit hash — from that commit to HEAD.",
    )
    diff_group.add_argument(
        "--last-n",
        type=int,
        default=None,
        dest="last_n_commits",
        help="Last N commits (equivalent to HEAD~N..HEAD).",
    )
    parser.add_argument(
        "--vector-source",
        choices=["diff", "db", "both"],
        default="both",
        dest="vector_source",
        help="Search scope when commit input given: 'both' (default), 'diff', or 'db'.",
    )
