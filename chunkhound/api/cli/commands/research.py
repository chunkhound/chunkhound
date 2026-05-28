"""Research command module - handles deep code research operations."""

import argparse
import os
import sys

from loguru import logger

from chunkhound.api.cli.utils import verify_database_exists
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.exceptions.core import ConfigurationError
from chunkhound.database_factory import DatabaseServices, create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


def setup_embedding_llm(
    formatter: RichOutputFormatter, config: Config
) -> tuple[EmbeddingManager, LLMManager | None]:
    """Set up embedding and LLM managers with error handling."""
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

    llm_manager: LLMManager | None = None
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
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

    return embedding_manager, llm_manager


async def run_research(
    services: DatabaseServices,
    embedding_manager: EmbeddingManager,
    llm_manager: LLMManager | None,
    query: str,
    path_filter: str | None,
    config: Config,
    formatter: RichOutputFormatter,
    commit_range: str | None = None,
    commit_hash: str | None = None,
    last_n_commits: int | None = None,
    vector_source: str = "both",
) -> None:
    """Run deep_research_impl with TreeProgressDisplay and print result."""
    progress_output = (
        sys.stderr if os.environ.get("CHUNKHOUND_QUICKRESEARCH_QUIET") else sys.stdout
    )
    with TreeProgressDisplay(output=progress_output) as tree_progress:
        try:
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=query,
                progress=tree_progress,
                path=path_filter,
                config=config,
                commit_range=commit_range,
                commit_hash=commit_hash,
                last_n_commits=last_n_commits,
                vector_source=vector_source,
            )
            print("\n")
            print(
                result.get(
                    "answer",
                    f"Research incomplete: Unable to analyze '{query}'. "
                    "Try a more specific query or check that relevant code exists.",
                )
            )
        except Exception as e:
            formatter.error(f"Research failed: {e}")
            logger.exception("Full error details:")
            sys.exit(1)


async def research_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the research command using deep code research.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    formatter = RichOutputFormatter(verbose=args.verbose)

    try:
        db_path = verify_database_exists(config)
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    embedding_manager, llm_manager = setup_embedding_llm(formatter, config)

    # Registry is configured in create_services().
    # Avoid double configuration here — opening the DB twice causes a self-lock.
    try:
        services = create_services(
            db_path=db_path, config=config, embedding_manager=embedding_manager
        )
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    await run_research(
        services, embedding_manager, llm_manager,
        args.query, args.path_filter, config, formatter,
        commit_range=getattr(args, "commit_range", None),
        commit_hash=getattr(args, "commit_hash", None),
        last_n_commits=getattr(args, "last_n_commits", None),
        vector_source=getattr(args, "vector_source", "diff"),
    )
