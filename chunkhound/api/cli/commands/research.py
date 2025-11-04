"""Research command module - handles deep code research operations."""

import argparse
import sys
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.llm_manager import LLMManager
from chunkhound.mcp_server.tools import deep_research_impl

from ..utils.rich_output import RichOutputFormatter
from ..utils.tree_progress import TreeProgressDisplay


async def research_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the research command using deep code research.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    # Initialize Rich output formatter
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Use database path from config
    if not config.database.path:
        formatter.error("Database path not configured")
        sys.exit(1)
    db_path = Path(config.database.path)

    # Check if database exists
    if not db_path.exists():
        formatter.error(f"Database not found at {db_path}")
        formatter.info("Run 'chunkhound index' to create the database first")
        sys.exit(1)

    # Registry is configured in database_factory.create_services().
    # Avoid double configuration here to prevent opening the DB twice and causing a self-lock.

    # Initialize embedding manager (exactly like MCP server)
    embedding_manager = EmbeddingManager()

    # Setup embedding provider (required for reranking)
    try:
        if config.embedding:
            provider = EmbeddingProviderFactory.create_provider(config.embedding)
            embedding_manager.register_provider(provider, set_default=True)
    except ValueError as e:
        # API key or configuration issue
        formatter.error(f"Embedding provider setup failed: {e}")
        formatter.info(
            "Configure an embedding provider via:\n"
            "1. Create .chunkhound.json with embedding configuration, OR\n"
            "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        # Unexpected error
        formatter.error(f"Unexpected error setting up embedding provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Initialize LLM manager (required for code research)
    llm_manager: LLMManager | None = None
    try:
        if config.llm:
            utility_config, synthesis_config = config.llm.get_provider_configs()
            llm_manager = LLMManager(utility_config, synthesis_config)
    except ValueError as e:
        # API key or configuration issue
        formatter.error(f"LLM provider setup failed: {e}")
        formatter.info(
            "Configure an LLM provider via:\n"
            "1. Create .chunkhound.json with llm configuration, OR\n"
            "2. Set CHUNKHOUND_LLM_API_KEY environment variable"
        )
        sys.exit(1)
    except Exception as e:
        # Unexpected error
        formatter.error(f"Unexpected error setting up LLM provider: {e}")
        logger.exception("Full error details:")
        sys.exit(1)

    # Create services using unified factory (exactly like MCP)
    try:
        services = create_services(
            db_path=db_path, config=config, embedding_manager=embedding_manager
        )
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    # Create tree progress display for terminal UI
    with TreeProgressDisplay() as tree_progress:
        try:
            # Perform deep research with tree progress tracking
            result = await deep_research_impl(
                services=services,
                embedding_manager=embedding_manager,
                llm_manager=llm_manager,
                query=args.query,
                progress=tree_progress,
            )

            # Output the markdown result (already formatted by the service)
            print("\n")  # Add newline after progress output
            print(result.get("answer", f"Research incomplete: Unable to analyze '{args.query}'. Try a more specific query or check that relevant code exists."))

        except Exception as e:
            formatter.error(f"Research failed: {e}")
            logger.exception("Full error details:")
            sys.exit(1)
