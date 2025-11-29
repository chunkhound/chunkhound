"""Search command module - handles semantic and regex search operations."""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from loguru import logger

from chunkhound.api.cli.utils import verify_database_exists
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.database_factory import create_services
from chunkhound.embeddings import EmbeddingManager
from chunkhound.mcp_server.tools import search_regex_impl, search_semantic_impl
from chunkhound.registry import configure_registry

from ..utils.rich_output import RichOutputFormatter


def format_path_native(path: str) -> str:
    """Convert stored forward-slash path to native platform format."""
    import os

    return path.replace("/", os.sep) if path else path


async def search_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the search command using the service layer.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    # Initialize Rich output formatter
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Verify database exists and get paths
    try:
        verify_database_exists(config)  # Validates and raises if missing
        db_path = config.database.path  # Raw path for provider
    except (ValueError, FileNotFoundError) as e:
        formatter.error(str(e))
        sys.exit(1)

    # Configure registry with the Config object (like index command does)
    try:
        configure_registry(config)
    except Exception as e:
        formatter.error(f"Failed to configure registry: {e}")
        sys.exit(1)

    # Initialize embedding manager (exactly like MCP server)
    embedding_manager = EmbeddingManager()

    # Setup embedding provider (optional - continue if it fails)
    try:
        if config.embedding:
            provider = EmbeddingProviderFactory.create_provider(config.embedding)
            embedding_manager.register_provider(provider, set_default=True)
    except ValueError as e:
        # API key or configuration issue - expected for search-only usage
        logger.debug(f"Embedding provider setup skipped: {e}")
    except Exception as e:
        # Unexpected error - log but continue
        logger.debug(f"Unexpected error setting up embedding provider: {e}")

    # Create services using unified factory (exactly like MCP)
    try:
        services = create_services(
            db_path=db_path, config=config, embedding_manager=embedding_manager
        )
    except Exception as e:
        formatter.error(f"Failed to initialize services: {e}")
        sys.exit(1)

    # Determine search strategy
    force_strategy = None
    if args.single_hop:
        force_strategy = "single_hop"
    elif args.multi_hop:
        force_strategy = "multi_hop"

    try:
        if args.regex:
            # Perform regex search
            result = await search_regex_impl(
                services=services,
                pattern=args.query,
                page_size=args.page_size,
                offset=args.offset,
                path=args.path_filter,
            )
            result_dict = cast(dict[str, Any], result)
        else:
            # For semantic search, we need to handle force_strategy
            if force_strategy:
                # CLI-specific: When force_strategy is set, we need to call
                # the service directly
                # First validate using MCP's validation logic
                if not embedding_manager or not embedding_manager.list_providers():
                    raise Exception(
                        "No embedding providers available. "
                        "Configure an embedding provider via:\n"
                        "1. Create .chunkhound.json with embedding configuration, OR\n"
                        "2. Set CHUNKHOUND_EMBEDDING__API_KEY environment variable"
                    )

                # Get provider/model like search_semantic_impl does
                provider_name = None
                model_name = None
                try:
                    default_provider_obj = embedding_manager.get_provider()
                    provider_name = default_provider_obj.name
                    model_name = default_provider_obj.model
                except ValueError:
                    raise Exception(
                        "No default embedding provider configured. "
                        "Configure a default provider in config."
                    )

                # Call service directly with force_strategy
                results, pagination = await services.search_service.search_semantic(
                    query=args.query,
                    page_size=args.page_size,
                    offset=args.offset,
                    threshold=None,
                    provider=provider_name,
                    model=model_name,
                    path_filter=args.path_filter,
                    force_strategy=force_strategy,
                )
                result_dict = {"results": results, "pagination": pagination}
            else:
                # No force_strategy - use MCP's search_semantic_impl directly
                result = await search_semantic_impl(
                    services=services,
                    embedding_manager=embedding_manager,
                    query=args.query,
                    page_size=args.page_size,
                    offset=args.offset,
                    path=args.path_filter,
                )
                result_dict = cast(dict[str, Any], result)

        # Format and display results
        _format_search_results(formatter, result_dict, args.query, args.regex)

    except Exception as e:
        formatter.error(f"Search failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


def _format_search_results(
    formatter: RichOutputFormatter, result: dict[str, Any], query: str, is_regex: bool
) -> None:
    """Format and display search results.

    Args:
        formatter: Rich output formatter for colored display
        result: Search result dictionary with 'results' and 'pagination' keys
        query: The search query/pattern
        is_regex: Whether this was a regex search
    """
    results = result.get("results", [])
    pagination = result.get("pagination", {})

    search_type = "regex" if is_regex else "semantic"

    if not results:
        formatter.info(f"No results found for {search_type} search: '{query}'")
        return

    # Display header
    total = pagination.get("total", len(results))
    offset = pagination.get("offset", 0)
    page_size = pagination.get("page_size", len(results))

    formatter.section_header(f"{search_type.title()} Search Results")
    formatter.info(f"Query: '{query}'")
    start_idx = offset + 1
    end_idx = offset + len(results)
    formatter.info(
        f"Results: {len(results)} of {total} (showing {start_idx}-{end_idx})"
    )

    # Display each result
    for i, result_item in enumerate(results, 1):
        file_path = result_item.get("file_path", "unknown")
        # Convert to native path format for display
        native_path = format_path_native(file_path)
        content = result_item.get("content", "")

        # Display result header
        formatter.info(f"\n[{offset + i}] {native_path}")

        # Show similarity score for semantic search
        if not is_regex:
            similarity = result_item.get("similarity")
            score = result_item.get("score")
            if score is not None:
                formatter.info(f"Score: {score:.3f}")
            elif similarity is not None:
                formatter.info(f"Similarity: {similarity:.3f}")

        # Display content with line numbers if available
        start_line = result_item.get("start_line")
        end_line = result_item.get("end_line")

        if start_line is not None:
            line_info = f"Lines {start_line}"
            if end_line is not None and end_line != start_line:
                line_info += f"-{end_line}"
            formatter.info(line_info)

        # Display the actual content
        if content:
            # Truncate very long content for readability
            if len(content) > 1000:
                content = content[:997] + "..."

            # Use a simple text box for content
            print(f"```\n{content}\n```")

    # Display pagination info
    has_more = pagination.get("has_more", False)
    if has_more:
        next_offset = pagination.get("next_offset", offset + page_size)
        formatter.info(
            f"\nMore results available. Use --offset {next_offset} to see next page."
        )
