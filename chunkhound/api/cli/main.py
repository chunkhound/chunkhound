"""New modular CLI entry point for ChunkHound."""

import argparse
import logging as _pylogging
import asyncio
import multiprocessing
import sys
from pathlib import Path

from loguru import logger

from .utils.config_factory import create_validated_config

# Required for PyInstaller multiprocessing support
multiprocessing.freeze_support()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI.

    Args:
        verbose: Whether to enable verbose logging
    """
    logger.remove()

    if verbose:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
        )
    else:
        logger.add(
            sys.stderr,
            level="WARNING",
            format=(
                "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                "<level>{message}</level>"
            ),
        )
    # Also set stdlib logging level to avoid mixed loggers being noisy
    _pylogging.basicConfig(level=_pylogging.DEBUG if verbose else _pylogging.ERROR)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the complete argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    # Import parsers dynamically to avoid early loading
    from .parsers import create_main_parser, setup_subparsers
    from .parsers.calibrate_parser import add_calibrate_subparser
    from .parsers.mcp_parser import add_mcp_subparser
    from .parsers.research_parser import add_research_subparser
    from .parsers.autodoc_parser import add_autodoc_subparser
    from .parsers.code_mapper_parser import add_map_subparser
    from .parsers.run_parser import add_run_subparser
    from .parsers.search_parser import add_search_subparser

    parser = create_main_parser()
    subparsers = setup_subparsers(parser)

    # Add command subparsers
    add_run_subparser(subparsers)
    add_mcp_subparser(subparsers)
    add_search_subparser(subparsers)
    add_research_subparser(subparsers)
    add_autodoc_subparser(subparsers)
    add_map_subparser(subparsers)
    # Diagnose command retired; functionality lives under: index --check-ignores
    add_calibrate_subparser(subparsers)

    return parser


async def async_main() -> None:
    """Async main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging for non-MCP commands (MCP already handled above)
    setup_logging(getattr(args, "verbose", False))

    # Validate args and create config
    # Special-case: index subtools (--simulate, --check-ignores) never require embeddings
    if args.command == "index" and (
        getattr(args, "simulate", False) or getattr(args, "check_ignores", False)
    ):
        setattr(args, "no_embeddings", True)
    config, validation_errors = create_validated_config(args, args.command)

    if validation_errors:
        # Check if we can offer interactive setup wizard for index command
        if args.command in [None, "index"]:
            from .setup_wizard import _should_run_setup_wizard, run_setup_wizard

            if _should_run_setup_wizard(validation_errors):
                wizard_config = await run_setup_wizard(Path(args.path), args)

                if wizard_config:
                    # Re-validate with new config
                    config, validation_errors = create_validated_config(
                        args, args.command
                    )
                else:
                    # Wizard was run but returned None (user cancelled save)
                    # Exit gracefully without showing original validation errors
                    logger.info("Setup cancelled by user")
                    sys.exit(0)

        # If we still have errors after wizard (or wizard was skipped/cancelled)
        if validation_errors:
            # Check if this is an embedding-related error
            embedding_error = any(
                "embedding provider" in str(e).lower() for e in validation_errors
            )

            # Log all errors to stderr
            for error in validation_errors:
                logger.error(f"Error: {error}")

            # If embedding error and not in interactive mode, show helpful messages to stdout
            if embedding_error and args.command in [None, "index"]:
                # Use print() for stdout output to match test expectations
                print("To fix this, you can:")
                print("  1. Create a .chunkhound.json config file with embeddings")
                print("  2. Use --no-embeddings to skip embeddings")

            sys.exit(1)

    try:
        if args.command == "index":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.run import run_command

            await run_command(args, config)
        elif args.command == "mcp":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.mcp import mcp_command

            await mcp_command(args, config)
        elif args.command == "search":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.search import search_command

            await search_command(args, config)
        elif args.command == "research":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.research import research_command

            await research_command(args, config)
        elif args.command == "map":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.code_mapper import code_mapper_command

            await code_mapper_command(args, config)
        elif args.command == "autodoc":
            from .commands.autodoc import autodoc_command

            await autodoc_command(args, config)
        elif args.command == "calibrate":
            # Dynamic import to avoid early chunkhound module loading
            from .commands.calibrate import calibrate_command

            await calibrate_command(args, config)
        # 'diagnose' command retired; use: chunkhound index --check-ignores --vs git
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        logger.exception("Full error details:")
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        sys.exit(0)
    except ImportError as e:
        # More specific handling for import errors
        logger.error(f"Import error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        # Check if this is a Pydantic validation error for missing provider
        error_str = str(e)
        if (
            "validation error for EmbeddingConfig" in error_str
            and "provider" in error_str
        ):
            logger.error(
                "Embedding provider must be specified. "
                "Choose from: openai\n"
                "Set via --provider, CHUNKHOUND_EMBEDDING__PROVIDER environment "
                "variable, or in config file."
            )
        else:
            error_type = type(e).__name__
            logger.error(f"Unexpected error ({error_type}): {e}")

            # Add additional context for common terminal/Rich issues
            if "color format" in str(e).lower() or "wrong color" in str(e).lower():
                logger.error(
                    "This appears to be a terminal compatibility issue. "
                    "Try running with CHUNKHOUND_NO_RICH=1 environment variable."
                )
        sys.exit(1)


if __name__ == "__main__":
    main()
