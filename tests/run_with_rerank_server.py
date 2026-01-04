#!/usr/bin/env python3
"""
Utility script to run tests with automatic reranking server management.

This script starts a mock reranking server, runs the specified tests,
and then cleans up the server afterwards.

Usage:
    python tests/run_with_rerank_server.py [pytest args]

Examples:
    # Run all multi-hop semantic search tests
    python tests/run_with_rerank_server.py tests/test_multi_hop_semantic_search.py -v

    # Run specific test
    python tests/run_with_rerank_server.py tests/test_embeddings.py::test_ollama_with_reranking_configuration -v

    # Run all tests that need reranking
    python tests/run_with_rerank_server.py -k rerank -v
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add tests directory to path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from fixtures.rerank_server_manager import RerankServerManager


async def run_tests_with_server(pytest_args: list[str]) -> int:
    """
    Run pytest with a mock reranking server.

    Args:
        pytest_args: Arguments to pass to pytest

    Returns:
        Exit code from pytest
    """
    manager = RerankServerManager()

    # Check if external server is already running
    if await manager.is_running():
        print("ℹ️  External reranking server detected on port 8001")
        print("   Using existing server instead of starting mock server")
        # Run tests with existing server
        result = subprocess.run(["pytest"] + pytest_args)
        return result.returncode

    print("🚀 Starting mock reranking server on port 8001...")

    async with manager:
        # Verify server is running
        if not await manager.is_running():
            print("❌ Failed to start mock reranking server")
            return 1

        print("✅ Mock reranking server started successfully")
        print(f"   Server URL: {manager.base_url}")
        print()
        print("🧪 Running tests...")
        print("-" * 50)

        # Run pytest with provided arguments
        result = subprocess.run(
            ["pytest"] + pytest_args,
            env={
                **subprocess.os.environ,
                # Ensure tests know server is available
                "CHUNKHOUND_TEST_RERANK_URL": manager.base_url,
            },
        )

        print("-" * 50)
        print()

    print("🛑 Mock reranking server stopped")

    return result.returncode


def main():
    """Main entry point."""
    # Get pytest arguments (everything after script name)
    pytest_args = sys.argv[1:] if len(sys.argv) > 1 else []

    # Default to running multi-hop tests if no args provided
    if not pytest_args:
        pytest_args = ["tests/test_multi_hop_semantic_search.py", "-v"]
        print("ℹ️  No arguments provided, defaulting to:")
        print(f"   pytest {' '.join(pytest_args)}")
        print()

    # Run tests with server
    exit_code = asyncio.run(run_tests_with_server(pytest_args))

    # Exit with same code as pytest
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
