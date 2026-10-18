"""Integration tests for OpenCode CLI LLM provider.

These tests exercise the real opencode subprocess and require the OpenCode CLI
to be installed. Pass --run-live and an absolute --live-config JSON file with
opencode.models and optional explicit environment credentials.
"""

import pytest

from chunkhound.providers.llm.opencode_cli_provider import OpenCodeCLIProvider

pytestmark = pytest.mark.live


@pytest.fixture
async def opencode_provider(opencode_models):
    """Exercise the real CLI with declared model inputs."""
    provider = OpenCodeCLIProvider(
        model=opencode_models[0],
        fallback_model=opencode_models[1] if len(opencode_models) > 1 else None,
        timeout=30,
        max_retries=1,
    )
    yield provider


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_cli_command_success(opencode_provider):
    """Test successful CLI command execution with real subprocess."""
    result = await opencode_provider._run_cli_command("Say hello briefly")

    assert isinstance(result, str)
    assert len(result) > 0
    assert result.strip() == result
    assert '{"type":' not in result


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_cli_command_with_system(opencode_provider):
    """Test CLI command with system prompt returns clean text."""
    result = await opencode_provider._run_cli_command(
        "What is 2+2?", system="Answer with a single number."
    )

    assert isinstance(result, str)
    assert len(result) > 0
    assert result.strip() == result
    assert '{"type":' not in result
