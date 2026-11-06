"""Integration tests for OpenCode CLI LLM provider.

The real ``opencode`` CLI subprocess is exercised but pointed at a local
loopback OpenAI-compatible stub server instead of a live provider. The test is
deterministic, reproducible, and offline while still covering the production
path: CLI invocation -> ``--format json`` NDJSON -> provider parsing.

Skipped automatically when the ``opencode`` CLI is not installed.
"""

import json
import shutil

import pytest

from chunkhound.providers.llm.opencode_cli_provider import OpenCodeCLIProvider
from tests.fixtures.openai_compatible_server import (
    ChatCompletionScript,
    OpenAICompatibleTestServer,
)

STUB_MODEL = "stub/test-model"
STUB_REPLY = "HELLO_FROM_STUB"
# Upper bound, not an assertion: opencode's `run` issues the main prompt plus
# an internal small_model call, and may add more across versions. A call past
# this cap gets a 400; assert_no_unmatched_requests turns an over-budget flake
# into a hard failure. Raise the cap if a new opencode version needs more.
_STUB_CALL_BUDGET = 8


@pytest.fixture
def opencode_stub(tmp_path, monkeypatch, request):
    """Start a loopback stub server and point the opencode CLI's config at it."""
    if not shutil.which("opencode"):
        pytest.skip("opencode CLI not installed")
    server = OpenAICompatibleTestServer(
        [
            ChatCompletionScript(
                name="stub-reply",
                # The stub answers every call (main prompt + small_model), so
                # match on message presence rather than a specific prompt.
                predicate=lambda _body, flattened: bool(flattened),
                content=STUB_REPLY,
                expected_calls=_STUB_CALL_BUDGET,
            )
        ]
    )
    server.start()
    request.addfinalizer(server.close)
    # OPENCODE_CONFIG adds this file to opencode's config resolution. It is
    # additive, not exclusive, so a user config may still be merged in; the
    # stub provider below is what keeps the run local. baseURL must end at /v1
    # (the SDK appends /chat/completions).
    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "stub": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "stub",
                "options": {"baseURL": server.base_url, "apiKey": "local"},
                "models": {"test-model": {"name": "test-model"}},
            }
        },
        "model": STUB_MODEL,
        "small_model": STUB_MODEL,
    }
    config_path = tmp_path / "opencode.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_path))
    # Isolate opencode's XDG state so the test never touches the user's
    # real ~/.local/share/opencode directory (it writes logs there).
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))
    provider = OpenCodeCLIProvider(model=STUB_MODEL, timeout=60, max_retries=1)
    yield provider, server


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_cli_command_success(opencode_stub):
    """Real CLI subprocess against a stub server returns clean extracted text."""
    provider, server = opencode_stub
    result = await provider._run_cli_command("Say hello briefly")

    assert result == STUB_REPLY
    assert '{"type":' not in result
    assert result.strip() == result
    assert server.requests, "the CLI subprocess must have called the stub server"
    server.assert_no_unmatched_requests()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_run_cli_command_with_system(opencode_stub):
    """System prompt is merged into the message the stub receives."""
    provider, server = opencode_stub
    result = await provider._run_cli_command(
        "What is 2+2?", system="Answer with a single number."
    )

    assert result == STUB_REPLY
    assert '{"type":' not in result
    # The provider merges system + user into a single stdin prompt, but opencode
    # may split the turn across several calls; join them so the assertion does
    # not depend on which call arrives first.
    flattened = "\n".join(request["flattened_messages"] for request in server.requests)
    assert "Answer with a single number." in flattened
    assert "What is 2+2?" in flattened
    server.assert_no_unmatched_requests()
