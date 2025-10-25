import pytest


def test_codex_cli_provider_import_and_name():
    # Red test: module does not exist yet
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    provider = CodexCLIProvider(model="codex")
    assert provider.name == "codex-cli"


def test_codex_cli_estimate_tokens_ratio():
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider  # type: ignore[attr-defined]

    provider = CodexCLIProvider(model="codex")
    text = "x" * 400
    # Expect ~chars/4 tokens (like Claude CLI provider pattern)
    assert provider.estimate_tokens(text) == 100

