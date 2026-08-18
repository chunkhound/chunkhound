"""Contract tests for `Config.validate_for_command_structured`.

Locks each `ConfigErrorCode` enum member into a triggering configuration so
that silent renames, removals, or unwiring cause a test failure. The
remote-config terminal delta gate (planned in PR 2) depends on these codes
being stable comparison keys.

Also asserts that the string-only `validate_for_command` wrapper stays a
projection of `validate_for_command_structured` — an invariant relied on by
every existing caller that reads only the messages.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config, ConfigErrorCode


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip CHUNKHOUND_* env vars and redirect HOME so global configs
    and dev-shell env leakage cannot interfere with sub-config construction.
    """
    import os

    for key in list(os.environ):
        if key.startswith("CHUNKHOUND_") or key in {
            "VOYAGE_API_KEY",
            "OPENAI_API_KEY",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def _codes(config: Config, command: str) -> list[ConfigErrorCode]:
    return [code for code, _ in config.validate_for_command_structured(command)]


def test_missing_required_config_code(tmp_path: Path) -> None:
    # Embedding present but missing api_key → embedding.get_missing_config()
    # returns a single-item list → MISSING_REQUIRED_CONFIG once.
    config = Config(
        target_dir=tmp_path,
        embedding={"provider": "openai"},
    )
    assert _codes(config, "search").count(ConfigErrorCode.MISSING_REQUIRED_CONFIG) == 1


def test_llm_not_configured_code(tmp_path: Path) -> None:
    config = Config(target_dir=tmp_path, llm=None)
    assert _codes(config, "research").count(ConfigErrorCode.LLM_NOT_CONFIGURED) == 1


def test_llm_missing_role_config_code(tmp_path: Path) -> None:
    # Openai provider without api_key → get_missing_config_for_roles reports
    # api_key missing for utility/synthesis roles. The branch emits one tuple
    # per missing item, so we assert ≥1 rather than pinning to an internal
    # count that could shift if role names or dedup behaviour changes.
    config = Config(
        target_dir=tmp_path,
        llm={"provider": "openai"},
    )
    codes = _codes(config, "research")
    assert codes.count(ConfigErrorCode.LLM_MISSING_ROLE_CONFIG) >= 1


def test_embedding_not_configured_code(tmp_path: Path) -> None:
    config = Config(target_dir=tmp_path, embedding=None)
    assert _codes(config, "index").count(ConfigErrorCode.EMBEDDING_NOT_CONFIGURED) == 1


def test_mcp_non_loopback_no_auth_code(tmp_path: Path) -> None:
    config = Config(
        target_dir=tmp_path,
        mcp={"transport": "http", "host": "0.0.0.0"},
    )
    assert _codes(config, "mcp").count(ConfigErrorCode.MCP_NON_LOOPBACK_NO_AUTH) == 1


def test_mcp_cors_no_auth_code(tmp_path: Path) -> None:
    config = Config(
        target_dir=tmp_path,
        mcp={"transport": "http", "cors": True},
    )
    assert _codes(config, "mcp").count(ConfigErrorCode.MCP_CORS_NO_AUTH) == 1


def test_db_readonly_wrong_command_code(tmp_path: Path) -> None:
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db", "read_only": True},
    )
    assert _codes(config, "index").count(ConfigErrorCode.DB_READONLY_WRONG_COMMAND) == 1


def test_db_readonly_non_duckdb_code(tmp_path: Path) -> None:
    config = Config(
        target_dir=tmp_path,
        database={
            "path": tmp_path / "chunks.db",
            "read_only": True,
            "provider": "lancedb",
        },
    )
    assert _codes(config, "mcp").count(ConfigErrorCode.DB_READONLY_NON_DUCKDB) == 1


def test_string_wrapper_projects_structured_messages(tmp_path: Path) -> None:
    # Pick one triggering config and verify the string wrapper output is
    # exactly the projection of the structured output. This locks the
    # no-behavior-change contract PR 2 depends on.
    config = Config(
        target_dir=tmp_path,
        mcp={"transport": "http", "host": "0.0.0.0"},
    )
    structured = config.validate_for_command_structured("mcp")
    strings = config.validate_for_command("mcp")
    assert strings == [msg for _, msg in structured]
