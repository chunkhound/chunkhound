"""Contract tests for MCP HTTP transport startup validation.

Covers user-visible invariants:
  (a) Config.validate_for_command rejects mcp.cors=True with no auth_token.
  (b) Setting auth_token alongside mcp.cors=True passes validation.
  (c) Config.validate_for_command rejects a non-loopback mcp.host with no
      auth_token.
"""

from chunkhound.core.config.config import Config


def test_validator_rejects_cors_without_auth_token(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db"},
        mcp={"transport": "http", "cors": True},
    )
    errors = config.validate_for_command("mcp")
    assert any("mcp.cors is enabled but no auth_token is set" in e for e in errors), (
        errors
    )


def test_validator_accepts_cors_with_auth_token(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db"},
        mcp={"transport": "http", "cors": True, "auth_token": "secret123"},
    )
    errors = config.validate_for_command("mcp")
    assert not any("cors" in e.lower() for e in errors), errors


def test_validator_rejects_nonlocal_host_without_auth_token(tmp_path):
    config = Config(
        target_dir=tmp_path,
        database={"path": tmp_path / "chunks.db"},
        mcp={"transport": "http", "host": "0.0.0.0"},
    )
    errors = config.validate_for_command("mcp")
    assert any("non-loopback" in e for e in errors), errors
