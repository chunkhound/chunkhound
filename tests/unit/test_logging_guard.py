"""Tests for chunkhound.providers.database.duckdb.logging_guard.log_if_not_mcp.

Verifies the MCP-safe logging guard suppresses output in MCP mode
(strict ``== "1"``) and emits normally otherwise.
"""

import io

import pytest
from loguru import logger

from chunkhound.utils.logging_guard import log_if_not_mcp


def test_suppresses_when_mcp_mode_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHUNKHOUND_MCP_MODE=1 must suppress all log output."""
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "1")
    buf = io.StringIO()
    sink_id = logger.add(buf, level="DEBUG")
    try:
        log_if_not_mcp("info", "should not appear")
        log_if_not_mcp("error", "should not appear either")
    finally:
        logger.remove(sink_id)
    assert buf.getvalue() == "", "MCP mode must suppress all log messages"


def test_emits_when_mcp_mode_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal mode (env var unset) must emit log messages."""
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)
    buf = io.StringIO()
    sink_id = logger.add(buf, level="DEBUG")
    try:
        log_if_not_mcp("info", "hello from test")
    finally:
        logger.remove(sink_id)
    assert "hello from test" in buf.getvalue()


def test_mcp_mode_zero_does_not_suppress(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHUNKHOUND_MCP_MODE=0 must NOT suppress — strict == '1' check."""
    monkeypatch.setenv("CHUNKHOUND_MCP_MODE", "0")
    buf = io.StringIO()
    sink_id = logger.add(buf, level="DEBUG")
    try:
        log_if_not_mcp("info", "visible in mode zero")
    finally:
        logger.remove(sink_id)
    assert "visible in mode zero" in buf.getvalue()


def test_level_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Level argument must be case-insensitive (lower and UPPER both work)."""
    monkeypatch.delenv("CHUNKHOUND_MCP_MODE", raising=False)
    buf = io.StringIO()
    sink_id = logger.add(buf, level="DEBUG")
    try:
        log_if_not_mcp("warning", "lower-case level")
        log_if_not_mcp("WARNING", "upper-case level")
    finally:
        logger.remove(sink_id)
    out = buf.getvalue()
    assert "lower-case level" in out
    assert "upper-case level" in out
