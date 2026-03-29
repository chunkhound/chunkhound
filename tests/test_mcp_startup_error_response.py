"""Test that _respond_with_startup_error sends a JSON-RPC error on stdout."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

from chunkhound.mcp_server.stdio import _respond_with_startup_error


def _patch_select(ready_list):
    """Patch select.select to return the given ready list."""
    return patch("select.select", return_value=(ready_list, [], []))


class TestRespondWithStartupError:
    """Verify JSON-RPC error response for startup failures."""

    def test_sends_jsonrpc_error_for_initialize_request(self, tmp_path):
        """Pipe a mock initialize request on stdin; assert JSON-RPC error on stdout."""
        init_request = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        fake_stdin = io.StringIO(init_request + "\n")
        fake_stdout = io.StringIO()

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.object(sys, "stdout", fake_stdout),
            _patch_select([fake_stdin]),
        ):
            _respond_with_startup_error(Exception("DB locked"))

        output = fake_stdout.getvalue().strip()
        assert output, "Expected JSON-RPC error on stdout"

        resp = json.loads(output)
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["error"]["code"] == -32000
        assert "DB locked" in resp["error"]["message"]

    def test_includes_hint_for_lock_errors(self, tmp_path):
        """Lock-related errors should include a hint in the message."""
        init_request = json.dumps(
            {"jsonrpc": "2.0", "id": 42, "method": "initialize", "params": {}}
        )
        fake_stdin = io.StringIO(init_request + "\n")
        fake_stdout = io.StringIO()

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.object(sys, "stdout", fake_stdout),
            _patch_select([fake_stdin]),
        ):
            _respond_with_startup_error(
                Exception("Could not set lock on database")
            )

        resp = json.loads(fake_stdout.getvalue().strip())
        assert "kill" in resp["error"]["message"]
        assert resp["id"] == 42

    def test_writes_log_file_with_traceback(self, tmp_path):
        """Error log should include traceback, not just str(error)."""
        log_path = tmp_path / "error.log"

        import builtins

        original_open = builtins.open

        def capturing_open(path, mode="r", **kwargs):
            if "error" in str(path) or str(path) == "/tmp/chunkhound_mcp_error.log":
                return original_open(str(log_path), mode, **kwargs)
            return original_open(path, mode, **kwargs)

        with (
            patch.object(sys, "stdin", io.StringIO("")),
            patch.object(sys, "stdout", io.StringIO()),
            _patch_select([]),
        ):
            # Simulate being inside an except block so format_exc has content
            try:
                raise ValueError("test traceback content")
            except ValueError as e:
                with patch("builtins.open", side_effect=capturing_open):
                    _respond_with_startup_error(e)

        log_content = log_path.read_text()
        assert "test traceback content" in log_content
        assert "Traceback" in log_content

    def test_log_file_uses_append_mode(self, tmp_path):
        """Log file should append, not truncate existing content."""
        log_path = tmp_path / "error.log"
        log_path.write_text("previous error entry\n")

        import builtins

        original_open = builtins.open

        def capturing_open(path, mode="r", **kwargs):
            if "error" in str(path) or str(path) == "/tmp/chunkhound_mcp_error.log":
                return original_open(str(log_path), mode, **kwargs)
            return original_open(path, mode, **kwargs)

        with (
            patch.object(sys, "stdin", io.StringIO("")),
            patch.object(sys, "stdout", io.StringIO()),
            _patch_select([]),
            patch("builtins.open", side_effect=capturing_open),
        ):
            _respond_with_startup_error(Exception("new error"))

        log_content = log_path.read_text()
        assert "previous error entry" in log_content
        assert "new error" in log_content

    def test_no_output_when_stdin_empty(self):
        """When no initialize request is pending, nothing should be written to stdout."""
        fake_stdout = io.StringIO()

        with (
            patch.object(sys, "stdin", io.StringIO("")),
            patch.object(sys, "stdout", fake_stdout),
            _patch_select([]),
        ):
            _respond_with_startup_error(Exception("some error"))

        assert fake_stdout.getvalue() == ""

    def test_ignores_non_initialize_requests(self):
        """Non-initialize JSON-RPC requests should not get a response."""
        other_request = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        fake_stdin = io.StringIO(other_request + "\n")
        fake_stdout = io.StringIO()

        with (
            patch.object(sys, "stdin", fake_stdin),
            patch.object(sys, "stdout", fake_stdout),
            _patch_select([fake_stdin]),
        ):
            _respond_with_startup_error(Exception("error"))

        assert fake_stdout.getvalue() == ""
