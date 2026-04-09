"""Test that CompactionError produces a structured retry-hint response."""

from chunkhound.core.exceptions import CompactionError
from chunkhound.mcp_server.stdio import _compaction_error_response


def test_transient_compaction_error_has_retry_hint():
    """Transient compaction errors advise the client to retry."""
    exc = CompactionError("Compaction in progress", operation="connection")
    resp = _compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert "retry" in resp["error"]["retry_hint"].lower()


def test_unrecoverable_compaction_error_no_retry():
    """Unrecoverable errors advise restore/re-index, not retry."""
    exc = CompactionError(
        "Unrecoverable: no valid database or backup found",
        operation="recovery",
    )
    resp = _compaction_error_response(exc)
    assert resp["error"]["type"] == "CompactionError"
    assert "recovery failed" in resp["error"]["retry_hint"].lower()
    assert "retry in" not in resp["error"]["retry_hint"].lower()
