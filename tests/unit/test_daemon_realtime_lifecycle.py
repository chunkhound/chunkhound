"""Daemon status contract tests.

These assertions stay on the public daemon_status surface instead of private
MCPServerBase lifecycle bookkeeping.
"""

from chunkhound.mcp_server.status import derive_daemon_status


def test_status_is_initializing_before_first_successful_scan() -> None:
    result = derive_daemon_status(
        {
            "is_scanning": False,
            "realtime": {"service_state": "active"},
        }
    )

    assert result["status"] == "initializing"
    assert result["query_ready"] is False


def test_status_is_ready_after_successful_scan_without_realtime_errors() -> None:
    result = derive_daemon_status(
        {
            "is_scanning": False,
            "query_ready_at": "2026-01-01T00:00:00",
            "realtime": {"service_state": "active"},
        }
    )

    assert result["status"] == "ready"
    assert result["query_ready"] is True


def test_status_stays_degraded_when_scan_error_arrives_after_successful_scan() -> None:
    result = derive_daemon_status(
        {
            "is_scanning": False,
            "query_ready_at": "2026-01-01T00:00:00",
            "scan_error": "disk I/O error",
            "realtime": {"service_state": "active"},
        }
    )

    assert result["status"] == "degraded"
    assert result["query_ready"] is True


def test_status_is_degraded_when_realtime_needs_resync() -> None:
    result = derive_daemon_status(
        {
            "is_scanning": False,
            "query_ready_at": "2026-01-01T00:00:00",
            "realtime": {
                "service_state": "active",
                "resync": {"needs_resync": True},
            },
        }
    )

    assert result["status"] == "degraded"
    assert result["query_ready"] is True
