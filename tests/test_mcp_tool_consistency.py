"""Test consistency of tool descriptions in the MCP server.

This test ensures the MCP stdio server exposes correct tool metadata from TOOL_REGISTRY,
preventing issues where tools have incorrect or missing descriptions.
"""

import inspect
import json

import pytest

from chunkhound.mcp_server.common import (
    first_error_tool_content,
    is_error_tool_content,
    tool_call_failed,
)
from chunkhound.mcp_server.tools import TOOL_REGISTRY, tool_requires_db
from chunkhound.providers.database.serial_executor import (
    DatabaseCompactionInProgressError,
)
from chunkhound.version import __version__


def _tool_error_payload(text: str) -> dict[str, str]:
    """Decode the public MCP error payload carried in text content."""
    return json.loads(text)["error"]


def test_tool_registry_populated():
    """Verify that TOOL_REGISTRY is populated by decorators."""
    assert len(TOOL_REGISTRY) > 0, "TOOL_REGISTRY should contain tools"

    # Check expected tools are present
    expected_tools = [
        "search",
        "daemon_status",
        "code_research",
        "websearch",
    ]
    for tool_name in expected_tools:
        assert tool_name in TOOL_REGISTRY, f"Tool '{tool_name}' should be in registry"

    # Verify old tools are removed
    removed_tools = ["get_stats", "health_check", "search_regex", "search_semantic"]
    for tool_name in removed_tools:
        assert tool_name not in TOOL_REGISTRY, f"Tool '{tool_name}' should be removed"


def test_tool_descriptions_not_empty():
    """Verify all tools have non-empty descriptions."""
    for tool_name, tool in TOOL_REGISTRY.items():
        assert tool.description, f"Tool '{tool_name}' should have a description"
        # All tools should have comprehensive descriptions
        assert len(tool.description) > 50, (
            f"Tool '{tool_name}' description should be comprehensive (>50 chars)"
        )


def test_tool_parameters_structure():
    """Verify all tools have properly structured parameter schemas."""
    for tool_name, tool in TOOL_REGISTRY.items():
        assert "type" in tool.parameters, (
            f"Tool '{tool_name}' parameters should have 'type'"
        )
        assert tool.parameters["type"] == "object", (
            f"Tool '{tool_name}' parameters type should be 'object'"
        )
        assert "properties" in tool.parameters, (
            f"Tool '{tool_name}' should have 'properties'"
        )


def test_search_schema():
    """Verify unified search has correct schema from decorator."""
    tool = TOOL_REGISTRY["search"]

    # Check description mentions both search types
    assert "regex" in tool.description.lower()
    assert "semantic" in tool.description.lower()

    # Check parameters
    props = tool.parameters["properties"]
    assert "type" in props, "search should have 'type' parameter"
    assert "query" in props, "search should have 'query' parameter"
    assert "page_size" in props, "search should have 'page_size' parameter"
    assert "offset" in props, "search should have 'offset' parameter"
    assert "path" in props, "search should have 'path' parameter"

    # Check required fields
    required = tool.parameters.get("required", [])
    assert "type" in required, "'type' should be required for search"
    assert "query" in required, "'query' should be required for search"


def test_code_research_schema():
    """Verify code_research has correct schema from decorator."""
    tool = TOOL_REGISTRY["code_research"]

    # Check description
    assert (
        "architecture" in tool.description.lower()
        or "analysis" in tool.description.lower()
    )
    assert len(tool.description) > 100, (
        "code_research should have comprehensive description"
    )

    # Check parameters
    props = tool.parameters["properties"]
    assert "query" in props, "code_research should have 'query' parameter"
    assert "max_depth" not in props, "code_research should not expose 'max_depth'"

    # Check required fields
    required = tool.parameters.get("required", [])
    assert "query" in required, "'query' should be required for code_research"


def test_daemon_status_schema():
    """Verify daemon_status has a zero-arg runtime schema."""
    tool = TOOL_REGISTRY["daemon_status"]

    assert "status" in tool.description.lower()
    props = tool.parameters["properties"]
    required = tool.parameters.get("required", [])

    assert props == {}, "daemon_status should not expose infrastructure arguments"
    assert required == [], "daemon_status should not require client-supplied args"


def test_capability_flags():
    """Verify tools correctly declare capability requirements."""
    # search: no special requirements (validates embedding at runtime)
    assert not TOOL_REGISTRY["search"].requires_embeddings
    assert not TOOL_REGISTRY["search"].requires_llm
    assert not TOOL_REGISTRY["search"].requires_reranker

    assert not TOOL_REGISTRY["daemon_status"].requires_embeddings
    assert not TOOL_REGISTRY["daemon_status"].requires_llm
    assert not TOOL_REGISTRY["daemon_status"].requires_reranker

    # code_research: requires all capabilities
    assert TOOL_REGISTRY["code_research"].requires_embeddings
    assert TOOL_REGISTRY["code_research"].requires_llm
    assert TOOL_REGISTRY["code_research"].requires_reranker

    # websearch: research stage needs the same three capabilities
    assert TOOL_REGISTRY["websearch"].requires_embeddings
    assert TOOL_REGISTRY["websearch"].requires_llm
    assert TOOL_REGISTRY["websearch"].requires_reranker


def test_websearch_schema():
    """Verify websearch has correct schema from decorator."""
    tool = TOOL_REGISTRY["websearch"]

    assert len(tool.description) > 100, (
        "websearch should have comprehensive description"
    )

    props = tool.parameters["properties"]
    assert "query" in props, "websearch should have 'query' parameter"
    assert "limit" in props, "websearch should have 'limit' parameter"
    assert "path_filter" not in props, (
        "websearch must not expose 'path_filter' — fetched pages live in a flat "
        "tmpdir, so a subdirectory filter would silently zero out results"
    )

    # limit default matches spec §4.2
    assert props["limit"].get("default") == 30

    required = tool.parameters.get("required", [])
    assert "query" in required, "'query' should be required for websearch"
    assert "limit" not in required, "'limit' should not be required (has default)"


def test_websearch_hidden_without_capabilities():
    """Verify websearch is filtered out when providers are missing.

    Calls _build_filtered_tool_dicts directly against a minimal stub — it only
    reads embedding_manager/llm_manager off self and delegates the reranker
    check to has_reranker_support, which already tolerates None. Avoids the
    full StdioMCPServer construction path (logger shaping, asyncio.Event,
    tool registration), which would silently absorb a MagicMock config and
    mask future initializer changes.
    """
    from types import SimpleNamespace

    from chunkhound.mcp_server.base import MCPServerBase

    stub = SimpleNamespace(embedding_manager=None, llm_manager=None)
    tool_dicts = MCPServerBase._build_filtered_tool_dicts(stub)  # type: ignore[arg-type]
    tool_names = [d["name"] for d in tool_dicts]

    assert "websearch" not in tool_names, (
        f"websearch should be hidden without capabilities, got {tool_names}"
    )


@pytest.mark.asyncio
async def test_daemon_status_tool_returns_scan_progress_snapshot():
    """Verify daemon_status returns the shared base scan_progress payload."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "ready"
    assert result["server_version"] == __version__
    assert result["query_ready"] is True
    assert result["scan_progress"]["realtime"]["service_state"] == "running"


@pytest.mark.asyncio
async def test_daemon_status_tool_degrades_on_realtime_state():
    """Verify daemon_status honors degraded realtime state even without scan_error."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "degraded",
            "last_error": None,
            "resync": {"last_error": None},
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "degraded"


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_watchman_realtime_details():
    """Verify daemon_status exposes stale Watchman state through the summary surface."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "configured_backend": "watchman",
            "effective_backend": "watchman",
            "monitoring_mode": "watchman",
            "service_state": "running",
            "watchman_sidecar_state": "running",
            "watchman_connection_state": "connected",
            "live_indexing_state": "idle",
            "live_indexing_hint": "Live indexing is connected and idle.",
            "watchman_watch_root": "/repo",
            "watchman_relative_root": "packages/api",
            "watchman_subscription_names": ["chunkhound-live-indexing"],
            "watchman_subscription_count": 1,
            "watchman_subscription_pdu_dropped": 3,
            "watchman_scopes": [
                {
                    "subscription_name": "chunkhound-live-indexing",
                    "scope_kind": "primary",
                    "requested_path": "/repo/packages/api",
                    "watch_root": "/repo",
                    "relative_root": "packages/api",
                }
            ],
            "last_warning": "watchman recrawl observed",
            "last_error": None,
            "watchman_loss_of_sync": {
                "count": 2,
                "fresh_instance_count": 1,
                "recrawl_count": 1,
                "disconnect_count": 0,
                "translation_failure_count": 0,
                "subscription_pdu_dropped_count": 0,
                "last_reason": "recrawl",
                "last_at": "2026-03-08T00:00:04Z",
                "last_details": {"warning": "Recrawled this watch"},
            },
            "watchman_reconnect": {
                "state": "restored",
                "attempt_count": 1,
                "retry_delay_seconds": None,
                "last_started_at": "2026-03-08T00:00:03Z",
                "last_completed_at": "2026-03-08T00:00:04Z",
                "last_error": None,
                "last_result": "restored",
            },
            "resync": {
                "needs_resync": True,
                "in_progress": False,
                "last_reason": "realtime_loss_of_sync",
                "last_error": None,
            },
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:02Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/packages/api/app.py",
                "last_accepted_event_at": "2026-03-08T00:00:02Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/packages/api/app.py",
                "last_processing_started_at": "2026-03-08T00:00:03Z",
                "last_processing_started_path": "/repo/packages/api/app.py",
                "last_processing_completed_at": "2026-03-08T00:00:04Z",
                "last_processing_completed_path": "/repo/packages/api/app.py",
                "filtered_event_count": 1,
                "suppressed_duplicate_count": 2,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    realtime = result["scan_progress"]["realtime"]
    assert result["status"] == "degraded"
    assert result["query_ready"] is True
    assert realtime["configured_backend"] == "watchman"
    assert realtime["watchman_sidecar_state"] == "running"
    assert realtime["watchman_connection_state"] == "connected"
    assert realtime["watchman_subscription_count"] == 1
    assert realtime["watchman_subscription_names"] == ["chunkhound-live-indexing"]
    assert realtime["watchman_subscription_pdu_dropped"] == 3
    assert realtime["live_indexing_state"] == "idle"
    assert realtime["live_indexing_hint"] == "Live indexing is connected and idle."
    assert realtime["watchman_watch_root"] == "/repo"
    assert realtime["watchman_relative_root"] == "packages/api"
    assert realtime["watchman_scopes"] == [
        {
            "subscription_name": "chunkhound-live-indexing",
            "scope_kind": "primary",
            "requested_path": "/repo/packages/api",
            "watch_root": "/repo",
            "relative_root": "packages/api",
        }
    ]
    assert realtime["watchman_loss_of_sync"]["fresh_instance_count"] == 1
    assert realtime["watchman_loss_of_sync"]["recrawl_count"] == 1
    assert realtime["watchman_loss_of_sync"]["translation_failure_count"] == 0
    assert realtime["watchman_loss_of_sync"]["subscription_pdu_dropped_count"] == 0
    assert realtime["watchman_reconnect"]["state"] == "restored"
    assert realtime["watchman_reconnect"]["last_result"] == "restored"
    assert realtime["resync"]["needs_resync"] is True
    assert realtime["resync"]["last_reason"] == "realtime_loss_of_sync"
    assert realtime["pipeline"]["filtered_event_count"] == 1
    assert realtime["pipeline"]["suppressed_duplicate_count"] == 2
    assert realtime["pipeline"]["last_processing_completed_path"] == (
        "/repo/packages/api/app.py"
    )


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_startup_timing_breakdown():
    """Startup phase timing should surface without changing the top-level summary."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "startup": {
                "state": "completed",
                "mode": "daemon",
                "started_at": "2026-03-08T00:00:00Z",
                "completed_at": "2026-03-08T00:00:04Z",
                "exposure_ready_at": "2026-03-08T00:00:04Z",
                "total_duration_seconds": 4.0,
                "current_phase": None,
                "last_error": None,
                "phases": {
                    "initialize": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:00Z",
                        "completed_at": "2026-03-08T00:00:00Z",
                        "duration_seconds": 0.12,
                    },
                    "db_connect": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:00Z",
                        "completed_at": "2026-03-08T00:00:01Z",
                        "duration_seconds": 0.83,
                    },
                    "realtime_start": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:01Z",
                        "completed_at": "2026-03-08T00:00:03Z",
                        "duration_seconds": 2.0,
                    },
                    "startup_barrier": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:03Z",
                        "completed_at": "2026-03-08T00:00:03Z",
                        "duration_seconds": 0.01,
                    },
                    "daemon_publish": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:03Z",
                        "completed_at": "2026-03-08T00:00:04Z",
                        "duration_seconds": 1.04,
                    },
                    "watchman_sidecar_start": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:01Z",
                        "completed_at": "2026-03-08T00:00:02Z",
                        "duration_seconds": 1.2,
                    },
                    "watchman_watch_project": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:02Z",
                        "completed_at": "2026-03-08T00:00:02Z",
                        "duration_seconds": 0.3,
                    },
                    "watchman_scope_discovery": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:02Z",
                        "completed_at": "2026-03-08T00:00:02Z",
                        "duration_seconds": 0.2,
                    },
                    "watchman_subscription_setup": {
                        "state": "completed",
                        "started_at": "2026-03-08T00:00:02Z",
                        "completed_at": "2026-03-08T00:00:03Z",
                        "duration_seconds": 0.3,
                    },
                    "watchdog_setup": {
                        "state": "uninitialized",
                        "started_at": None,
                        "completed_at": None,
                        "duration_seconds": None,
                    },
                    "polling_setup": {
                        "state": "uninitialized",
                        "started_at": None,
                        "completed_at": None,
                        "duration_seconds": None,
                    },
                },
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    startup = result["scan_progress"]["realtime"]["startup"]
    assert result["status"] == "ready"
    assert result["query_ready"] is True
    assert startup["mode"] == "daemon"
    assert startup["exposure_ready_at"] == "2026-03-08T00:00:04Z"
    assert startup["phases"]["watchman_sidecar_start"]["duration_seconds"] == 1.2
    assert startup["phases"]["daemon_publish"]["duration_seconds"] == 1.04


@pytest.mark.asyncio
async def test_daemon_status_tool_degrades_stalled_pipeline_summary():
    """A stalled pipeline should degrade the top-level daemon summary."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "resync": {
                "needs_resync": False,
                "in_progress": False,
                "last_reason": None,
                "last_error": None,
            },
            "live_indexing_state": "stalled",
            "live_indexing_hint": (
                "Accepted events are queued but processing has not advanced in "
                "30s; inspect pipeline timestamps and processing_error_count."
            ),
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:01Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/app.py",
                "last_accepted_event_at": "2026-03-08T00:00:01Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/app.py",
                "last_processing_started_at": None,
                "last_processing_started_path": None,
                "last_processing_completed_at": None,
                "last_processing_completed_path": None,
                "filtered_event_count": 0,
                "suppressed_duplicate_count": 0,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "degraded"
    assert result["query_ready"] is True
    assert result["scan_progress"]["realtime"]["live_indexing_state"] == "stalled"


@pytest.mark.asyncio
async def test_daemon_status_tool_keeps_query_ready_after_realtime_failure():
    """Later realtime reconciliation failures should stay queryable."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "scan_error": (
            "Storage reconciliation cleanup failed: "
            "database invalidated during orphan cleanup"
        ),
        "realtime": {
            "service_state": "degraded",
            "last_error": (
                "Realtime resync failed: "
                "Storage reconciliation cleanup failed: "
                "database invalidated during orphan cleanup"
            ),
            "resync": {
                "needs_resync": True,
                "in_progress": False,
                "last_reason": "realtime_loss_of_sync",
                "last_error": (
                    "Storage reconciliation cleanup failed: "
                    "database invalidated during orphan cleanup"
                ),
            },
            "live_indexing_state": "degraded",
            "live_indexing_hint": (
                "Live indexing remains degraded after reconciliation failure; "
                "inspect resync.last_error."
            ),
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "degraded"
    assert result["query_ready"] is True
    assert (
        "Storage reconciliation cleanup failed" in result["scan_progress"]["scan_error"]
    )
    assert (
        "Storage reconciliation cleanup failed"
        in result["scan_progress"]["realtime"]["resync"]["last_error"]
    )


@pytest.mark.asyncio
async def test_daemon_status_tool_keeps_initial_scan_failure_unqueryable():
    """Initial scan failure should still leave daemon_status unqueryable."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 0,
        "chunks_created": 0,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": None,
        "scan_error": "Initial directory scan failed: database unavailable",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "resync": {
                "needs_resync": False,
                "in_progress": False,
                "last_reason": None,
                "last_error": None,
            },
            "live_indexing_state": "idle",
            "live_indexing_hint": "Live indexing is connected and idle.",
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "degraded"
    assert result["query_ready"] is False
    assert result["scan_progress"]["scan_error"] == (
        "Initial directory scan failed: database unavailable"
    )


@pytest.mark.asyncio
async def test_daemon_status_tool_keeps_query_ready_during_post_bootstrap_scan():
    """Post-bootstrap scans should not clear query readiness."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": True,
        "scan_started_at": "2026-04-01T00:00:10",
        "query_ready_at": "2026-04-01T00:00:05",
        "scan_error": None,
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "resync": {
                "needs_resync": False,
                "in_progress": True,
                "last_reason": "realtime_loss_of_sync",
                "last_error": None,
            },
            "live_indexing_state": "busy",
            "live_indexing_hint": "Live indexing is applying a reconciliation scan.",
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "ready"
    assert result["query_ready"] is True
    assert result["scan_progress"]["is_scanning"] is True


@pytest.mark.asyncio
async def test_daemon_status_tool_keeps_initial_scan_in_progress_unqueryable():
    """Initial bootstrap scans should remain unqueryable until the first success."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 1,
        "chunks_created": 2,
        "is_scanning": True,
        "scan_started_at": "2026-04-01T00:00:00",
        "query_ready_at": None,
        "scan_error": None,
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "resync": {
                "needs_resync": False,
                "in_progress": False,
                "last_reason": None,
                "last_error": None,
            },
            "live_indexing_state": "busy",
            "live_indexing_hint": "Live indexing is applying the initial scan.",
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    assert result["status"] == "initializing"
    assert result["query_ready"] is False


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_pending_mutation_backlog_details():
    """Pending mutation composition should be visible through daemon_status."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "pending_files": 3,
            "pending_mutations": {
                "total": 4,
                "unique_paths": 3,
                "counts_by_operation": {
                    "change": 2,
                    "delete": 1,
                    "embed": 1,
                    "dir_delete": 0,
                    "dir_index": 0,
                },
                "retry_counts_by_operation": {
                    "change": 0,
                    "delete": 1,
                    "embed": 0,
                    "dir_delete": 0,
                    "dir_index": 0,
                },
                "retrying_mutations": 1,
                "oldest_pending_at": "2026-03-08T00:00:01Z",
                "oldest_pending_age_seconds": 37,
                "oldest_pending_operation": "delete",
                "oldest_pending_path": "/repo/retry_delete.py",
                "oldest_pending_retry_count": 1,
                "recovery_phase": "mutation_drain",
                "resync_reason": None,
            },
            "resync": {
                "needs_resync": False,
                "in_progress": False,
                "last_reason": None,
                "last_error": None,
            },
            "live_indexing_state": "busy",
            "live_indexing_hint": "Live indexing is actively processing changes.",
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:01Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/retry_delete.py",
                "last_accepted_event_at": "2026-03-08T00:00:01Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/retry_delete.py",
                "last_processing_started_at": "2026-03-08T00:00:02Z",
                "last_processing_started_path": "/repo/retry_delete.py",
                "last_processing_completed_at": "2026-03-08T00:00:03Z",
                "last_processing_completed_path": "/repo/retry_delete.py",
                "filtered_event_count": 0,
                "suppressed_duplicate_count": 0,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    realtime = result["scan_progress"]["realtime"]
    assert result["status"] == "ready"
    assert result["query_ready"] is True
    assert realtime["live_indexing_state"] == "busy"
    assert realtime["pending_files"] == 3
    assert realtime["pending_mutations"]["total"] == 4
    assert realtime["pending_mutations"]["unique_paths"] == 3
    assert realtime["pending_mutations"]["counts_by_operation"]["change"] == 2
    assert realtime["pending_mutations"]["counts_by_operation"]["delete"] == 1
    assert realtime["pending_mutations"]["retry_counts_by_operation"]["delete"] == 1
    assert realtime["pending_mutations"]["retrying_mutations"] == 1
    assert realtime["pending_mutations"]["oldest_pending_operation"] == "delete"
    assert realtime["pending_mutations"]["oldest_pending_path"] == (
        "/repo/retry_delete.py"
    )
    assert realtime["pending_mutations"]["recovery_phase"] == "mutation_drain"
    assert realtime["pending_mutations"]["resync_reason"] is None


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_hot_path_event_pressure():
    """Hot-path pressure detail should surface through daemon_status unchanged."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "live_indexing_state": "busy",
            "live_indexing_hint": "Live indexing is actively processing changes.",
            "event_pressure": {
                "state": "elevated",
                "sample_path": "/repo/generated/build.log",
                "sample_scope": "excluded",
                "sample_event_type": "modified",
                "events_in_window": 42,
                "coalesced_updates": 0,
                "window_seconds": 30.0,
                "last_observed_at": "2026-03-08T00:00:04Z",
            },
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:04Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/generated/build.log",
                "last_accepted_event_at": "2026-03-08T00:00:01Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/app.py",
                "last_processing_started_at": "2026-03-08T00:00:03Z",
                "last_processing_started_path": "/repo/app.py",
                "last_processing_completed_at": "2026-03-08T00:00:03Z",
                "last_processing_completed_path": "/repo/app.py",
                "filtered_event_count": 42,
                "suppressed_duplicate_count": 0,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    realtime = result["scan_progress"]["realtime"]
    assert result["status"] == "ready"
    assert result["query_ready"] is True
    assert realtime["event_pressure"]["state"] == "elevated"
    assert realtime["event_pressure"]["sample_path"] == "/repo/generated/build.log"
    assert realtime["event_pressure"]["sample_scope"] == "excluded"
    assert realtime["event_pressure"]["sample_event_type"] == "modified"
    assert realtime["event_pressure"]["events_in_window"] == 42
    assert realtime["event_pressure"]["coalesced_updates"] == 0
    assert realtime["event_pressure"]["window_seconds"] == 30.0


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_event_queue_overflow_reconciling_payload():
    """Overflow-burst status should be visible through the public daemon_status tool."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "running",
            "last_error": None,
            "live_indexing_state": "degraded",
            "live_indexing_hint": (
                "Live indexing is reconciling after internal event queue "
                "overflow; inspect event_queue.overflow and resync.last_reason."
            ),
            "event_queue": {
                "size": 1000,
                "maxsize": 1000,
                "accepted": 1007,
                "dropped": 2097,
                "last_reason": "queue_full",
                "last_event_type": "modified",
                "last_file_path": "/repo/overflow.py",
                "last_enqueued_at": "2026-03-08T00:00:04Z",
                "last_dropped_at": "2026-03-08T00:00:04Z",
                "overflow": {
                    "state": "reconciling",
                    "burst_count": 1,
                    "current_burst_dropped": 2097,
                    "last_burst_dropped": 0,
                    "last_started_at": "2026-03-08T00:00:04Z",
                    "last_cleared_at": None,
                    "sample_event_type": "modified",
                    "sample_file_path": "/repo/overflow.py",
                },
            },
            "resync": {
                "needs_resync": True,
                "in_progress": False,
                "last_reason": "event_queue_overflow",
                "last_error": None,
            },
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:04Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/overflow.py",
                "last_accepted_event_at": "2026-03-08T00:00:04Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/overflow.py",
                "last_processing_started_at": None,
                "last_processing_started_path": None,
                "last_processing_completed_at": None,
                "last_processing_completed_path": None,
                "filtered_event_count": 0,
                "suppressed_duplicate_count": 0,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    realtime = result["scan_progress"]["realtime"]
    assert result["status"] == "degraded"
    assert result["query_ready"] is True
    assert realtime["event_queue"]["overflow"]["state"] == "reconciling"
    assert realtime["event_queue"]["overflow"]["current_burst_dropped"] == 2097
    assert realtime["event_queue"]["overflow"]["sample_file_path"] == (
        "/repo/overflow.py"
    )
    assert realtime["resync"]["last_reason"] == "event_queue_overflow"
    assert realtime["live_indexing_hint"] == (
        "Live indexing is reconciling after internal event queue overflow; "
        "inspect event_queue.overflow and resync.last_reason."
    )


@pytest.mark.asyncio
async def test_daemon_status_tool_exposes_event_queue_overflow_failed_payload():
    """Failed overflow recovery should stay explicit through daemon_status."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "query_ready_at": "2026-03-08T00:00:05",
        "realtime": {
            "service_state": "degraded",
            "last_error": "No resync callback configured",
            "live_indexing_state": "degraded",
            "live_indexing_hint": (
                "Live indexing remains degraded after internal event queue "
                "overflow; inspect event_queue.overflow and resync.last_error."
            ),
            "event_queue": {
                "size": 1000,
                "maxsize": 1000,
                "accepted": 1007,
                "dropped": 2097,
                "last_reason": "queue_full",
                "last_event_type": "modified",
                "last_file_path": "/repo/overflow.py",
                "last_enqueued_at": "2026-03-08T00:00:04Z",
                "last_dropped_at": "2026-03-08T00:00:04Z",
                "overflow": {
                    "state": "failed",
                    "burst_count": 1,
                    "current_burst_dropped": 2097,
                    "last_burst_dropped": 2097,
                    "last_started_at": "2026-03-08T00:00:04Z",
                    "last_cleared_at": None,
                    "sample_event_type": "modified",
                    "sample_file_path": "/repo/overflow.py",
                },
            },
            "resync": {
                "needs_resync": True,
                "in_progress": False,
                "last_reason": "event_queue_overflow",
                "last_error": "No resync callback configured",
            },
            "pipeline": {
                "last_source_event_at": "2026-03-08T00:00:04Z",
                "last_source_event_type": "modified",
                "last_source_event_path": "/repo/overflow.py",
                "last_accepted_event_at": "2026-03-08T00:00:04Z",
                "last_accepted_event_type": "modified",
                "last_accepted_event_path": "/repo/overflow.py",
                "last_processing_started_at": None,
                "last_processing_started_path": None,
                "last_processing_completed_at": None,
                "last_processing_completed_path": None,
                "filtered_event_count": 0,
                "suppressed_duplicate_count": 0,
                "translation_error_count": 0,
                "processing_error_count": 0,
                "stall_threshold_seconds": 30.0,
            },
        },
    }

    result = await execute_tool(
        tool_name="daemon_status",
        services=None,
        embedding_manager=None,
        arguments={},
        scan_progress=scan_progress,
    )

    realtime = result["scan_progress"]["realtime"]
    assert result["status"] == "degraded"
    assert result["query_ready"] is True
    assert realtime["event_queue"]["overflow"]["state"] == "failed"
    assert realtime["event_queue"]["overflow"]["last_burst_dropped"] == 2097
    assert realtime["resync"]["last_error"] == "No resync callback configured"
    assert realtime["live_indexing_hint"] == (
        "Live indexing remains degraded after internal event queue overflow; "
        "inspect event_queue.overflow and resync.last_error."
    )


def test_stdio_server_uses_registry_descriptions():
    """Verify MCP server base imports and uses TOOL_REGISTRY for descriptions.

    This is a structural test - it ensures the shared filtering logic in
    MCPServerBase references TOOL_REGISTRY to prevent regression to hardcoded
    descriptions.  The filtering now lives in base.py (used by both the stdio
    server and the daemon), so that is the canonical place to check.
    """
    from pathlib import Path

    base_server_path = (
        Path(__file__).parent.parent / "chunkhound" / "mcp_server" / "base.py"
    )
    content = base_server_path.read_text()

    # Check that TOOL_REGISTRY is referenced in the shared base
    assert "TOOL_REGISTRY" in content, (
        "MCPServerBase should reference TOOL_REGISTRY for tool definitions"
    )


def test_default_values_in_schema():
    """Verify that default values are properly captured in schemas."""
    # search defaults
    search_props = TOOL_REGISTRY["search"].parameters["properties"]
    assert search_props["page_size"].get("default") == 10
    assert search_props["offset"].get("default") == 0


def test_no_duplicate_tool_dataclass():
    """Verify there's only one Tool dataclass definition in tools.py.

    Prevents regression where Tool was defined twice (once for decorator,
    once in old TOOL_DEFINITIONS approach).
    """
    from pathlib import Path

    tools_path = Path(__file__).parent.parent / "chunkhound" / "mcp_server" / "tools.py"
    content = tools_path.read_text()

    # Count occurrences of "@dataclass\nclass Tool:"
    import re

    matches = re.findall(r"@dataclass\s+class Tool:", content)
    assert len(matches) == 1, "There should be exactly one Tool dataclass definition"


def test_no_tool_definitions_list():
    """Verify old TOOL_DEFINITIONS list has been removed.

    The old pattern was:
        TOOL_DEFINITIONS = [Tool(...), Tool(...), ...]

    This should no longer exist since we use the @register_tool decorator.
    """
    from pathlib import Path

    tools_path = Path(__file__).parent.parent / "chunkhound" / "mcp_server" / "tools.py"
    content = tools_path.read_text()

    # Check that TOOL_DEFINITIONS list doesn't exist
    assert "TOOL_DEFINITIONS = [" not in content, (
        "Old TOOL_DEFINITIONS list should be removed "
        "(registry now populated by decorators)"
    )


@pytest.mark.asyncio
async def test_non_db_tools_do_not_trigger_provider_connect() -> None:
    """daemon_status/websearch must not conflate MCP sessions with DB opens."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    connect_provider = AsyncMock()
    server._connect_provider = connect_provider

    await server.ensure_tool_services("daemon_status")
    await server.ensure_tool_services("websearch")

    connect_provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_db_tools_still_trigger_provider_connect() -> None:
    """DB-backed tools still require the authoritative provider instance."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    connect_provider = AsyncMock()
    server._connect_provider = connect_provider

    await server.ensure_tool_services("search")

    connect_provider.assert_awaited_once()


@pytest.mark.asyncio
async def test_db_tools_surface_explicit_compaction_error() -> None:
    """DB-backed MCP tools must surface compaction-busy errors explicitly."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    server._connect_provider = AsyncMock(
        side_effect=DatabaseCompactionInProgressError("busy")
    )

    with pytest.raises(DatabaseCompactionInProgressError, match="busy"):
        await server.ensure_tool_services("search")


@pytest.mark.asyncio
async def test_unknown_stdio_tool_returns_unknown_without_connect_attempt() -> None:
    """Unknown stdio MCP tools must fail before any DB reconnect attempt."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from mcp.types import CallToolRequest

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    server._initialization_complete.set()
    server.ensure_tool_services = AsyncMock(
        side_effect=DatabaseCompactionInProgressError("busy")
    )

    handler = server.server.request_handlers[CallToolRequest]
    response = await handler(
        CallToolRequest(
            method="tools/call",
            params={"name": "__missing_tool__", "arguments": {}},
            id=1,
        )
    )

    server.ensure_tool_services.assert_not_awaited()
    assert response.root.isError is True
    payload = _tool_error_payload(response.root.content[0].text)
    assert payload["message"] == "Unknown tool: __missing_tool__"


@pytest.mark.asyncio
async def test_unknown_daemon_tool_returns_unknown_without_connect_attempt() -> None:
    """Unknown daemon MCP tools must fail before any DB reconnect attempt."""
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from chunkhound.daemon.server import ChunkHoundDaemon

    daemon = ChunkHoundDaemon(
        config=MagicMock(),
        args=MagicMock(),
        socket_path="",
        project_dir=Path("/tmp"),
    )
    daemon.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    daemon._initialization_complete.set()
    daemon.ensure_tool_services = AsyncMock(
        side_effect=DatabaseCompactionInProgressError("busy")
    )

    response = await daemon._handle_tools_call(
        {
            "id": "1",
            "params": {"name": "__missing_tool__", "arguments": {}},
        }
    )

    daemon.ensure_tool_services.assert_not_awaited()
    assert response["result"]["isError"] is True
    payload = _tool_error_payload(response["result"]["content"][0]["text"])
    assert payload["message"] == "Unknown tool: __missing_tool__"


@pytest.mark.asyncio
async def test_daemon_handle_tools_call_skips_connect_for_non_db_tools() -> None:
    """Daemon _handle_tools_call must not trigger DB connect for daemon_status."""
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from chunkhound.daemon.server import ChunkHoundDaemon

    daemon = ChunkHoundDaemon(
        config=MagicMock(),
        args=MagicMock(),
        socket_path="",
        project_dir=Path("/tmp"),
    )
    daemon.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    connect_mock = AsyncMock()
    daemon._connect_provider = connect_mock

    msg = {
        "id": "1",
        "params": {"name": "daemon_status", "arguments": {}},
    }
    with patch(
        "chunkhound.daemon.server.handle_tool_call",
        new_callable=AsyncMock,
        return_value=[],
    ):
        await daemon._handle_tools_call(msg)

    connect_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_stdio_tool_call_surfaces_compaction_busy_payload() -> None:
    """Stdio MCP clients should receive the serialized compaction error payload."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from mcp.types import CallToolRequest

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    server._initialization_complete.set()
    server.ensure_tool_services = AsyncMock(
        side_effect=DatabaseCompactionInProgressError("busy")
    )

    handler = server.server.request_handlers[CallToolRequest]
    response = await handler(
        CallToolRequest(
            method="tools/call",
            params={"name": "search", "arguments": {"type": "regex", "query": "x"}},
            id=1,
        )
    )

    assert response.root.isError is True
    payload = _tool_error_payload(response.root.content[0].text)
    assert payload == {
        "type": "DatabaseCompactionInProgressError",
        "message": "busy",
    }


@pytest.mark.asyncio
async def test_stdio_tool_call_surfaces_generic_tool_failure_payload() -> None:
    """Stdio MCP clients should receive serialized unexpected tool failures."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from mcp.types import CallToolRequest

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server._initialization_complete.set()

    handler = server.server.request_handlers[CallToolRequest]
    with patch(
        "chunkhound.mcp_server.common.execute_tool",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        response = await handler(
            CallToolRequest(
                method="tools/call",
                params={"name": "daemon_status", "arguments": {}},
                id=1,
            )
        )

    assert response.root.isError is True
    payload = _tool_error_payload(response.root.content[0].text)
    assert payload == {"type": "RuntimeError", "message": "boom"}


@pytest.mark.asyncio
async def test_stdio_tool_call_uses_error_content_when_success_precedes_error() -> None:
    """Stdio transport must raise the actual error-bearing content item."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from mcp.types import CallToolRequest, TextContent

    from chunkhound.mcp_server.stdio import StdioMCPServer

    server = StdioMCPServer(config=MagicMock())
    server._initialization_complete.set()

    mixed_contents = [
        TextContent(type="text", text=json.dumps({"status": "ok"})),
        TextContent(
            type="text",
            text=json.dumps(
                {"error": {"type": "RuntimeError", "message": "second"}}
            ),
        ),
    ]

    handler = server.server.request_handlers[CallToolRequest]
    with patch(
        "chunkhound.mcp_server.stdio.handle_tool_call",
        new=AsyncMock(return_value=mixed_contents),
    ):
        response = await handler(
            CallToolRequest(
                method="tools/call",
                params={"name": "daemon_status", "arguments": {}},
                id=1,
            )
        )

    assert response.root.isError is True
    payload = _tool_error_payload(response.root.content[0].text)
    assert payload == {"type": "RuntimeError", "message": "second"}


@pytest.mark.asyncio
async def test_daemon_tool_call_marks_compaction_busy_as_error() -> None:
    """Daemon MCP clients should see compaction-busy tool failures as errors."""
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from chunkhound.daemon.server import ChunkHoundDaemon

    daemon = ChunkHoundDaemon(
        config=MagicMock(),
        args=MagicMock(),
        socket_path="",
        project_dir=Path("/tmp"),
    )
    daemon.services = SimpleNamespace(provider=SimpleNamespace(is_connected=False))
    daemon._initialization_complete.set()
    daemon.ensure_tool_services = AsyncMock(
        side_effect=DatabaseCompactionInProgressError("busy")
    )

    response = await daemon._handle_tools_call(
        {
            "id": "1",
            "params": {"name": "search", "arguments": {"type": "regex", "query": "x"}},
        }
    )

    assert response["result"]["isError"] is True
    payload = _tool_error_payload(response["result"]["content"][0]["text"])
    assert payload == {
        "type": "DatabaseCompactionInProgressError",
        "message": "busy",
    }


@pytest.mark.asyncio
async def test_daemon_tool_call_marks_generic_tool_failure_as_error() -> None:
    """Daemon MCP clients should see unexpected tool exceptions as errors."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from chunkhound.daemon.server import ChunkHoundDaemon

    daemon = ChunkHoundDaemon(
        config=MagicMock(),
        args=MagicMock(),
        socket_path="",
        project_dir=Path("/tmp"),
    )
    daemon._initialization_complete.set()

    with patch(
        "chunkhound.mcp_server.common.execute_tool",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        response = await daemon._handle_tools_call(
            {
                "id": "1",
                "params": {"name": "daemon_status", "arguments": {}},
            }
        )

    assert response["result"]["isError"] is True
    payload = _tool_error_payload(response["result"]["content"][0]["text"])
    assert payload == {"type": "RuntimeError", "message": "boom"}


@pytest.mark.asyncio
async def test_daemon_tool_call_propagates_mixed_content_error_flag() -> None:
    """Daemon transport must preserve mixed content while flagging later errors."""
    from pathlib import Path
    from unittest.mock import AsyncMock, MagicMock, patch

    from mcp.types import TextContent

    from chunkhound.daemon.server import ChunkHoundDaemon

    daemon = ChunkHoundDaemon(
        config=MagicMock(),
        args=MagicMock(),
        socket_path="",
        project_dir=Path("/tmp"),
    )
    daemon._initialization_complete.set()

    mixed_contents = [
        TextContent(type="text", text=json.dumps({"status": "ok"})),
        TextContent(
            type="text",
            text=json.dumps(
                {"error": {"type": "RuntimeError", "message": "second"}}
            ),
        ),
    ]

    with patch(
        "chunkhound.daemon.server.handle_tool_call",
        new=AsyncMock(return_value=mixed_contents),
    ):
        response = await daemon._handle_tools_call(
            {
                "id": "1",
                "params": {"name": "daemon_status", "arguments": {}},
            }
        )

    assert response["result"]["isError"] is True
    assert len(response["result"]["content"]) == 2
    payload = _tool_error_payload(response["result"]["content"][1]["text"])
    assert payload == {"type": "RuntimeError", "message": "second"}


@pytest.mark.parametrize(
    "tool_name,expected",
    [
        (name, "services" in inspect.signature(tool.implementation).parameters)
        for name, tool in TOOL_REGISTRY.items()
    ],
)
def test_tool_requires_db_matches_signature(
    tool_name: str, expected: bool
) -> None:
    """Every tool's DB-service requirement must match its implementation signature."""
    assert tool_requires_db(tool_name) is expected


@pytest.mark.parametrize(
    "tool_name",
    list(TOOL_REGISTRY.keys()),
)
def test_requires_db_field_matches_implementation(tool_name: str) -> None:
    """The declarative requires_db field must match the implementation signature."""
    tool = TOOL_REGISTRY[tool_name]
    has_services = "services" in inspect.signature(tool.implementation).parameters
    assert tool.requires_db is has_services, (
        f"Tool '{tool_name}': requires_db={tool.requires_db} "
        f"but implementation {'has' if has_services else 'lacks'} "
        f"a 'services' parameter"
    )


def test_tool_requires_db_returns_true_for_unknown_tool() -> None:
    """Unknown tool names must conservatively assume DB is needed."""
    assert tool_requires_db("__nonexistent_tool__") is True


def test_search_enum_restricted_without_embeddings():
    """Verify search type enum is restricted to regex when embeddings unavailable.

    This tests the dynamic schema mutation in build_available_tools() that restricts
    the search type to only ["regex"] when no embedding provider is available.
    """
    from unittest.mock import MagicMock

    from chunkhound.mcp_server.stdio import StdioMCPServer
    from chunkhound.mcp_server.tools import TOOL_REGISTRY

    # Create server with mocked config (build_available_tools doesn't use config)
    mock_config = MagicMock()
    mock_config.debug = False
    server = StdioMCPServer(config=mock_config)

    # Ensure no embedding/llm managers (already None from base class)
    assert server.embedding_manager is None
    assert server.llm_manager is None

    # Call actual server method
    tools = server.build_available_tools()

    # Find the search tool
    search_tool = next((t for t in tools if t.name == "search"), None)
    assert search_tool is not None, "search tool should be in list"
    daemon_status_tool = next((t for t in tools if t.name == "daemon_status"), None)
    assert daemon_status_tool is not None, "daemon_status tool should be in list"

    # Verify the type enum is restricted to regex only
    type_schema = search_tool.inputSchema["properties"]["type"]
    assert type_schema["enum"] == ["regex"], (
        f"Expected ['regex'] without embeddings, got {type_schema['enum']}"
    )

    # Verify the original TOOL_REGISTRY was NOT mutated
    original_enum = TOOL_REGISTRY["search"].parameters["properties"]["type"]["enum"]
    assert "semantic" in original_enum, (
        "TOOL_REGISTRY should not be mutated - 'semantic' should still be in enum"
    )


def test_llm_capability_gating_with_registry_provider():
    """Verify that a registry provider (deepseek) without model hides LLM tools.

    DeepSeek is an OPENAI_COMPATIBLE_PROVIDER with no baked-in default model.
    When ``llm.provider`` is set to ``"deepseek"`` without an explicit
    ``llm.model``, ``LLMManager._create_openai_compatible_provider()`` raises
    ``ValueError`` because model is empty.  The exception is caught in
    ``MCPServerBase.initialize()``, leaving ``self.llm_manager = None``, which
    causes ``_build_filtered_tool_dicts`` to skip all ``requires_llm=True``
    tools (``code_research``, ``websearch``).

    This test validates the gating logic directly, without requiring a full
    server bootstrap, by constructing the provider config that triggers the
    failure path.
    """
    from chunkhound.core.config.llm_config import LLMConfig
    from chunkhound.llm_manager import LLMManager

    # ── Without model: ValueError during provider creation ──
    # Build the config dict directly (bypass ``build_provider_config`` which
    # would also catch this) to test the runtime factory error path.
    no_model_cfg = {
        "provider": "deepseek",
        "model": "",
        "api_key": "sk-test",
    }
    with pytest.raises(ValueError, match="Model is required for.*deepseek"):
        LLMManager(no_model_cfg, no_model_cfg)

    # ── With model: provider creation succeeds ──
    config_with_model = LLMConfig(
        provider="deepseek", model="deepseek-v4-flash", api_key="sk-test"
    )
    util_cfg, synth_cfg = config_with_model.get_provider_configs()

    assert util_cfg["model"] == "deepseek-v4-flash"
    assert synth_cfg["model"] == "deepseek-v4-flash"

    # Creation should not raise
    manager = LLMManager(util_cfg, synth_cfg)
    assert manager.is_configured()


def test_filtered_tool_dicts_hides_llm_tools_without_manager():
    """Verify ``_build_filtered_tool_dicts`` drops LLM-requiring tools when
    ``llm_manager`` is ``None`` (simulating the silent initialization failure).
    """
    from types import SimpleNamespace

    from chunkhound.mcp_server.base import MCPServerBase

    # Stub with no LLM manager and no embedding provider
    stub = SimpleNamespace(
        embedding_manager=None,
        llm_manager=None,
    )
    tool_dicts = MCPServerBase._build_filtered_tool_dicts(stub)
    names = {d["name"] for d in tool_dicts}

    # Tools with requires_llm=True must be absent
    assert "code_research" not in names
    assert "websearch" not in names

    # Tools without requires_llm must be present
    assert "search" in names, "search should be present (no LLM required)"
    assert "daemon_status" in names, "daemon_status should be present (no LLM required)"

    # search should have degraded description
    search_tool = next(d for d in tool_dicts if d["name"] == "search")
    assert "research" not in search_tool["description"].lower(), (
        "search description should be degraded without LLM"
    )


# ── MCP error content helpers ────────────────────────────────────────────


class TestIsErrorToolContent:
    """is_error_tool_content identifies serialized error payloads."""

    def test_valid_error_payload(self) -> None:
        """A JSON object with an 'error' dict is detected as an error."""
        text = json.dumps({"error": {"type": "RuntimeError", "message": "boom"}})
        assert is_error_tool_content(text) is True

    def test_non_error_json_object(self) -> None:
        """A JSON object without an 'error' key is not an error."""
        assert is_error_tool_content('{"status": "ok"}') is False

    def test_error_as_string_is_not_detected(self) -> None:
        """An 'error' key with a string value (not dict) is not an error."""
        assert is_error_tool_content('{"error": "boom"}') is False

    def test_malformed_json_is_not_detected(self) -> None:
        """Non-JSON text is not an error."""
        assert is_error_tool_content("not json") is False

    def test_empty_string_is_not_detected(self) -> None:
        """Empty string is not an error."""
        assert is_error_tool_content("") is False


class TestFirstErrorToolContent:
    """first_error_tool_content returns the first error-bearing TextContent."""

    def test_returns_first_error(self) -> None:
        """When multiple contents exist, the first error is returned."""
        from mcp.types import TextContent

        contents = [
            TextContent(type="text", text=json.dumps({"status": "ok"})),
            TextContent(
                type="text",
                text=json.dumps({"error": {"type": "RuntimeError", "message": "boom"}}),
            ),
        ]
        result = first_error_tool_content(contents)
        assert result is not None
        assert "boom" in result.text

    def test_returns_none_when_no_error(self) -> None:
        """When no content is an error, returns None."""
        from mcp.types import TextContent

        contents = [
            TextContent(type="text", text=json.dumps({"status": "ok"})),
        ]
        assert first_error_tool_content(contents) is None

    def test_returns_none_for_empty_list(self) -> None:
        """Empty list returns None."""
        assert first_error_tool_content([]) is None


class TestToolCallFailed:
    """tool_call_failed detects error-bearing content lists."""

    def test_returns_true_when_error_present(self) -> None:
        """A list with an error content returns True."""
        from mcp.types import TextContent

        contents = [
            TextContent(
                type="text",
                text=json.dumps({"error": {"type": "RuntimeError", "message": "boom"}}),
            ),
        ]
        assert tool_call_failed(contents) is True

    def test_returns_false_when_no_error(self) -> None:
        """A list with only success content returns False."""
        from mcp.types import TextContent

        contents = [
            TextContent(type="text", text=json.dumps({"status": "ok"})),
        ]
        assert tool_call_failed(contents) is False

    def test_returns_false_for_empty_list(self) -> None:
        """Empty list returns False."""
        assert tool_call_failed([]) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
