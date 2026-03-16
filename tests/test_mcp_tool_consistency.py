"""Test consistency of tool descriptions in the MCP server.

This test ensures the MCP stdio server exposes correct tool metadata from TOOL_REGISTRY,
preventing issues where tools have incorrect or missing descriptions.
"""

import pytest

from chunkhound.mcp_server.tools import TOOL_REGISTRY


def test_tool_registry_populated():
    """Verify that TOOL_REGISTRY is populated by decorators."""
    assert len(TOOL_REGISTRY) > 0, "TOOL_REGISTRY should contain tools"

    # Check expected tools are present
    expected_tools = [
        "search",
        "daemon_status",
        "code_research",
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


@pytest.mark.asyncio
async def test_daemon_status_tool_returns_scan_progress_snapshot():
    """Verify daemon_status returns the shared base scan_progress payload."""
    from chunkhound.mcp_server.tools import execute_tool

    scan_progress = {
        "files_processed": 3,
        "chunks_created": 9,
        "is_scanning": False,
        "scan_started_at": "2026-03-08T00:00:00",
        "scan_completed_at": "2026-03-08T00:00:05",
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
        "scan_completed_at": "2026-03-08T00:00:05",
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
        "scan_completed_at": "2026-03-08T00:00:05",
        "realtime": {
            "configured_backend": "watchman",
            "effective_backend": "watchman",
            "monitoring_mode": "watchman",
            "service_state": "running",
            "watchman_sidecar_state": "running",
            "watchman_connection_state": "connected",
            "watchman_watch_root": "/repo",
            "watchman_relative_root": "packages/api",
            "watchman_subscription_names": ["chunkhound-live-indexing"],
            "watchman_subscription_count": 1,
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
                "last_reason": "recrawl",
                "last_at": "2026-03-08T00:00:04Z",
                "last_details": {"warning": "Recrawled this watch"},
            },
            "watchman_reconnect": {
                "state": "restored",
                "attempt_count": 1,
                "max_attempts": 3,
                "retry_delay_seconds": 1.0,
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
    assert realtime["watchman_reconnect"]["state"] == "restored"
    assert realtime["watchman_reconnect"]["last_result"] == "restored"
    assert realtime["resync"]["needs_resync"] is True
    assert realtime["resync"]["last_reason"] == "realtime_loss_of_sync"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
