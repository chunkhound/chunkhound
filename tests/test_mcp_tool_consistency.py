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


def test_capability_flags():
    """Verify tools correctly declare capability requirements."""
    # search: no special requirements (validates embedding at runtime)
    assert not TOOL_REGISTRY["search"].requires_embeddings
    assert not TOOL_REGISTRY["search"].requires_llm
    assert not TOOL_REGISTRY["search"].requires_reranker

    # code_research: requires all capabilities
    assert TOOL_REGISTRY["code_research"].requires_embeddings
    assert TOOL_REGISTRY["code_research"].requires_llm
    assert TOOL_REGISTRY["code_research"].requires_reranker


def test_stdio_server_uses_registry_descriptions():
    """Verify stdio server imports and uses TOOL_REGISTRY for descriptions.

    This is a structural test - it ensures the stdio server code references
    TOOL_REGISTRY to prevent regression to hardcoded descriptions.
    """
    from pathlib import Path

    stdio_server_path = (
        Path(__file__).parent.parent / "chunkhound" / "mcp_server" / "stdio.py"
    )
    content = stdio_server_path.read_text()

    # Check that TOOL_REGISTRY is imported
    assert "from .tools import" in content and "TOOL_REGISTRY" in content, (
        "Stdio server should import TOOL_REGISTRY"
    )

    # Check that tools are registered from TOOL_REGISTRY
    # The server should iterate over TOOL_REGISTRY to expose tools
    assert "TOOL_REGISTRY" in content, (
        "Server should reference TOOL_REGISTRY for tool definitions"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
