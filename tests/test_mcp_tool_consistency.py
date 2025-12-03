"""Test consistency of tool descriptions between HTTP and stdio MCP servers.

This test ensures that both server modes expose identical tool metadata,
preventing the issue where tools had different descriptions in different modes.
"""

import pytest

from chunkhound.mcp_server.tools import TOOL_REGISTRY


def test_tool_registry_populated():
    """Verify that TOOL_REGISTRY is populated by decorators."""
    assert len(TOOL_REGISTRY) > 0, "TOOL_REGISTRY should contain tools"

    # Check expected tools are present
    expected_tools = ["get_stats", "health_check", "search_regex", "search_semantic", "code_research"]
    for tool_name in expected_tools:
        assert tool_name in TOOL_REGISTRY, f"Tool '{tool_name}' should be in registry"


def test_tool_descriptions_not_empty():
    """Verify all tools have non-empty descriptions."""
    # Simple utility tools can have shorter descriptions
    simple_tools = {"get_stats", "health_check"}

    for tool_name, tool in TOOL_REGISTRY.items():
        assert tool.description, f"Tool '{tool_name}' should have a description"

        # Complex search/research tools should have comprehensive descriptions
        if tool_name not in simple_tools:
            assert len(tool.description) > 50, \
                f"Tool '{tool_name}' description should be comprehensive (>50 chars)"


def test_tool_parameters_structure():
    """Verify all tools have properly structured parameter schemas."""
    for tool_name, tool in TOOL_REGISTRY.items():
        assert "type" in tool.parameters, f"Tool '{tool_name}' parameters should have 'type'"
        assert tool.parameters["type"] == "object", f"Tool '{tool_name}' parameters type should be 'object'"
        assert "properties" in tool.parameters, f"Tool '{tool_name}' should have 'properties'"


def test_search_regex_schema():
    """Verify search_regex has correct schema from decorator."""
    tool = TOOL_REGISTRY["search_regex"]

    # Check description
    assert "regular expressions" in tool.description.lower()
    assert "exact" in tool.description.lower() or "precise" in tool.description.lower()

    # Check parameters
    props = tool.parameters["properties"]
    assert "pattern" in props, "search_regex should have 'pattern' parameter"
    assert "page_size" in props, "search_regex should have 'page_size' parameter"
    assert "offset" in props, "search_regex should have 'offset' parameter"
    assert "max_response_tokens" in props, "search_regex should have 'max_response_tokens' parameter"
    assert "path" in props, "search_regex should have 'path' parameter"

    # Check required fields
    required = tool.parameters.get("required", [])
    assert "pattern" in required, "'pattern' should be required for search_regex"


def test_search_semantic_schema():
    """Verify search_semantic has correct schema from decorator."""
    tool = TOOL_REGISTRY["search_semantic"]

    # Check description
    assert "semantic" in tool.description.lower() or "meaning" in tool.description.lower()

    # Check parameters
    props = tool.parameters["properties"]
    assert "query" in props, "search_semantic should have 'query' parameter"
    assert "page_size" in props, "search_semantic should have 'page_size' parameter"
    assert "offset" in props, "search_semantic should have 'offset' parameter"
    assert "max_response_tokens" in props, "search_semantic should have 'max_response_tokens' parameter"
    assert "path" in props, "search_semantic should have 'path' parameter"
    assert "provider" in props, "search_semantic should have 'provider' parameter"
    assert "model" in props, "search_semantic should have 'model' parameter"
    assert "threshold" in props, "search_semantic should have 'threshold' parameter"

    # Check required fields
    required = tool.parameters.get("required", [])
    assert "query" in required, "'query' should be required for search_semantic"


def test_code_research_schema():
    """Verify code_research has correct schema from decorator."""
    tool = TOOL_REGISTRY["code_research"]

    # Check description
    assert "research" in tool.description.lower() or "architecture" in tool.description.lower()
    assert len(tool.description) > 100, "code_research should have comprehensive description"

    # Check parameters
    props = tool.parameters["properties"]
    assert "query" in props, "code_research should have 'query' parameter"

    # Check required fields
    required = tool.parameters.get("required", [])
    assert "query" in required, "'query' should be required for code_research"


def test_requires_embeddings_flag():
    """Verify tools correctly declare embedding requirements."""
    # Tools that don't require embeddings
    assert not TOOL_REGISTRY["get_stats"].requires_embeddings
    assert not TOOL_REGISTRY["health_check"].requires_embeddings
    assert not TOOL_REGISTRY["search_regex"].requires_embeddings

    # Tools that require embeddings
    assert TOOL_REGISTRY["search_semantic"].requires_embeddings
    assert TOOL_REGISTRY["code_research"].requires_embeddings


def test_http_server_uses_registry_descriptions():
    """Verify HTTP server imports and uses TOOL_REGISTRY for descriptions.

    This is a structural test - it ensures the HTTP server code references
    TOOL_REGISTRY to prevent regression to hardcoded descriptions.
    """
    from pathlib import Path

    http_server_path = Path(__file__).parent.parent / "chunkhound" / "mcp_server" / "http_server.py"
    content = http_server_path.read_text()

    # Check that TOOL_REGISTRY is imported
    assert "from .tools import" in content and "TOOL_REGISTRY" in content, \
        "HTTP server should import TOOL_REGISTRY"

    # Check that tools use TOOL_REGISTRY for descriptions
    assert 'TOOL_REGISTRY["get_stats"].description' in content, \
        "get_stats should use TOOL_REGISTRY description"
    assert 'TOOL_REGISTRY["search_regex"].description' in content, \
        "search_regex should use TOOL_REGISTRY description"
    assert 'TOOL_REGISTRY["search_semantic"].description' in content, \
        "search_semantic should use TOOL_REGISTRY description"
    assert 'TOOL_REGISTRY["code_research"].description' in content, \
        "code_research should use TOOL_REGISTRY description"


def test_default_values_in_schema():
    """Verify that default values are properly captured in schemas."""
    # search_regex defaults
    regex_props = TOOL_REGISTRY["search_regex"].parameters["properties"]
    assert regex_props["page_size"].get("default") == 10
    assert regex_props["offset"].get("default") == 0
    assert regex_props["max_response_tokens"].get("default") == 20000

    # search_semantic defaults
    semantic_props = TOOL_REGISTRY["search_semantic"].parameters["properties"]
    assert semantic_props["page_size"].get("default") == 10
    assert semantic_props["offset"].get("default") == 0
    assert semantic_props["max_response_tokens"].get("default") == 20000
    # provider and model should NOT have defaults - they should be None
    # to allow auto-detection from configured embedding provider
    assert "default" not in semantic_props["provider"], \
        "provider should not have default value (allows auto-detection)"
    assert "default" not in semantic_props["model"], \
        "model should not have default value (allows auto-detection)"


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
    matches = re.findall(r'@dataclass\s+class Tool:', content)
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
    assert "TOOL_DEFINITIONS = [" not in content, \
        "Old TOOL_DEFINITIONS list should be removed (registry now populated by decorators)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
