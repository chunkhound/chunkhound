"""Test HTTP server tool schema consistency with TOOL_REGISTRY.

This test ensures HTTP tools have complete parameter descriptions
that match the stdio mode schemas defined in TOOL_REGISTRY.

TDD RED Phase: These tests should FAIL until we implement schema patching.
"""

import pytest

from chunkhound.core.config.config import Config
from chunkhound.mcp_server.http_server import HttpMCPServer
from chunkhound.mcp_server.tools import TOOL_REGISTRY


@pytest.fixture
def http_server():
    """Create HTTP server instance for testing."""
    # Create minimal config for testing
    config = Config()
    server = HttpMCPServer(config, port=5173)
    return server


@pytest.mark.asyncio
async def test_http_tools_match_tool_registry(http_server):
    """Verify HTTP tool schemas match TOOL_REGISTRY exactly.

    This is the core consistency test - HTTP and stdio modes should
    expose identical tool schemas to ensure consistent LLM experience.
    """
    # Get all tools from HTTP server
    for tool_name in TOOL_REGISTRY.keys():
        # Access tool via FastMCP's internal tool manager (async)
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        assert http_tool is not None, f"Tool '{tool_name}' not found in HTTP server"

        # Get expected schema from TOOL_REGISTRY
        expected_schema = TOOL_REGISTRY[tool_name].parameters

        # Compare schemas - HTTP should match TOOL_REGISTRY exactly
        http_schema = http_tool.parameters

        assert http_schema == expected_schema, (
            f"Tool '{tool_name}' HTTP schema does not match TOOL_REGISTRY.\n"
            f"Expected properties: {list(expected_schema.get('properties', {}).keys())}\n"
            f"HTTP properties: {list(http_schema.get('properties', {}).keys())}\n"
            f"Expected schema: {expected_schema}\n"
            f"HTTP schema: {http_schema}"
        )


@pytest.mark.asyncio
async def test_http_tool_parameter_descriptions(http_server):
    """Verify all HTTP tool parameters have descriptions.

    Parameter descriptions are essential for LLMs to understand
    how to use each parameter correctly.
    """
    for tool_name in TOOL_REGISTRY.keys():
        # Access tool via FastMCP's internal tool manager (async)
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        assert http_tool is not None, f"Tool '{tool_name}' not found in HTTP server"

        # Check each parameter has a description
        http_schema = http_tool.parameters
        properties = http_schema.get("properties", {})

        for param_name, param_schema in properties.items():
            assert "description" in param_schema, (
                f"Tool '{tool_name}' parameter '{param_name}' missing description in HTTP schema.\n"
                f"Parameter schema: {param_schema}\n"
                f"Expected description from TOOL_REGISTRY: "
                f"{TOOL_REGISTRY[tool_name].parameters['properties'][param_name].get('description', 'N/A')}"
            )

            # Description should be non-empty
            description = param_schema["description"]
            assert description and description.strip(), (
                f"Tool '{tool_name}' parameter '{param_name}' has empty description in HTTP schema"
            )


@pytest.mark.asyncio
async def test_http_optional_parameters_format(http_server):
    """Verify optional parameters don't use anyOf format.

    FastMCP auto-generates schemas with `anyOf: [{type: "string"}, {type: "null"}]`
    for optional parameters. TOOL_REGISTRY uses simpler format without anyOf.
    This test ensures HTTP schemas match the cleaner TOOL_REGISTRY format.
    """
    for tool_name in TOOL_REGISTRY.keys():
        # Access tool via FastMCP's internal tool manager (async)
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        assert http_tool is not None, f"Tool '{tool_name}' not found in HTTP server"

        http_schema = http_tool.parameters
        properties = http_schema.get("properties", {})

        for param_name, param_schema in properties.items():
            # Check that optional parameters don't use anyOf format
            if "anyOf" in param_schema:
                # This is the format FastMCP auto-generates - we want to avoid it
                expected_schema = TOOL_REGISTRY[tool_name].parameters["properties"][param_name]

                assert False, (
                    f"Tool '{tool_name}' parameter '{param_name}' uses anyOf format in HTTP schema.\n"
                    f"HTTP schema: {param_schema}\n"
                    f"Expected schema from TOOL_REGISTRY: {expected_schema}\n"
                    f"Optional parameters should use simple type format, not anyOf."
                )


@pytest.mark.asyncio
async def test_http_required_parameters_match(http_server):
    """Verify required parameters list matches TOOL_REGISTRY."""
    for tool_name in TOOL_REGISTRY.keys():
        # Access tool via FastMCP's internal tool manager (async)
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        assert http_tool is not None, f"Tool '{tool_name}' not found in HTTP server"

        expected_required = set(TOOL_REGISTRY[tool_name].parameters.get("required", []))
        http_required = set(http_tool.parameters.get("required", []))

        assert http_required == expected_required, (
            f"Tool '{tool_name}' required parameters don't match.\n"
            f"Expected: {sorted(expected_required)}\n"
            f"HTTP: {sorted(http_required)}"
        )


@pytest.mark.asyncio
async def test_http_default_values_match(http_server):
    """Verify default values match TOOL_REGISTRY."""
    for tool_name in TOOL_REGISTRY.keys():
        # Access tool via FastMCP's internal tool manager (async)
        http_tool = await http_server.app._tool_manager.get_tool(tool_name)
        assert http_tool is not None, f"Tool '{tool_name}' not found in HTTP server"

        expected_props = TOOL_REGISTRY[tool_name].parameters.get("properties", {})
        http_props = http_tool.parameters.get("properties", {})

        for param_name in expected_props.keys():
            expected_default = expected_props[param_name].get("default")
            http_default = http_props.get(param_name, {}).get("default")

            assert http_default == expected_default, (
                f"Tool '{tool_name}' parameter '{param_name}' default value mismatch.\n"
                f"Expected: {expected_default}\n"
                f"HTTP: {http_default}"
            )


@pytest.mark.asyncio
async def test_search_regex_http_schema_details(http_server):
    """Detailed test for search_regex tool schema consistency."""
    tool_name = "search_regex"
    http_tool = await http_server.app._tool_manager.get_tool(tool_name)

    # Check specific parameters have descriptions
    props = http_tool.parameters["properties"]

    # Pattern parameter
    assert "description" in props["pattern"], "pattern parameter missing description"
    assert "regex" in props["pattern"]["description"].lower(), (
        "pattern description should mention 'regex'"
    )

    # Path parameter
    assert "description" in props["path"], "path parameter missing description"
    assert "optional" in props["path"]["description"].lower() or "limit" in props["path"]["description"].lower(), (
        "path description should explain it's optional or limits search scope"
    )

    # Max response tokens parameter
    assert "description" in props["max_response_tokens"], (
        "max_response_tokens parameter missing description"
    )
    assert "token" in props["max_response_tokens"]["description"].lower(), (
        "max_response_tokens description should mention 'token'"
    )


@pytest.mark.asyncio
async def test_search_semantic_http_schema_details(http_server):
    """Detailed test for search_semantic tool schema consistency."""
    tool_name = "search_semantic"
    http_tool = await http_server.app._tool_manager.get_tool(tool_name)

    # Check specific parameters have descriptions
    props = http_tool.parameters["properties"]

    # Query parameter
    assert "description" in props["query"], "query parameter missing description"

    # Provider parameter
    assert "description" in props["provider"], "provider parameter missing description"

    # Model parameter
    assert "description" in props["model"], "model parameter missing description"

    # Threshold parameter
    assert "description" in props["threshold"], "threshold parameter missing description"


@pytest.mark.asyncio
async def test_code_research_http_schema_details(http_server):
    """Detailed test for code_research tool schema consistency."""
    tool_name = "code_research"
    http_tool = await http_server.app._tool_manager.get_tool(tool_name)

    # Check query parameter has description
    props = http_tool.parameters["properties"]
    assert "description" in props["query"], "query parameter missing description"
    assert "research" in props["query"]["description"].lower(), (
        "query description should mention 'research'"
    )

    # Optional path parameter should exist and describe scoping behavior
    assert "path" in props, "code_research tool missing optional 'path' parameter"
    assert "description" in props["path"], "path parameter missing description"
    path_desc = props["path"]["description"].lower()
    assert "scope" in path_desc or "limit" in path_desc or "path" in path_desc, (
        "path description should explain it limits research scope to a subdirectory"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
