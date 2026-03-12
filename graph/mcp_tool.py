"""MCP tool definition for the ``code_graph`` dependency graph tool.

Registers a single MCP tool that AI assistants can invoke to traverse
import / call / inheritance relationships in the indexed codebase.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from chunkhound.graph.models import EdgeType, GraphQuery
from chunkhound.graph.service import GraphService


async def handle_code_graph(
    arguments: dict[str, Any],
    graph_service: GraphService,
) -> dict[str, Any]:
    """Handle a ``code_graph`` MCP tool invocation.

    Expected *arguments* schema::

        {
            "symbol":      str,                           # required
            "direction":   "upstream"|"downstream"|"both", # default "both"
            "depth":       int,                            # default 3
            "edge_types":  ["import","call","inheritance"],# default all
            "path_filter": str | null,                     # default null
            "format":      "json"|"mermaid"|"text"         # default "json"
        }
    """
    symbol = arguments.get("symbol", "")
    if not symbol:
        return {"error": "symbol parameter is required"}

    edge_type_strs = arguments.get("edge_types") or []
    edge_types = [EdgeType.from_string(et) for et in edge_type_strs] or None

    query = GraphQuery(
        symbol=symbol,
        direction=arguments.get("direction", "both"),
        depth=int(arguments.get("depth", 3)),
        edge_types=edge_types,
        path_filter=arguments.get("path_filter"),
    )

    result = graph_service.traverse(query)

    output_format = arguments.get("format", "json")
    if output_format == "mermaid":
        return {"content": result.to_mermaid(), "format": "mermaid"}
    if output_format == "text":
        return {"content": result.to_text_tree(), "format": "text"}
    return result.to_dict()


# ------------------------------------------------------------------
# Tool schema for MCP registration
# ------------------------------------------------------------------

CODE_GRAPH_TOOL_SCHEMA: dict[str, Any] = {
    "name": "code_graph",
    "description": (
        "Query dependency relationships (imports, calls, inheritance) "
        "between code symbols in the indexed codebase. Returns connected "
        "nodes and edges as JSON, Mermaid diagram, or text tree."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["symbol"],
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Symbol name or file path prefix to start traversal from",
            },
            "direction": {
                "type": "string",
                "enum": ["upstream", "downstream", "both"],
                "default": "both",
                "description": "upstream = who depends on me, downstream = what I depend on",
            },
            "depth": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum hops to traverse",
            },
            "edge_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["import", "call", "inheritance"]},
                "description": "Restrict to specific edge types (omit for all)",
            },
            "path_filter": {
                "type": "string",
                "description": "Optional path prefix to scope results (e.g. 'src/')",
            },
            "format": {
                "type": "string",
                "enum": ["json", "mermaid", "text"],
                "default": "json",
                "description": "Output format",
            },
        },
    },
}