"""MCP schema surface tests for the websearch chain feature.

Guards two invariants:

1. **Positive:** The `websearch` MCP tool must advertise a `previous_query`
   parameter with a stable description, so `tools/list` clients discover the
   chain feature before inspecting parameters. A silent docstring drift
   would remove the only client-visible chain documentation.
2. **Negative:** The `code_research` MCP tool must NOT expose
   `previous_query`. The internal chaining hook is threaded through
   `deep_research_impl` for the websearch subprocess path only; its
   ``Annotated[..., InternalOnly()]`` marker keeps it off `code_research`'s
   auto-generated schema. Dropping the marker would silently leak the
   internal parameter.

Also covers the CLI-silent guard: the visible ``chunkhound research``
subcommand must not gain a ``--previous-query`` flag, and a mechanism-level
check that the ``InternalOnly`` marker itself works generically.
"""

import argparse
from typing import Annotated

from chunkhound.mcp_server.tools import (
    TOOL_REGISTRY,
    InternalOnly,
    _generate_json_schema_from_signature,
)


def test_websearch_mcp_schema_advertises_previous_query_description() -> None:
    schema = TOOL_REGISTRY["websearch"].parameters
    props = schema["properties"]
    assert "previous_query" in props
    assert props["previous_query"]["type"] == "string"
    assert (
        props["previous_query"]["description"]
        == "Previous query to build context and chain knowledge (optional)."
    )
    # Default is None → optional; must NOT be in required[].
    assert "previous_query" not in schema.get("required", [])


def test_code_research_mcp_schema_omits_previous_query() -> None:
    schema = TOOL_REGISTRY["code_research"].parameters
    assert "previous_query" not in schema["properties"], (
        "internal chaining hook leaked onto code_research's schema; check "
        "the Annotated[..., InternalOnly()] marker on deep_research_impl's "
        "`previous_query` parameter in chunkhound/mcp_server/tools.py"
    )
    assert "previous_query" not in schema.get("required", [])


def test_internal_only_marker_hides_parameter_generically() -> None:
    """The `InternalOnly` marker works on any parameter, not just previous_query."""

    def fake_tool(
        visible: str,
        hidden: Annotated[str | None, InternalOnly()] = None,
    ) -> None: ...

    schema = _generate_json_schema_from_signature(fake_tool)
    assert "visible" in schema["properties"]
    assert "hidden" not in schema["properties"]
    assert "hidden" not in schema.get("required", [])


def test_research_cli_has_no_previous_query_flag() -> None:
    """Visible ``chunkhound research`` CLI stays silent — no --previous-query."""
    from chunkhound.api.cli.parsers.research_parser import add_research_subparser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    research_parser = add_research_subparser(subparsers)

    dests = {a.dest for a in research_parser._actions}
    assert "previous_query" not in dests, (
        "research CLI must not expose --previous-query — chain feature is "
        "websearch-scoped only"
    )
    option_strings = {
        s for a in research_parser._actions for s in a.option_strings
    }
    assert "--previous-query" not in option_strings
