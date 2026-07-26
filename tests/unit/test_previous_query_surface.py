"""MCP schema + CLI surface tests for the ``previous_query`` follow-up hook."""

import argparse

from chunkhound.mcp_server.tools import TOOL_REGISTRY


def test_websearch_mcp_schema_advertises_previous_query() -> None:
    schema = TOOL_REGISTRY["websearch"].parameters
    props = schema["properties"]
    assert "previous_query" in props
    assert props["previous_query"]["type"] == "string"
    description = props["previous_query"]["description"].lower()
    assert "previous query" in description
    assert "chain" in description or "context" in description
    assert "previous_query" not in schema.get("required", [])


def test_code_research_mcp_schema_advertises_previous_query_framing_only() -> None:
    """code_research must expose previous_query with framing-only wording.

    The description must convey that the parameter only affects synthesis
    framing, not what gets searched/retrieved — otherwise callers would
    wrongly expect it to steer retrieval (which is websearch's behavior).
    """
    schema = TOOL_REGISTRY["code_research"].parameters
    props = schema["properties"]
    assert "previous_query" in props
    assert props["previous_query"]["type"] == "string"
    description = props["previous_query"]["description"].lower()
    assert "framing" in description or "synthesis" in description
    # Must not promise retrieval-steering behavior — that's websearch-only.
    for forbidden in ("expand", "expander", "expansion", "query steer"):
        assert forbidden not in description, (
            f"code_research previous_query description contains "
            f"retrieval-steering wording ({forbidden!r}); in this path the "
            f"parameter only reaches the synthesis engine."
        )
    assert "previous_query" not in schema.get("required", [])


def test_research_cli_exposes_previous_query_flag() -> None:
    from chunkhound.api.cli.parsers.research_parser import add_research_subparser

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    research_parser = add_research_subparser(subparsers)

    dests = {a.dest for a in research_parser._actions}
    assert "previous_query" in dests
    option_strings = {
        s for a in research_parser._actions for s in a.option_strings
    }
    assert "--previous-query" in option_strings
