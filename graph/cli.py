"""CLI subcommand: ``chunkhound graph``.

Usage examples::

    chunkhound graph EmbeddingProvider --direction upstream --depth 2
    chunkhound graph IndexingCoordinator.process_file --direction downstream
    chunkhound graph chunkhound/core/ --format mermaid --output deps.mermaid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from chunkhound.graph.models import EdgeType, GraphQuery
from chunkhound.graph.service import GraphService


def add_graph_subparser(subparsers: Any) -> None:
    """Register the ``graph`` subcommand on an argparse subparser group."""
    parser = subparsers.add_parser(
        "graph",
        help="Explore dependency relationships between code symbols",
        description=(
            "Traverse import, call, and inheritance edges in the indexed "
            "codebase.  Outputs JSON, Mermaid diagrams, or text trees."
        ),
    )
    parser.add_argument(
        "symbol",
        help="Symbol name or file path prefix to start from",
    )
    parser.add_argument(
        "--direction",
        choices=["upstream", "downstream", "both"],
        default="both",
        help="Traversal direction (default: both)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Maximum traversal hops (default: 3)",
    )
    parser.add_argument(
        "--edge-types",
        nargs="+",
        choices=["import", "call", "inheritance"],
        help="Restrict to specific edge types (default: all)",
    )
    parser.add_argument(
        "--path-filter",
        help="Optional path prefix to scope results (e.g. 'src/')",
    )
    parser.add_argument(
        "--format",
        choices=["json", "mermaid", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write output to file instead of stdout",
    )
    parser.set_defaults(func=_run_graph_command)


def _run_graph_command(args: argparse.Namespace) -> None:
    """Entry point called by the CLI dispatcher."""
    from chunkhound.core.config.config import Config
    from chunkhound.database_factory import create_services

    config = Config(args=args)
    db_path = config.database.get_db_path()
    services = create_services(db_path, config)

    graph_service = GraphService(execute_fn=services.provider.execute_query)

    edge_types = (
        [EdgeType.from_string(et) for et in args.edge_types]
        if args.edge_types
        else None
    )

    query = GraphQuery(
        symbol=args.symbol,
        direction=args.direction,
        depth=args.depth,
        edge_types=edge_types,
        path_filter=args.path_filter,
    )

    result = graph_service.traverse(query)

    if args.format == "mermaid":
        output = result.to_mermaid()
    elif args.format == "json":
        import json
        output = json.dumps(result.to_dict(), indent=2)
    else:
        output = result.to_text_tree()

    if not result.nodes:
        print(f"No results found for symbol: {args.symbol}", file=sys.stderr)
        return

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Graph written to {args.output}")
    else:
        print(output)