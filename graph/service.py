"""GraphService – high-level API for dependency graph traversal.

Consumed by:
* MCP ``code_graph`` tool  (see ``mcp_tool.py``)
* CLI ``chunkhound graph`` subcommand  (see ``cli.py``)
* Deep research graph-expansion phase
"""

from __future__ import annotations

from collections import deque
from typing import Any

from loguru import logger

from chunkhound.graph.models import (
    Edge,
    EdgeType,
    GraphNode,
    GraphQuery,
    GraphResult,
)
from chunkhound.graph.storage import (
    find_chunk_ids_by_symbol,
    get_edges_from,
    get_edges_to,
)


class GraphService:
    """Stateless service that performs BFS traversal over the edges table."""

    def __init__(self, execute_fn: Any) -> None:
        """
        Args:
            execute_fn: A callable that runs SQL and returns ``list[dict]``,
                        typically ``provider.execute_query``.
        """
        self._exec = execute_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def traverse(self, query: GraphQuery) -> GraphResult:
        """Run a BFS traversal from the given symbol.

        Returns a ``GraphResult`` containing discovered nodes and edges.
        """
        root_ids = find_chunk_ids_by_symbol(
            self._exec, query.symbol, query.path_filter
        )
        if not root_ids:
            logger.info(f"Graph: no chunks found for symbol {query.symbol!r}")
            return GraphResult(
                root_symbol=query.symbol,
                direction=query.direction,
            )

        visited_chunks: set[int] = set()
        result_nodes: list[GraphNode] = []
        result_edges: list[Edge] = []

        # Seed the BFS queue with root chunk IDs at depth 0.
        queue: deque[tuple[int, int]] = deque()
        for cid in root_ids:
            queue.append((cid, 0))
            visited_chunks.add(cid)
            meta = self._chunk_meta(cid)
            result_nodes.append(
                GraphNode(
                    chunk_id=cid,
                    symbol=meta.get("symbol"),
                    file_path=meta.get("file_path"),
                    chunk_type=meta.get("chunk_type"),
                    depth=0,
                )
            )

        while queue:
            current_id, depth = queue.popleft()
            if depth >= query.depth:
                continue

            neighbours = self._get_neighbours(
                current_id, query.direction, query.edge_types
            )

            for edge_row, neighbour_id in neighbours:
                edge = Edge(
                    id=edge_row.get("id"),
                    source_chunk_id=int(edge_row["source_chunk_id"]),
                    target_chunk_id=int(edge_row["target_chunk_id"]),
                    edge_type=EdgeType.from_string(
                        str(edge_row.get("edge_type", "import"))
                    ),
                    confidence=float(edge_row.get("confidence", 1.0)),
                )
                result_edges.append(edge)

                if neighbour_id not in visited_chunks:
                    visited_chunks.add(neighbour_id)
                    meta = self._chunk_meta(neighbour_id)

                    # Apply optional path filter.
                    if query.path_filter:
                        fpath = meta.get("file_path") or ""
                        if not fpath.startswith(query.path_filter):
                            continue

                    result_nodes.append(
                        GraphNode(
                            chunk_id=neighbour_id,
                            symbol=meta.get("symbol"),
                            file_path=meta.get("file_path"),
                            chunk_type=meta.get("chunk_type"),
                            depth=depth + 1,
                        )
                    )
                    queue.append((neighbour_id, depth + 1))

        return GraphResult(
            root_symbol=query.symbol,
            direction=query.direction,
            nodes=result_nodes,
            edges=result_edges,
        )

    def expand_chunks(
        self,
        chunk_ids: list[int],
        edge_types: list[EdgeType] | None = None,
        hops: int = 1,
    ) -> list[int]:
        """Return chunk IDs reachable within *hops* from the seed set.

        Used by deep research to pull in structurally connected chunks
        that vector search may have missed.
        """
        visited: set[int] = set(chunk_ids)
        frontier: set[int] = set(chunk_ids)

        for _ in range(hops):
            next_frontier: set[int] = set()
            for cid in frontier:
                for _row, neighbour_id in self._get_neighbours(
                    cid, "both", edge_types
                ):
                    if neighbour_id not in visited:
                        visited.add(neighbour_id)
                        next_frontier.add(neighbour_id)
            frontier = next_frontier
            if not frontier:
                break

        return sorted(visited)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_neighbours(
        self,
        chunk_id: int,
        direction: str,
        edge_types: list[EdgeType] | None,
    ) -> list[tuple[dict[str, Any], int]]:
        """Return ``(edge_row, neighbour_chunk_id)`` pairs."""
        results: list[tuple[dict[str, Any], int]] = []

        if direction in ("downstream", "both"):
            for row in get_edges_from(self._exec, chunk_id, edge_types):
                results.append((row, int(row["target_chunk_id"])))

        if direction in ("upstream", "both"):
            for row in get_edges_to(self._exec, chunk_id, edge_types):
                results.append((row, int(row["source_chunk_id"])))

        return results

    def _chunk_meta(self, chunk_id: int) -> dict[str, Any]:
        """Fetch lightweight metadata for a chunk."""
        rows = self._exec(
            f"SELECT c.symbol, c.chunk_type, f.path AS file_path "
            f"FROM chunks c LEFT JOIN files f ON f.id = c.file_id "
            f"WHERE c.id = {chunk_id}"
        )
        return rows[0] if rows else {}