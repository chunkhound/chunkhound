"""Domain models for the Dependency Graph Explorer.

Follows ChunkHound conventions:
- frozen dataclasses for immutable domain objects
- Pydantic BaseModel where validation / serialisation is needed
- NewType aliases for semantic clarity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType

EdgeId = NewType("EdgeId", int)


class EdgeType(Enum):
    """Relationship kinds captured during parsing."""

    IMPORT = "import"
    CALL = "call"
    INHERITANCE = "inheritance"

    @classmethod
    def from_string(cls, value: str) -> EdgeType:
        try:
            return cls(value.lower())
        except ValueError:
            return cls.IMPORT


@dataclass(frozen=True)
class Edge:
    """A single directed relationship between two chunks.

    Attributes:
        source_chunk_id: Chunk that *uses* the target (caller / importer / subclass).
        target_chunk_id: Chunk that is *used* (callee / importee / superclass).
        edge_type:       Relationship kind.
        confidence:      1.0 for explicit edges (``import X``), lower for inferred.
        metadata:        Extra info – e.g. alias used in import, call site line.
        id:              Database primary key (``None`` before insertion).
    """

    source_chunk_id: int
    target_chunk_id: int
    edge_type: EdgeType
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: EdgeId | None = None

    # ------------------------------------------------------------------
    # Serialisation helpers (mirror Chunk.to_dict / from_dict pattern)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source_chunk_id": self.source_chunk_id,
            "target_chunk_id": self.target_chunk_id,
            "edge_type": self.edge_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
        if self.id is not None:
            result["id"] = self.id
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        return cls(
            id=EdgeId(data["id"]) if "id" in data else None,
            source_chunk_id=int(data["source_chunk_id"]),
            target_chunk_id=int(data["target_chunk_id"]),
            edge_type=EdgeType.from_string(str(data.get("edge_type", "import"))),
            confidence=float(data.get("confidence", 1.0)),
            metadata=data.get("metadata") or {},
        )


@dataclass
class GraphQuery:
    """Parameters for a graph traversal request.

    Attributes:
        symbol:      Starting symbol name **or** file path prefix.
        direction:   ``"upstream"`` (who depends on me), ``"downstream"``
                     (what do I depend on), or ``"both"``.
        depth:       Maximum hops to traverse (default 3).
        edge_types:  Restrict to specific edge types (``None`` = all).
        path_filter: Optional path prefix to scope results.
    """

    symbol: str
    direction: str = "both"
    depth: int = 3
    edge_types: list[EdgeType] | None = None
    path_filter: str | None = None

    def __post_init__(self) -> None:
        allowed = {"upstream", "downstream", "both"}
        if self.direction not in allowed:
            raise ValueError(
                f"direction must be one of {allowed}, got {self.direction!r}"
            )
        if self.depth < 1:
            raise ValueError("depth must be >= 1")


@dataclass
class GraphNode:
    """A node in a traversal result."""

    chunk_id: int
    symbol: str | None
    file_path: str | None
    chunk_type: str | None
    depth: int  # 0 = root


@dataclass
class GraphResult:
    """Result of a graph traversal."""

    root_symbol: str
    direction: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_symbol": self.root_symbol,
            "direction": self.direction,
            "nodes": [
                {
                    "chunk_id": n.chunk_id,
                    "symbol": n.symbol,
                    "file_path": n.file_path,
                    "chunk_type": n.chunk_type,
                    "depth": n.depth,
                }
                for n in self.nodes
            ],
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_mermaid(self) -> str:
        """Render the result as a Mermaid flowchart."""
        lines: list[str] = ["graph TD"]
        node_labels: dict[int, str] = {}
        for node in self.nodes:
            label = node.symbol or f"chunk_{node.chunk_id}"
            safe = label.replace('"', "'")
            node_labels[node.chunk_id] = f"N{node.chunk_id}"
            lines.append(f'    N{node.chunk_id}["{safe}"]')
        for edge in self.edges:
            src = node_labels.get(edge.source_chunk_id, f"N{edge.source_chunk_id}")
            tgt = node_labels.get(edge.target_chunk_id, f"N{edge.target_chunk_id}")
            lines.append(f"    {src} -->|{edge.edge_type.value}| {tgt}")
        return "\n".join(lines)

    def to_text_tree(self, indent: int = 2) -> str:
        """Render the result as a compact indented text tree."""
        by_depth: dict[int, list[GraphNode]] = {}
        for node in self.nodes:
            by_depth.setdefault(node.depth, []).append(node)
        lines: list[str] = []
        for depth in sorted(by_depth):
            prefix = " " * (depth * indent)
            for node in by_depth[depth]:
                label = node.symbol or f"chunk_{node.chunk_id}"
                path_info = f" ({node.file_path})" if node.file_path else ""
                lines.append(f"{prefix}- {label}{path_info}")
        return "\n".join(lines)