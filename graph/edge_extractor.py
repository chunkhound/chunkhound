"""Edge extraction from parsed AST trees.

Each language parser can optionally expose an ``extract_edges`` helper.
This module provides the Python implementation and a registry-based
dispatcher so parsers that do *not* support edge extraction simply
return an empty list (no breaking changes).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.graph.models import Edge, EdgeType


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def extract_edges_for_file(
    file_path: Path,
    chunks: list[dict[str, Any]],
    all_indexed_symbols: dict[str, int] | None = None,
) -> list[Edge]:
    """Dispatch to language-specific edge extraction.

    Args:
        file_path:           Parsed file.
        chunks:              Chunks already extracted by the parser for this file.
        all_indexed_symbols: Mapping ``symbol_name → chunk_id`` across the whole
                             index.  Used to resolve call / inheritance targets.

    Returns:
        List of ``Edge`` objects ready for insertion.
    """
    ext = file_path.suffix.lower()
    if ext == ".py":
        return _extract_python_edges(file_path, chunks, all_indexed_symbols or {})
    if ext in {".js", ".ts", ".jsx", ".tsx"}:
        return _extract_js_ts_edges(file_path, chunks, all_indexed_symbols or {})
    if ext in {".java", ".kt", ".kts"}:
        return _extract_jvm_edges(file_path, chunks, all_indexed_symbols or {})
    # Unsupported language → empty list (safe fallback)
    return []


# ---------------------------------------------------------------------------
# Python edge extraction
# ---------------------------------------------------------------------------

_PYTHON_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s]+)", re.MULTILINE
)
_PYTHON_CALL_RE = re.compile(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(")
_PYTHON_CLASS_INHERIT_RE = re.compile(r"class\s+\w+\s*\(([^)]+)\)")


def _extract_python_edges(
    file_path: Path,
    chunks: list[dict[str, Any]],
    symbols: dict[str, int],
) -> list[Edge]:
    edges: list[Edge] = []

    for chunk in chunks:
        code: str = chunk.get("code", "")
        source_id: int | None = chunk.get("id") or chunk.get("chunk_id")
        if source_id is None:
            continue

        # --- Import edges ---
        for match in _PYTHON_IMPORT_RE.finditer(code):
            module = match.group(1) or ""
            names = [n.strip() for n in match.group(2).split(",")]
            for name in names:
                clean = name.split(" as ")[0].strip()
                full_symbol = f"{module}.{clean}" if module else clean
                target_id = symbols.get(clean) or symbols.get(full_symbol)
                if target_id is not None and target_id != source_id:
                    edges.append(
                        Edge(
                            source_chunk_id=source_id,
                            target_chunk_id=target_id,
                            edge_type=EdgeType.IMPORT,
                            confidence=1.0,
                            metadata={"module": module, "name": clean},
                        )
                    )

        # --- Call edges ---
        for match in _PYTHON_CALL_RE.finditer(code):
            callee = match.group(1)
            # Skip builtins / common stdlib
            if callee in _PYTHON_BUILTINS:
                continue
            # Try the last segment (e.g. ``foo.bar()`` → ``bar``)
            short = callee.rsplit(".", 1)[-1]
            target_id = symbols.get(callee) or symbols.get(short)
            if target_id is not None and target_id != source_id:
                edges.append(
                    Edge(
                        source_chunk_id=source_id,
                        target_chunk_id=target_id,
                        edge_type=EdgeType.CALL,
                        confidence=0.8,
                        metadata={"callee": callee},
                    )
                )

        # --- Inheritance edges ---
        for match in _PYTHON_CLASS_INHERIT_RE.finditer(code):
            bases = [b.strip() for b in match.group(1).split(",")]
            for base in bases:
                base_short = base.rsplit(".", 1)[-1]
                target_id = symbols.get(base) or symbols.get(base_short)
                if target_id is not None and target_id != source_id:
                    edges.append(
                        Edge(
                            source_chunk_id=source_id,
                            target_chunk_id=target_id,
                            edge_type=EdgeType.INHERITANCE,
                            confidence=1.0,
                            metadata={"base_class": base},
                        )
                    )

    return edges


_PYTHON_BUILTINS = frozenset({
    "print", "len", "range", "int", "str", "float", "bool", "list", "dict",
    "set", "tuple", "type", "isinstance", "issubclass", "super", "getattr",
    "setattr", "hasattr", "open", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "min", "max", "sum", "any", "all", "abs", "round",
    "id", "hash", "repr", "format", "vars", "dir", "next", "iter",
})


# ---------------------------------------------------------------------------
# JavaScript / TypeScript edge extraction
# ---------------------------------------------------------------------------

_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:\{[^}]*\}|[\w*]+(?:\s+as\s+\w+)?)\s+from\s+['"]([^'"]+)['"]"""
    r"""|require\s*\(\s*['"]([^'"]+)['"]\s*\))""",
    re.MULTILINE,
)
_JS_CLASS_EXTENDS_RE = re.compile(r"class\s+\w+\s+extends\s+(\w+)")


def _extract_js_ts_edges(
    file_path: Path,
    chunks: list[dict[str, Any]],
    symbols: dict[str, int],
) -> list[Edge]:
    edges: list[Edge] = []
    for chunk in chunks:
        code = chunk.get("code", "")
        source_id = chunk.get("id") or chunk.get("chunk_id")
        if source_id is None:
            continue

        for match in _JS_IMPORT_RE.finditer(code):
            module = match.group(1) or match.group(2) or ""
            module_short = module.rsplit("/", 1)[-1]
            target_id = symbols.get(module_short) or symbols.get(module)
            if target_id is not None and target_id != source_id:
                edges.append(
                    Edge(
                        source_chunk_id=source_id,
                        target_chunk_id=target_id,
                        edge_type=EdgeType.IMPORT,
                        confidence=1.0,
                        metadata={"module": module},
                    )
                )

        for match in _JS_CLASS_EXTENDS_RE.finditer(code):
            base = match.group(1)
            target_id = symbols.get(base)
            if target_id is not None and target_id != source_id:
                edges.append(
                    Edge(
                        source_chunk_id=source_id,
                        target_chunk_id=target_id,
                        edge_type=EdgeType.INHERITANCE,
                        confidence=1.0,
                        metadata={"base_class": base},
                    )
                )

    return edges


# ---------------------------------------------------------------------------
# Java / Kotlin edge extraction
# ---------------------------------------------------------------------------

_JVM_IMPORT_RE = re.compile(r"import\s+([\w.]+);?")
_JVM_EXTENDS_RE = re.compile(r"(?:extends|implements)\s+([\w.,\s]+)")


def _extract_jvm_edges(
    file_path: Path,
    chunks: list[dict[str, Any]],
    symbols: dict[str, int],
) -> list[Edge]:
    edges: list[Edge] = []
    for chunk in chunks:
        code = chunk.get("code", "")
        source_id = chunk.get("id") or chunk.get("chunk_id")
        if source_id is None:
            continue

        for match in _JVM_IMPORT_RE.finditer(code):
            fqn = match.group(1)
            short = fqn.rsplit(".", 1)[-1]
            target_id = symbols.get(short) or symbols.get(fqn)
            if target_id is not None and target_id != source_id:
                edges.append(
                    Edge(
                        source_chunk_id=source_id,
                        target_chunk_id=target_id,
                        edge_type=EdgeType.IMPORT,
                        confidence=1.0,
                        metadata={"fqn": fqn},
                    )
                )

        for match in _JVM_EXTENDS_RE.finditer(code):
            bases = [b.strip() for b in match.group(1).split(",")]
            for base in bases:
                base_short = base.rsplit(".", 1)[-1]
                target_id = symbols.get(base_short) or symbols.get(base)
                if target_id is not None and target_id != source_id:
                    edges.append(
                        Edge(
                            source_chunk_id=source_id,
                            target_chunk_id=target_id,
                            edge_type=EdgeType.INHERITANCE,
                            confidence=1.0,
                            metadata={"base_class": base},
                        )
                    )

    return edges