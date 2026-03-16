"""CSS language mapping for unified parser architecture.

Maps CSS AST nodes to semantic chunks:
- rule_set            → DEFINITION (selector string as name)
- @media/@keyframes   → BLOCK
- :root / * with vars → STRUCTURE
- @import             → IMPORT
- comment             → COMMENT
"""

from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

_MAX_SELECTOR_LEN = 60


def _selector_text(node: Node, content: bytes) -> str:
    """Extract selector string from a rule_set node."""
    for child in node.children:
        if child.type == "selectors":
            raw = content[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            if len(raw) > _MAX_SELECTOR_LEN:
                raw = raw[:_MAX_SELECTOR_LEN] + "…"
            return raw
    return f"rule_line{node.start_point[0] + 1}"


def _is_root_vars(node: Node, content: bytes) -> bool:
    """Return True if rule_set is :root or * containing custom properties."""
    sel = _selector_text(node, content)
    if sel not in (":root", "*"):
        return False
    # Check if block contains any --var declarations
    for child in node.children:
        if child.type == "block":
            block_text = content[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            )
            if "--" in block_text:
                return True
    return False


class CssMapping(BaseMapping):
    """CSS-specific mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.CSS)

    def get_function_query(self) -> str:
        """CSS has no function definitions to extract."""
        return ""

    def get_class_query(self) -> str:
        """CSS has no class definitions to extract."""
        return ""

    def get_comment_query(self) -> str:
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """CSS has no function definitions; always returns empty string."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """CSS has no class definitions; always returns empty string."""
        return ""

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        if concept == UniversalConcept.DEFINITION:
            return "(rule_set) @definition"
        elif concept == UniversalConcept.BLOCK:
            return """
                (media_statement) @definition
                (keyframes_statement) @definition
                (supports_statement) @definition
            """
        elif concept == UniversalConcept.STRUCTURE:
            # :root rules with CSS custom properties
            return "(rule_set) @definition"
        elif concept == UniversalConcept.IMPORT:
            return "(import_statement) @definition"
        elif concept == UniversalConcept.COMMENT:
            return "(comment) @definition"
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.DEFINITION:
            return _selector_text(node, content)

        elif concept == UniversalConcept.BLOCK:
            if node.type == "media_statement":
                # Get the media condition
                for child in node.children:
                    if child.type not in ("@media", "block"):
                        cond = content[child.start_byte : child.end_byte].decode(
                            "utf-8", errors="replace"
                        ).strip()
                        return f"@media {cond[:40]}"
                return f"@media_line{node.start_point[0] + 1}"
            elif node.type == "keyframes_statement":
                for child in node.children:
                    if child.type == "keyframes_name":
                        name = content[child.start_byte : child.end_byte].decode(
                            "utf-8", errors="replace"
                        ).strip()
                        return f"@keyframes {name}"
                return f"@keyframes_line{node.start_point[0] + 1}"
            elif node.type == "supports_statement":
                return f"@supports_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.STRUCTURE:
            return ":root_vars"

        elif concept == UniversalConcept.IMPORT:
            raw = content[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            # Strip '@import ' prefix and trailing semicolon
            raw = raw.removeprefix("@import").strip().rstrip(";").strip()
            return raw[:60]

        elif concept == UniversalConcept.COMMENT:
            return f"comment_line{node.start_point[0] + 1}"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        if node is None:
            return ""
        # STRUCTURE: only :root/:* blocks with --variables
        if concept == UniversalConcept.STRUCTURE:
            if not _is_root_vars(node, content):
                return ""
        # DEFINITION: exclude :root/:* var blocks (those are STRUCTURE)
        if concept == UniversalConcept.DEFINITION:
            if _is_root_vars(node, content):
                return ""
        return content[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        node = captures.get("definition") or (
            next(iter(captures.values()), None) if captures else None
        )
        metadata: dict[str, Any] = {}
        if node is not None:
            metadata["node_type"] = node.type
            if node.type == "rule_set":
                metadata["selector"] = _selector_text(node, content)
                metadata["is_root_vars"] = _is_root_vars(node, content)
                metadata["chunk_type_hint"] = "block"
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        # Strip quotes and url()
        path = import_text.strip("\"'")
        if path.startswith("url("):
            path = path[4:].rstrip(")").strip("\"'")
        candidate = base_dir / path
        if candidate.exists():
            return [candidate]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        return None
