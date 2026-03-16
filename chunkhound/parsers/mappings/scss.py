"""SCSS language mapping for unified parser architecture.

Extends CSS parsing with SCSS-specific constructs:
- @mixin/@function       → DEFINITION (with name)
- $variable declarations → STRUCTURE
- @include               → BLOCK (inline call)
- rule_set               → DEFINITION (selector string)
- @media/@keyframes      → BLOCK
- @import/@use/@forward  → IMPORT
- comment                → COMMENT
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


def _identifier_name(node: Node, content: bytes) -> str:
    """Get the identifier child text from a mixin/function statement."""
    for child in node.children:
        if child.type == "identifier":
            return content[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
    return f"unknown_line{node.start_point[0] + 1}"


def _property_name(node: Node, content: bytes) -> str:
    """Get property_name from a declaration (SCSS $variable)."""
    for child in node.children:
        if child.type == "property_name":
            return content[child.start_byte : child.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
    return f"var_line{node.start_point[0] + 1}"


class ScssMapping(BaseMapping):
    """SCSS-specific mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.SCSS)

    def get_function_query(self) -> str:
        return "(mixin_statement) @definition (function_statement) @definition"

    def get_class_query(self) -> str:
        return ""

    def get_comment_query(self) -> str:
        return "(comment) @definition"

    def extract_function_name(self, node: Node | None, source: str) -> str:
        if node is None:
            return ""
        return _identifier_name(node, source.encode("utf-8", errors="replace"))

    def extract_class_name(self, node: Node | None, source: str) -> str:
        return ""

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        if concept == UniversalConcept.DEFINITION:
            return """
                (mixin_statement) @definition
                (function_statement) @definition
                (rule_set) @definition
            """
        elif concept == UniversalConcept.BLOCK:
            return """
                (media_statement) @definition
                (keyframes_statement) @definition
                (include_statement) @definition
                (each_statement) @definition
                (for_statement) @definition
                (while_statement) @definition
                (if_statement) @definition
            """
        elif concept == UniversalConcept.STRUCTURE:
            # $variable declarations (top-level declarations starting with $)
            return "(declaration) @definition"
        elif concept == UniversalConcept.IMPORT:
            return """
                (import_statement) @definition
                (use_statement) @definition
                (forward_statement) @definition
            """
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
            if node.type == "mixin_statement":
                return f"@mixin {_identifier_name(node, content)}"
            elif node.type == "function_statement":
                return f"@function {_identifier_name(node, content)}"
            elif node.type == "rule_set":
                return _selector_text(node, content)

        elif concept == UniversalConcept.BLOCK:
            if node.type == "media_statement":
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
            elif node.type == "include_statement":
                return f"@include_line{node.start_point[0] + 1}"
            else:
                type_name = node.type.replace("_statement", "")
                return f"@{type_name}_line{node.start_point[0] + 1}"

        elif concept == UniversalConcept.STRUCTURE:
            name = _property_name(node, content)
            return name

        elif concept == UniversalConcept.IMPORT:
            raw = content[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            ).strip()
            # Strip @import/@use/@forward prefix and trailing semicolon
            for prefix in ("@forward", "@import", "@use"):
                if raw.startswith(prefix):
                    raw = raw[len(prefix) :].strip().rstrip(";").strip()
                    break
            return raw.strip("\"'")[:60]

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
        # STRUCTURE: only $variable declarations
        if concept == UniversalConcept.STRUCTURE:
            if node.type != "declaration":
                return ""
            prop = _property_name(node, content)
            if not prop.startswith("$"):
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
            if node.type in ("mixin_statement", "function_statement"):
                metadata["name"] = _identifier_name(node, content)
                metadata["chunk_type_hint"] = "function"
            elif node.type == "rule_set":
                metadata["selector"] = _selector_text(node, content)
                metadata["chunk_type_hint"] = "block"
            elif node.type == "declaration":
                prop = _property_name(node, content)
                metadata["property"] = prop
                if prop.startswith("$"):
                    metadata["kind"] = "variable"
        return metadata

    def resolve_import_paths(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> list[Path]:
        path = import_text.strip("\"'")
        # Try with and without leading underscore (SCSS partials)
        candidates = [base_dir / path]
        stem = Path(path).stem
        parent = Path(path).parent
        candidates.append(base_dir / parent / f"_{stem}.scss")
        candidates.append(base_dir / f"{path}.scss")
        for c in candidates:
            if c.exists():
                return [c]
        return []

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        return None
