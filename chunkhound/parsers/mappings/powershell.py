"""PowerShell language mapping for the unified parser architecture.

Provides PowerShell-specific tree-sitter queries and extraction logic so that
PowerShell users get function/class/method-aware chunks instead of the generic
UNKNOWN fallback. The grammar ships with tree-sitter-language-pack.

Handled constructs (real grammar node types):
- function_statement      -> FUNCTION  (covers `function` and `filter`)
- class_statement         -> CLASS
- class_method_definition -> METHOD
- class_property_definition-> PROPERTY
- enum_statement          -> ENUM
"""

from typing import Any

from tree_sitter import Node as TSNode

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalConcept

from .base import BaseMapping

# AST node type -> semantic kind. The router (universal_parser) maps "kind"
# to a ChunkType. We deliberately route on "kind" alone and never expose the
# raw node_type: "class_method_definition" / "class_property_definition" both
# contain the substring "class", which the router's `"class" in node_type`
# check would otherwise mis-route to ChunkType.CLASS.
_KIND_BY_TYPE = {
    "function_statement": "function",
    "class_statement": "class",
    "class_method_definition": "method",
    "class_property_definition": "property",
    "enum_statement": "enum",
}


class PowerShellMapping(BaseMapping):
    """PowerShell-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize PowerShell mapping."""
        super().__init__(Language.POWERSHELL)

    # Legacy query interface (required abstract methods) -------------------

    def get_function_query(self) -> str:
        """Tree-sitter query for PowerShell function/filter definitions."""
        return """
        (function_statement) @function_def
        (class_method_definition) @function_def
        """

    def get_class_query(self) -> str:
        """Tree-sitter query for PowerShell class/enum definitions."""
        return """
        (class_statement) @class_def
        (enum_statement) @class_def
        """

    def get_comment_query(self) -> str:
        """Tree-sitter query for PowerShell comments (# and <# #>)."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: TSNode | None, source: str) -> str:
        """Extract a function/method name from a definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")
        return self._extract_definition_name(node, source)

    def extract_class_name(self, node: TSNode | None, source: str) -> str:
        """Extract a class/enum name from a definition node."""
        if node is None:
            return self.get_fallback_name(node, "type")
        return self._extract_definition_name(node, source)

    # Universal concept interface -----------------------------------------

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for a universal concept in PowerShell."""
        if concept == UniversalConcept.DEFINITION:
            return """
            (function_statement) @definition
            (class_statement) @definition
            (class_method_definition) @definition
            (class_property_definition) @definition
            (enum_statement) @definition
            """
        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """
        return None

    def extract_name(
        self, concept: "UniversalConcept", captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract the symbol name from captures for this concept."""
        source = content.decode("utf-8")
        node = captures.get("definition")
        if node is None:
            return "unnamed"

        if concept == UniversalConcept.DEFINITION:
            return self._extract_definition_name(node, source)
        elif concept == UniversalConcept.COMMENT:
            return f"comment_line_{node.start_point[0] + 1}"
        return "unnamed"

    def extract_content(
        self, concept: "UniversalConcept", captures: dict[str, TSNode], content: bytes
    ) -> str:
        """Extract the source text from captures for this concept."""
        source = content.decode("utf-8")
        node = captures.get("definition")
        if node is None and captures:
            node = next(iter(captures.values()))
        return self.get_node_text(node, source)

    def extract_metadata(
        self, concept: "UniversalConcept", captures: dict[str, TSNode], content: bytes
    ) -> dict[str, Any]:
        """Extract PowerShell-specific metadata (drives ChunkType routing)."""
        metadata: dict[str, Any] = {}
        if concept != UniversalConcept.DEFINITION:
            return metadata

        node = captures.get("definition")
        if node is None:
            return metadata

        kind = _KIND_BY_TYPE.get(node.type)
        if kind:
            metadata["kind"] = kind
        return metadata

    # Helpers --------------------------------------------------------------

    def _extract_definition_name(self, node: TSNode, source: str) -> str:
        """Resolve the symbol name for any PowerShell definition node."""
        node_type = node.type

        if node_type == "function_statement":
            name_node = self.find_child_by_type(node, "function_name")
        elif node_type == "class_property_definition":
            # Property name lives in a `variable` child, e.g. `$Size`.
            var_node = self.find_child_by_type(node, "variable")
            if var_node:
                return self.get_node_text(var_node, source).lstrip("$").strip()
            name_node = None
        else:
            # class_statement, class_method_definition, enum_statement
            name_node = self.find_child_by_type(node, "simple_name")

        if name_node:
            return self.get_node_text(name_node, source).strip()
        return self.get_fallback_name(node, "definition")
