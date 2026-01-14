"""Shared extraction logic for JavaScript-family mappings (JS/TS/JSX/TSX).

Provides a single, minimal implementation of the universal extraction trio:
`extract_name`, `extract_metadata`, and `extract_content`.

All implementations are behavior-preserving copies of the existing JS/TS
methods, consolidated here to avoid duplication across mappings.
"""

from typing import Any

from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH
from chunkhound.parsers.universal_engine import UniversalConcept

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TSNode = Any  # type: ignore[misc,assignment]
    TREE_SITTER_AVAILABLE = False


class JSFamilyExtraction:
    """Mixin: shared extraction helpers for JS-family mappings."""

    # BaseMapping supplies get_node_text; subclasses must inherit it via MRO

    def extract_name(
        self,
        concept: UniversalConcept,
        captures: dict[str, TSNode],
        content: bytes,
    ) -> str:
        source = content.decode("utf-8", errors="replace")

        if concept == UniversalConcept.DEFINITION:
            if "name" in captures:
                name_text = self.get_node_text(captures["name"], source).strip()  # type: ignore[attr-defined]
                return name_text or "definition"
            if "definition" in captures:
                node = captures["definition"]
                text = self.get_node_text(node, source).strip()  # type: ignore[attr-defined]
                if text.startswith("export default"):
                    return "export_default"
                if "module.exports" in text:
                    return "module_exports"
                line = node.start_point[0] + 1
                return f"definition_line_{line}"

        if concept == UniversalConcept.COMMENT and "definition" in captures:
            node = captures["definition"]
            return f"comment_line_{node.start_point[0] + 1}"

        return f"unnamed_{concept.value}"

    def extract_metadata(
        self,
        concept: UniversalConcept,
        captures: dict[str, TSNode],
        content: bytes,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {"concept": concept.value}

        if concept == UniversalConcept.DEFINITION and "definition" in captures:
            node = captures["definition"]
            init = captures.get("init")
            target = init or node
            try:
                node_type = getattr(target, "type", "")
                if node_type == "object":
                    meta["chunk_type_hint"] = "object"
                elif node_type == "array":
                    meta["chunk_type_hint"] = "array"
                else:
                    for i in range(getattr(target, "child_count", 0)):
                        child = target.child(i)
                        if not child:
                            continue
                        if child.type == "object":
                            meta["chunk_type_hint"] = "object"
                            break
                        if child.type == "array":
                            meta["chunk_type_hint"] = "array"
                            break
            except Exception:
                # Best-effort only; do not set hint on failure
                pass
        return meta

    def extract_content(
        self,
        concept: UniversalConcept,
        captures: dict[str, TSNode],
        content: bytes,
    ) -> str:
        source = content.decode("utf-8", errors="replace")
        node = captures.get("definition") or next(iter(captures.values()), None)
        return self.get_node_text(node, source) if node is not None else ""  # type: ignore[attr-defined]

    def extract_constants(
        self,
        concept: UniversalConcept,
        captures: dict[str, TSNode],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from JavaScript/TypeScript code.

        Detects:
        - const declarations (const FOO = "value")
        - TypeScript enum members (enum Color { RED = 1 })

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name' and 'value' keys, or None
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        if not TREE_SITTER_AVAILABLE:
            return None

        source = content.decode("utf-8", errors="replace")
        def_node = captures.get("definition")

        if not def_node:
            return None

        # Handle TypeScript enum members
        if def_node.type == "enum_declaration":
            return self._extract_enum_constants(def_node, source)

        # Handle const declarations (lexical_declaration with const keyword)
        if def_node.type == "lexical_declaration":
            return self._extract_lexical_constants(def_node, source)

        # Handle export statements wrapping lexical/enum declarations
        if def_node.type == "export_statement":
            inner_decl = self.find_child_by_type(def_node, "lexical_declaration")  # type: ignore[attr-defined]
            if inner_decl:
                return self._extract_lexical_constants(inner_decl, source)
            inner_enum = self.find_child_by_type(def_node, "enum_declaration")  # type: ignore[attr-defined]
            if inner_enum:
                return self._extract_enum_constants(inner_enum, source)

        return None

    def _extract_enum_constants(
        self, enum_node: TSNode, source: str
    ) -> list[dict[str, str]] | None:
        """Extract constants from TypeScript enum declaration.

        Args:
            enum_node: Tree-sitter enum_declaration node
            source: Source code string

        Returns:
            List of constant dictionaries or None
        """
        constants = []

        # Find enum body
        enum_body = self.find_child_by_type(enum_node, "enum_body")  # type: ignore[attr-defined]
        if not enum_body:
            return None

        # Extract each enum member
        # Handles both:
        # - enum_assignment nodes (explicit values: RED = 1)
        # - property_identifier nodes (implicit values: RED, GREEN, BLUE)
        for i in range(enum_body.child_count):
            child = enum_body.child(i)
            if not child:
                continue

            if child.type == "enum_assignment":
                # Explicit value: enum Color { RED = 1 }
                name_node = self.find_child_by_type(child, "property_identifier")  # type: ignore[attr-defined]
                if not name_node:
                    continue

                member_name = self.get_node_text(name_node, source).strip()  # type: ignore[attr-defined]
                if not member_name:
                    continue

                # Find value (number, string, or other expression)
                value = ""
                for j in range(child.child_count):
                    value_child = child.child(j)
                    if value_child and value_child.type not in ("property_identifier", "="):
                        value = self.get_node_text(value_child, source).strip()  # type: ignore[attr-defined]
                        break

                # Truncate value to 50 characters if longer
                if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                    value = value[:MAX_CONSTANT_VALUE_LENGTH]

                constants.append({"name": member_name, "value": value})

            elif child.type == "property_identifier":
                # Implicit value: enum Color { RED, GREEN, BLUE }
                member_name = self.get_node_text(child, source).strip()  # type: ignore[attr-defined]
                if not member_name:
                    continue

                constants.append({"name": member_name, "value": ""})

        return constants if constants else None

    def _extract_lexical_constants(
        self, lexical_node: TSNode, source: str
    ) -> list[dict[str, str]] | None:
        """Extract constants from const declarations.

        Args:
            lexical_node: Tree-sitter lexical_declaration node
            source: Source code string

        Returns:
            List of constant dictionaries or None
        """
        # Check if it's a const declaration (not let)
        node_text = self.get_node_text(lexical_node, source).strip()  # type: ignore[attr-defined]
        if not node_text.startswith("const "):
            return None

        constants = []

        # Find all variable_declarator children
        for i in range(lexical_node.child_count):
            child = lexical_node.child(i)
            if not child or child.type != "variable_declarator":
                continue

            # Extract variable name
            name_node = self.find_child_by_type(child, "identifier")  # type: ignore[attr-defined]
            if not name_node:
                continue

            var_name = self.get_node_text(name_node, source).strip()  # type: ignore[attr-defined]
            if not var_name:
                continue

            # Extract value from the declarator
            # Look for the value after the name (skip type annotations in TS)
            value = ""
            for j in range(child.child_count):
                value_child = child.child(j)
                if not value_child:
                    continue

                # Skip name, type annotations, and operators
                if value_child.type in ["identifier", "type_annotation", ":", "="]:
                    continue

                # This should be the actual value
                value = self.get_node_text(value_child, source).strip()  # type: ignore[attr-defined]
                break

            # Truncate value to 50 characters if longer
            if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                value = value[:MAX_CONSTANT_VALUE_LENGTH]

            constants.append({"name": var_name, "value": value})

        return constants if constants else None

