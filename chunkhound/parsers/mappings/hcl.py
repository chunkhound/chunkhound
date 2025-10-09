"""HCL language mapping for the universal parser.

This mapping targets HashiCorp Configuration Language (HCL) including Terraform
(.tf/.tfvars) files using tree-sitter-hcl. It extracts blocks (resource, module,
variable, output, data, provider, terraform, etc.), attributes within blocks,
and comments, providing meaningful names and metadata for chunking.
"""

from __future__ import annotations

from typing import Any, List

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class HclMapping(BaseMapping):
    """HCL-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        super().__init__(Language.HCL)

    # --- BaseMapping abstract API (not directly used by the universal adapter) ---
    def get_function_query(self) -> str:
        return ""

    def get_class_query(self) -> str:
        return ""

    def get_comment_query(self) -> str:
        # Provided for completeness; the universal concept path also defines comments
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        return ""

    # --- LanguageMapping protocol for universal concepts ---
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Return tree-sitter queries for HCL concepts.

        Notes on HCL grammar (tree-sitter-hcl):
        - Root nodes: `config_file` -> (`body` | `object` for .tfvars.json)
        - Definitions: `block` (identifier + labels + body), `attribute` (key = expr)
        - Comments: `comment`
        """
        if concept == UniversalConcept.DEFINITION:
            # Capture attribute key/value pairs and object elements (but not blocks here)
            return """
            (attribute
                (identifier) @key
                (expression) @value
            ) @definition

            ; object key/value pairs inside attribute/object values
            (object_elem
                key: (expression) @inner_key
                val: (expression) @inner_value
            ) @definition
            """

        if concept == UniversalConcept.BLOCK:
            # Only use explicit HCL blocks as containers (avoid noisy body/object captures)
            return """
            (block) @definition
            """

        if concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        if concept == UniversalConcept.STRUCTURE:
            return """
            (config_file) @definition
            (body) @definition
            """

        # No explicit import concept in HCL
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (list(captures.values())[0] if captures else None)
        if node is None:
            return f"unnamed_{concept.value}"

        if node.type == "block":
            return self._block_display_name(node, content)

        if node.type == "attribute":
            # Prefer captured key when available
            key_node = captures.get("key")
            if key_node is not None:
                key = self.get_node_text(key_node, self._decode(content)).strip()
            else:
                key = self._attribute_key(node, content)
            parent_path = self._parent_block_path(node, content)
            if parent_path:
                return f"{parent_path}.{key}" if key else parent_path
            return key or "attribute"

        # Nested object pair inside an attribute value
        if node.type == "object_elem":
            src = self._decode(content)
            inner_key_node = captures.get("inner_key")
            inner_key_raw = self.get_node_text(inner_key_node, src).strip() if inner_key_node else ""
            inner_key = self._strip_quotes(inner_key_raw)

            # Find nearest attribute ancestor to get the attribute key
            attr_parent = node.parent
            while attr_parent is not None and attr_parent.type != "attribute":
                attr_parent = attr_parent.parent
            attr_key = self._attribute_key(attr_parent, content) if attr_parent else ""

            # Compose from nearest block path
            parent_path = self._parent_block_path(node, content)
            parts = [p for p in [parent_path, attr_key, inner_key] if p]
            return ".".join(parts) if parts else "object_elem"

        if concept == UniversalConcept.STRUCTURE:
            return "hcl_document"

        if concept == UniversalConcept.BLOCK:
            # Prefer block-specific naming when possible
            if node.type == "block":
                return self._block_display_name(node, content)
            return "hcl_block"

        if concept == UniversalConcept.COMMENT:
            line = node.start_point[0] + 1
            return f"comment_line_{line}"

        return f"unnamed_{concept.value}"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        node = captures.get("definition") or (list(captures.values())[0] if captures else None)
        if node is None:
            return ""
        start, end = node.start_byte, node.end_byte
        return content[start:end].decode("utf-8", errors="replace")

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        node = captures.get("definition") or (list(captures.values())[0] if captures else None)
        meta: dict[str, Any] = {"concept": concept.value, "language": self.language.value}
        if node is None:
            return meta

        if node.type == "block":
            btype, labels = self._block_header(node, content)
            if btype:
                meta["block_type"] = btype
            if labels:
                meta["labels"] = labels
            meta["path"] = self._block_path(btype, labels)
        elif node.type == "attribute":
            # Key
            key_node = captures.get("key")
            if key_node is not None:
                key = self.get_node_text(key_node, self._decode(content)).strip()
            else:
                key = self._attribute_key(node, content)

            meta["attribute"] = key
            meta["key"] = key

            # Parent path and full path
            parent_path = self._parent_block_path(node, content)
            meta["parent_path"] = parent_path
            if key:
                meta["path"] = f"{parent_path}.{key}" if parent_path else key

            # Value type analysis (coarse)
            value_node = captures.get("value")
            if value_node is None:
                # try to approximate: second child of attribute is expression
                if node.child_count >= 3:
                    value_node = node.child(2)
            if value_node is not None:
                vtype = self._classify_value_node(value_node)
                if vtype:
                    meta["value_type"] = vtype

        elif node.type == "object_elem":
            src = self._decode(content)
            # Extract inner key/value
            ik = captures.get("inner_key")
            iv = captures.get("inner_value")
            inner_key_raw = self.get_node_text(ik, src).strip() if ik else ""
            inner_key = self._strip_quotes(inner_key_raw)

            # Find parent attribute and block path
            attr_parent = node.parent
            while attr_parent is not None and attr_parent.type != "attribute":
                attr_parent = attr_parent.parent
            attr_key = self._attribute_key(attr_parent, content) if attr_parent else ""
            parent_path = self._parent_block_path(node, content)

            if attr_key:
                meta["key"] = f"{attr_key}.{inner_key}" if inner_key else attr_key
            else:
                meta["key"] = inner_key

            meta["parent_path"] = parent_path
            parts = [p for p in [parent_path, attr_key, inner_key] if p]
            if parts:
                meta["path"] = ".".join(parts)

            if iv is not None:
                vtype = self._classify_value_node(iv)
                if vtype:
                    meta["value_type"] = vtype

        return meta

    # --- Helpers ---
    def _decode(self, content: bytes) -> str:
        return content.decode("utf-8", errors="replace")

    def _strip_quotes(self, text: str) -> str:
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
        return text

    def _block_header(self, node: Node, content: bytes) -> tuple[str, List[str]]:
        """Extract block type and labels from a `block` node.

        Structure: identifier (type), then 0..n labels (string_lit or identifier) until `{`.
        """
        src = self._decode(content)
        if node.type != "block" or node.child_count == 0:
            return "", []

        # child(0) should be the type identifier
        btype = ""
        labels: List[str] = []
        for i in range(node.child_count):
            child = node.child(i)
            if child is None:
                continue
            if i == 0 and child.type == "identifier":
                btype = self.get_node_text(child, src).strip()
                continue
            # labels appear before block_start
            if child.type in {"string_lit", "identifier"}:
                # string_lit tokens include quotes via external scanner
                text = self.get_node_text(child, src).strip()
                labels.append(self._strip_quotes(text))
                continue
            if child.type == "block_start":
                break
        return btype, labels

    def _block_path(self, btype: str, labels: List[str]) -> str:
        if not btype:
            return "block"
        if labels:
            return ".".join([btype] + labels)
        return btype

    def _block_display_name(self, node: Node, content: bytes) -> str:
        btype, labels = self._block_header(node, content)
        path = self._block_path(btype, labels)
        return path

    def _attribute_key(self, node: Node, content: bytes) -> str:
        src = self._decode(content)
        if node.type != "attribute" or node.child_count == 0:
            return ""
        # attribute := identifier '=' expression
        first = node.child(0)
        if first and first.type == "identifier":
            return self.get_node_text(first, src).strip()
        return ""

    def _parent_block_path(self, node: Node, content: bytes) -> str:
        # Walk up to nearest block ancestor and format its path
        parent = node.parent
        while parent is not None and parent.type != "block":
            parent = parent.parent
        if parent is None:
            return ""
        btype, labels = self._block_header(parent, content)
        return self._block_path(btype, labels)

    def _classify_value_node(self, node: Node) -> str:
        """Map HCL expression node to a coarse value type for metadata."""
        t = node.type
        # Primitive literals
        if t == "numeric_lit":
            return "number"
        if t == "bool_lit":
            return "bool"
        if t == "null_lit":
            return "null"
        if t == "string_lit":
            return "string"
        # Collections
        if t == "tuple":
            return "array"
        if t == "object":
            return "object"
        # Combined literal wrapper
        if t == "literal_value":
            try:
                if node.child_count > 0:
                    inner = node.child(0)
                    if inner is not None:
                        return self._classify_value_node(inner)
            except Exception:
                pass
            return "literal"
        # Expressions
        if t == "variable_expr":
            return "variable"
        if t == "function_call":
            return "function"
        if t == "template_expr" or t == "quoted_template" or t == "heredoc_template":
            return "template"
        # Fallback / unwrap generic expression nodes to inspect inner literal
        if t == "expression":
            try:
                if node.child_count > 0:
                    inner = node.child(0)
                    if inner is not None:
                        return self._classify_value_node(inner)
            except Exception:
                pass
            return "expression"
        return t
