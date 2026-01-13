"""YAML language mapping for unified parser architecture.

This module provides YAML-specific tree-sitter queries and extraction logic
for the universal concept system. It handles YAML documents, mappings, sequences,
and other structural elements using tree-sitter-yaml.
"""

from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class YamlMapping(BaseMapping):
    """YAML-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize YAML mapping."""
        super().__init__(Language.YAML)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions (not applicable to YAML)."""
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to YAML)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node (not applicable to YAML)."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to YAML)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in YAML."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (block_mapping_pair
                key: (flow_node) @key
                value: (_) @value
            ) @definition
            
            (flow_pair
                key: (flow_node) @key
                value: (_) @value
            ) @definition
            
            (block_sequence_item
                (flow_node) @item
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (block_mapping) @definition
            
            (flow_mapping) @definition
            
            (block_sequence) @definition
            
            (flow_sequence) @definition
            
            (document) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            # YAML doesn't have explicit imports, but we can look for common patterns
            return """
            (block_mapping_pair
                key: (flow_node) @key
                value: (_) @value
                (#match? @key "^(include|import|extends|inherit)$")
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (document) @definition
            
            (stream) @definition
            """

        # All cases handled above
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # For key-value pairs, use the key as the name
            if "key" in captures:
                key_node = captures["key"]
                key_text = self.get_node_text(key_node, source).strip()
                # Clean up YAML key format
                key_text = key_text.strip("\"'")
                if key_text.endswith(":"):
                    key_text = key_text[:-1]
                return key_text

            # For sequence items, use a generic name
            if "item" in captures:
                item_node = captures["item"]
                line = item_node.start_point[0] + 1
                return f"item_line_{line}"

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                block_type = node.type.replace("_", " ")
                return f"{block_type}_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use location-based naming for comments
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "key" in captures and "value" in captures:
                key_node = captures["key"]
                value_node = captures["value"]

                key_text = self.get_node_text(key_node, source).strip().strip("\"'")
                value_text = self.get_node_text(value_node, source).strip().strip("\"'")

                return f"{key_text}_{value_text}"

            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            if "definition" in captures:
                node = captures["definition"]
                if node.type == "document":
                    # Check if it's a multi-document stream
                    line = node.start_point[0] + 1
                    return f"document_{line}"
                elif node.type == "stream":
                    return "yaml_stream"

            return "yaml_structure"

        # All cases handled above
        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif captures:
            # Use the first available capture
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract YAML-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For mapping pairs
                if "key" in captures and "value" in captures:
                    key_node = captures["key"]
                    value_node = captures["value"]

                    key_text = self.get_node_text(key_node, source).strip().strip("\"'")
                    value_text = self.get_node_text(value_node, source).strip()

                    metadata["kind"] = "mapping_pair"
                    metadata["key"] = key_text

                    # Analyze value type
                    value_type = self._analyze_yaml_value_type(value_node, source)
                    metadata["value_type"] = value_type

                    # Store short values for reference
                    if len(value_text) < 100 and value_type in ["scalar", "string"]:
                        metadata["value"] = value_text.strip("\"'")

                    # Check for common configuration patterns
                    if key_text.lower() in ["name", "title", "id", "version"]:
                        metadata["config_type"] = "identifier"
                    elif key_text.lower() in ["host", "port", "url", "database"]:
                        metadata["config_type"] = "connection"
                    elif key_text.lower() in ["env", "environment", "stage"]:
                        metadata["config_type"] = "environment"

                # For sequence items
                elif "item" in captures:
                    item_node = captures["item"]
                    metadata["kind"] = "sequence_item"

                    item_type = self._analyze_yaml_value_type(item_node, source)
                    metadata["item_type"] = item_type

        elif concept == UniversalConcept.BLOCK:
            if "definition" in captures:
                block_node = captures["definition"]
                metadata["block_type"] = block_node.type

                # Analyze block content
                if block_node.type in ["block_mapping", "flow_mapping"]:
                    # Count key-value pairs
                    pairs = self._count_mapping_pairs(block_node)
                    metadata["pair_count"] = pairs
                    metadata["structure_type"] = "mapping"

                elif block_node.type in ["block_sequence", "flow_sequence"]:
                    # Count sequence items
                    items = self._count_sequence_items(block_node)
                    metadata["item_count"] = items
                    metadata["structure_type"] = "sequence"

                elif block_node.type == "document":
                    # Analyze document structure
                    metadata["structure_type"] = "document"
                    metadata["has_directives"] = self._has_yaml_directives(
                        block_node, source
                    )

        elif concept == UniversalConcept.IMPORT:
            if "key" in captures and "value" in captures:
                key_node = captures["key"]
                value_node = captures["value"]

                key_text = self.get_node_text(key_node, source).strip().strip("\"'")
                value_text = self.get_node_text(value_node, source).strip().strip("\"'")

                metadata["import_type"] = key_text.lower()
                metadata["import_target"] = value_text

                # Determine if it's a file path or module reference
                if (
                    "/" in value_text
                    or "\\" in value_text
                    or value_text.endswith((".yaml", ".yml"))
                ):
                    metadata["target_type"] = "file"
                else:
                    metadata["target_type"] = "reference"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Clean and analyze comment
                clean_text = self.clean_comment_text(comment_text)

                # Detect special comment types
                comment_type = "regular"

                if clean_text:
                    upper_text = clean_text.upper()
                    if any(
                        prefix in upper_text
                        for prefix in ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]
                    ):
                        comment_type = "annotation"
                    elif clean_text.startswith("---") or clean_text.startswith("..."):
                        comment_type = "document_marker"
                    elif len(clean_text) > 50:
                        comment_type = "documentation"

                metadata["comment_type"] = comment_type

        elif concept == UniversalConcept.STRUCTURE:
            if "definition" in captures:
                structure_node = captures["definition"]
                metadata["structure_type"] = structure_node.type

                # Calculate structure complexity
                if structure_node.type == "document":
                    depth = self._calculate_yaml_depth(structure_node)
                    metadata["max_depth"] = depth
                elif structure_node.type == "stream":
                    doc_count = len(
                        [
                            child
                            for child in self.walk_tree(structure_node)
                            if child and child.type == "document"
                        ]
                    )
                    metadata["document_count"] = doc_count

        return metadata

    def _analyze_yaml_value_type(self, node: Node, source: str) -> str:
        """Analyze the type of a YAML value node."""
        if node.type in ["plain_scalar", "single_quote_scalar", "double_quote_scalar"]:
            return "scalar"
        elif node.type == "block_mapping":
            return "mapping"
        elif node.type == "flow_mapping":
            return "inline_mapping"
        elif node.type == "block_sequence":
            return "sequence"
        elif node.type == "flow_sequence":
            return "inline_sequence"
        elif node.type in ["literal_scalar", "folded_scalar"]:
            return "multiline_scalar"
        elif node.type == "anchor":
            return "anchor"
        elif node.type == "alias":
            return "alias"
        else:
            return node.type

    def _count_mapping_pairs(self, mapping_node: Node) -> int:
        """Count key-value pairs in a mapping node."""
        count = 0
        for child in self.walk_tree(mapping_node):
            if child and child.type in ["block_mapping_pair", "flow_pair"]:
                count += 1
        return count

    def _count_sequence_items(self, sequence_node: Node) -> int:
        """Count items in a sequence node."""
        count = 0
        for child in self.walk_tree(sequence_node):
            if child and child.type in ["block_sequence_item", "flow_sequence_item"]:
                count += 1
        return count

    def _has_yaml_directives(self, document_node: Node, source: str) -> bool:
        """Check if document has YAML directives (like %YAML, %TAG)."""
        doc_text = self.get_node_text(document_node, source)
        return doc_text.strip().startswith("%")

    def _calculate_yaml_depth(self, node: Node, current_depth: int = 1) -> int:
        """Calculate maximum nesting depth of YAML structure."""
        max_depth = current_depth

        for child in node.children if hasattr(node, "children") else []:
            if child and child.type in [
                "block_mapping",
                "flow_mapping",
                "block_sequence",
                "flow_sequence",
            ]:
                child_depth = self._calculate_yaml_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)

        return max_depth

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path
    ) -> Path | None:
        """Data formats don't have imports."""
        return None

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract top-level YAML keys and their scalar values, plus anchors.

        Returns constant definitions for:
        - Top-level keys with scalar values
        - Anchors as named values

        Args:
            concept: The UniversalConcept being extracted
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dicts with 'name' and 'value' keys, or None if not applicable
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        import yaml

        source = content.decode("utf-8")

        try:
            # Parse YAML to extract top-level key-value pairs from all documents
            constants = []

            for data in yaml.safe_load_all(source):
                if not isinstance(data, dict):
                    continue

                # Extract top-level scalar values
                for key, value in data.items():
                    # Only include scalar values (skip None, objects, arrays)
                    if value is not None and isinstance(value, (str, int, float, bool)):
                        # Convert value to string and truncate to 50 chars
                        value_str = str(value)
                        if len(value_str) > MAX_CONSTANT_VALUE_LENGTH:
                            value_str = value_str[:MAX_CONSTANT_VALUE_LENGTH]

                        constants.append({
                            "name": key,
                            "value": value_str
                        })

            # Extract anchors from source text
            # Anchors are defined as &anchor_name
            import re
            anchor_pattern = r'&([a-zA-Z_][a-zA-Z0-9_-]*)\s+(.*?)(?:\n|$)'
            for match in re.finditer(anchor_pattern, source):
                anchor_name = match.group(1)
                anchor_value = match.group(2).strip()

                # Truncate value to 50 chars
                if len(anchor_value) > MAX_CONSTANT_VALUE_LENGTH:
                    anchor_value = anchor_value[:MAX_CONSTANT_VALUE_LENGTH]

                constants.append({
                    "name": f"&{anchor_name}",
                    "value": anchor_value
                })

            return constants if constants else None

        except yaml.YAMLError:
            return None
