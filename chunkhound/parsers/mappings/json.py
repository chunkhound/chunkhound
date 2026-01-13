"""JSON language mapping for unified parser architecture.

This module provides JSON-specific parsing and extraction logic
for the universal concept system. Since tree-sitter-json is not available,
it uses Python's built-in json module with custom structure detection.
"""

import json
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class JsonMapping(BaseMapping):
    """JSON-specific mapping for universal concepts using built-in JSON parsing."""

    def __init__(self) -> None:
        """Initialize JSON mapping."""
        super().__init__(Language.JSON)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions (not applicable to JSON)."""
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to JSON)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments (JSON doesn't support comments)."""
        return ""

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node (not applicable to JSON)."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to JSON)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in JSON.

        Returns valid tree-sitter queries that work with tree-sitter-json grammar.
        """
        if concept == UniversalConcept.DEFINITION:
            # Match individual key-value pairs and primitive values as definitions
            return """
                (pair) @definition
                
                (string) @definition
                
                (number) @definition
                
                (true) @definition
                
                (false) @definition
                
                (null) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            # Match small container structures
            return """
                (object) @definition
                
                (array) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            # No document-level captures - let cAST handle structure
            return ""

        elif concept == UniversalConcept.COMMENT:
            # JSON doesn't have comments, but we provide empty query for consistency
            return ""

        elif concept == UniversalConcept.IMPORT:
            # JSON doesn't have imports, but we provide empty query for consistency
            return ""

        else:
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for JSON parsing
        source = content.decode("utf-8")

        try:
            data = json.loads(source)

            if concept == UniversalConcept.DEFINITION:
                # For JSON objects with identifiable keys
                if isinstance(data, dict):
                    # Look for common naming patterns
                    for key in ["name", "id", "title", "key", "label"]:
                        if key in data and isinstance(data[key], str):
                            return f"definition_{data[key]}"

                    # Use first key as name
                    first_key = next(iter(data.keys()), None)
                    if first_key:
                        return f"definition_{first_key}"

                elif isinstance(data, list) and data:
                    # For arrays, use first element if it has a name
                    first_item = data[0]
                    if isinstance(first_item, dict):
                        for key in ["name", "id", "title"]:
                            if key in first_item and isinstance(first_item[key], str):
                                return f"array_{first_item[key]}"
                    return "array_definition"

                return "json_definition"

            elif concept == UniversalConcept.BLOCK:
                if isinstance(data, dict):
                    return "object_block"
                elif isinstance(data, list):
                    return "array_block"
                return "json_block"

            elif concept == UniversalConcept.STRUCTURE:
                return "json_document"

        except json.JSONDecodeError:
            pass

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Get the specific node to extract
        def_node = (
            captures.get("definition") or captures.get("block") or captures.get("node")
        )
        if not def_node and captures:
            def_node = list(captures.values())[0]

        if not def_node:
            return ""

        # Decode source
        source = content.decode("utf-8")

        # Return ONLY this node's content, not the entire file
        # This allows cAST merging to work properly
        node_content = source[def_node.start_byte : def_node.end_byte]

        return node_content

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract JSON-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        try:
            data = json.loads(source)

            if concept == UniversalConcept.DEFINITION:
                if isinstance(data, dict):
                    metadata["data_type"] = "object"
                    metadata["key_count"] = len(data)

                    # Analyze key types and patterns
                    key_types = set()
                    value_types = set()

                    for key, value in data.items():
                        key_types.add(type(key).__name__)
                        value_types.add(type(value).__name__)

                    metadata["key_types"] = list(key_types)
                    metadata["value_types"] = list(value_types)

                    # Check for common schemas
                    if any(key in data for key in ["name", "version", "description"]):
                        metadata["schema_hint"] = "package"
                    elif any(key in data for key in ["id", "title", "content"]):
                        metadata["schema_hint"] = "document"
                    elif any(key in data for key in ["host", "port", "database"]):
                        metadata["schema_hint"] = "config"

                elif isinstance(data, list):
                    metadata["data_type"] = "array"
                    metadata["item_count"] = len(data)

                    if data:
                        # Analyze array item types
                        item_types = set()
                        for item in data:
                            item_types.add(type(item).__name__)
                        metadata["item_types"] = list(item_types)

                        # Check if it's an array of objects with consistent schema
                        if all(isinstance(item, dict) for item in data):
                            if data:
                                keys = set(data[0].keys())
                                if all(set(item.keys()) == keys for item in data[1:]):
                                    metadata["schema_consistent"] = True
                                    metadata["object_keys"] = list(keys)

            elif concept == UniversalConcept.BLOCK:
                metadata["json_type"] = type(data).__name__

                if isinstance(data, dict):
                    metadata["nested_objects"] = sum(
                        1 for v in data.values() if isinstance(v, dict)
                    )
                    metadata["nested_arrays"] = sum(
                        1 for v in data.values() if isinstance(v, list)
                    )
                elif isinstance(data, list):
                    metadata["nested_objects"] = sum(
                        1 for item in data if isinstance(item, dict)
                    )
                    metadata["nested_arrays"] = sum(
                        1 for item in data if isinstance(item, list)
                    )

            elif concept == UniversalConcept.STRUCTURE:
                metadata["root_type"] = type(data).__name__

                # Calculate depth and complexity
                depth = self._calculate_json_depth(data)
                metadata["max_depth"] = depth

                # Count total elements
                total_elements = self._count_json_elements(data)
                metadata["total_elements"] = total_elements

        except json.JSONDecodeError as e:
            metadata["parse_error"] = str(e)
            metadata["is_valid_json"] = False
        else:
            # Store formatted version in metadata for cases where pretty-printed JSON is needed
            try:
                metadata["formatted_json"] = json.dumps(data, indent=2)
            except:
                pass  # Skip if formatting fails

        return metadata

    def _calculate_json_depth(self, data: Any, current_depth: int = 1) -> int:
        """Calculate maximum depth of JSON structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_json_depth(v, current_depth + 1) for v in data.values()
            )
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(
                self._calculate_json_depth(item, current_depth + 1) for item in data
            )
        else:
            return current_depth

    def _count_json_elements(self, data: Any) -> int:
        """Count total number of elements in JSON structure."""
        if isinstance(data, dict):
            return 1 + sum(self._count_json_elements(v) for v in data.values())
        elif isinstance(data, list):
            return 1 + sum(self._count_json_elements(item) for item in data)
        else:
            return 1

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
        """Extract top-level JSON object keys and their scalar values.

        Returns constant definitions for:
        - Top-level object keys with scalar values (string, number, boolean)
        - Skips nested objects/arrays as values

        Args:
            concept: The UniversalConcept being extracted
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dicts with 'name' and 'value' keys, or None if not applicable
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        source = content.decode("utf-8")

        try:
            data = json.loads(source)

            # Only extract from top-level objects
            if not isinstance(data, dict):
                return None

            constants = []
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

            return constants if constants else None

        except json.JSONDecodeError:
            return None
