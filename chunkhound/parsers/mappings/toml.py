"""TOML language mapping for unified parser architecture.

This module provides TOML-specific parsing and extraction logic
for the universal concept system. Since tree-sitter-toml is not available,
it uses Python's built-in tomllib/toml module with custom structure detection.
"""

import sys
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

# Handle tomllib availability (Python 3.11+)
if sys.version_info >= (3, 11):
    import tomllib

    HAS_TOMLLIB = True
else:
    try:
        import tomli as tomllib

        HAS_TOMLLIB = True
    except ImportError:
        HAS_TOMLLIB = False
        tomllib = None


class TomlMapping(BaseMapping):
    """TOML-specific mapping for universal concepts using built-in TOML parsing."""

    def __init__(self) -> None:
        """Initialize TOML mapping."""
        super().__init__(Language.TOML)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions (not applicable to TOML)."""
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to TOML)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments (TOML supports comments but no tree-sitter)."""
        return ""

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node (not applicable to TOML)."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to TOML)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in TOML.

        Returns valid tree-sitter queries that work with tree-sitter-toml grammar.
        """
        if concept == UniversalConcept.DEFINITION:
            # Match TOML tables and key-value pairs as definitions
            return """
                (table) @definition
                
                (pair) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            # Match string values and other content blocks
            return """
                (string) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            # No document-level captures - let cAST handle structure
            return ""

        elif concept == UniversalConcept.COMMENT:
            # Match TOML comments
            return """
                (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            # TOML doesn't have imports, but we provide empty query for consistency
            return ""

        else:
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for TOML parsing
        source = content.decode("utf-8")

        if not HAS_TOMLLIB:
            return "toml_unavailable"

        try:
            data = tomllib.loads(source)

            if concept == UniversalConcept.DEFINITION:
                # For TOML tables and key-value pairs
                if isinstance(data, dict):
                    # Look for common naming patterns at root level
                    for key in ["name", "title", "package", "project"]:
                        if key in data:
                            if isinstance(data[key], dict) and "name" in data[key]:
                                return f"definition_{data[key]['name']}"
                            elif isinstance(data[key], str):
                                return f"definition_{data[key]}"

                    # Use first key as name
                    first_key = next(iter(data.keys()), None)
                    if first_key:
                        return f"definition_{first_key}"

                return "toml_definition"

            elif concept == UniversalConcept.BLOCK:
                # Look for table structures
                if isinstance(data, dict):
                    # Count nested tables
                    table_count = sum(1 for v in data.values() if isinstance(v, dict))
                    if table_count > 0:
                        return "toml_tables"
                    else:
                        return "toml_kvpairs"

                return "toml_block"

            elif concept == UniversalConcept.STRUCTURE:
                return "toml_document"

        except Exception:
            # Handle TOML parsing errors
            pass

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept.

        Extracts only the matched node's content to avoid duplication.
        """
        # Extract only the specific captured node's content
        def_node = captures.get("definition") or (
            list(captures.values())[0] if captures else None
        )
        if not def_node:
            return ""

        # Return only the matched node's text, not the entire file
        return content.decode("utf-8")[def_node.start_byte : def_node.end_byte]

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract TOML-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        if not HAS_TOMLLIB:
            metadata["parser_unavailable"] = True
            metadata["parser_note"] = "tomllib/tomli not available"
            return metadata

        try:
            data = tomllib.loads(source)
            # Store parsed data in metadata for structured access if needed
            metadata["parsed_toml"] = data

            if concept == UniversalConcept.DEFINITION:
                if isinstance(data, dict):
                    metadata["data_type"] = "table"
                    metadata["key_count"] = len(data)

                    # Analyze key and value types
                    value_types = set()
                    table_keys = []
                    scalar_keys = []
                    array_keys = []

                    for key, value in data.items():
                        value_type = type(value).__name__
                        value_types.add(value_type)

                        if isinstance(value, dict):
                            table_keys.append(key)
                        elif isinstance(value, list):
                            array_keys.append(key)
                        else:
                            scalar_keys.append(key)

                    metadata["value_types"] = list(value_types)
                    if table_keys:
                        metadata["table_keys"] = table_keys
                    if array_keys:
                        metadata["array_keys"] = array_keys
                    if scalar_keys:
                        metadata["scalar_keys"] = scalar_keys

                    # Detect common TOML file types
                    if "package" in data or "project" in data:
                        metadata["toml_type"] = "project_config"
                    elif "tool" in data:
                        metadata["toml_type"] = "tool_config"
                    elif any(key in data for key in ["server", "database", "client"]):
                        metadata["toml_type"] = "app_config"
                    elif "build-system" in data or "dependencies" in data:
                        metadata["toml_type"] = "build_config"

                    # Analyze nested structure depth
                    max_depth = self._calculate_toml_depth(data)
                    metadata["max_depth"] = max_depth

            elif concept == UniversalConcept.BLOCK:
                metadata["toml_type"] = type(data).__name__

                if isinstance(data, dict):
                    # Count different types of content
                    metadata["nested_tables"] = sum(
                        1 for v in data.values() if isinstance(v, dict)
                    )
                    metadata["arrays"] = sum(
                        1 for v in data.values() if isinstance(v, list)
                    )
                    metadata["scalars"] = sum(
                        1 for v in data.values() if not isinstance(v, (dict, list))
                    )

                    # Detect array of tables pattern
                    array_of_tables = []
                    for key, value in data.items():
                        if (
                            isinstance(value, list)
                            and value
                            and isinstance(value[0], dict)
                        ):
                            array_of_tables.append(key)

                    if array_of_tables:
                        metadata["array_of_tables"] = array_of_tables

            elif concept == UniversalConcept.STRUCTURE:
                metadata["root_type"] = type(data).__name__

                # Calculate overall document statistics
                total_keys = self._count_toml_keys(data)
                metadata["total_keys"] = total_keys

                max_depth = self._calculate_toml_depth(data)
                metadata["max_depth"] = max_depth

                # Detect comments in source (since we can't parse them structurally)
                comment_count = source.count("#")
                if comment_count > 0:
                    metadata["comment_lines"] = comment_count

        except Exception as e:
            metadata["parse_error"] = str(e)
            metadata["is_valid_toml"] = False

        return metadata

    def _calculate_toml_depth(self, data: Any, current_depth: int = 1) -> int:
        """Calculate maximum depth of TOML structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_toml_depth(v, current_depth + 1) for v in data.values()
            )
        elif isinstance(data, list):
            if not data:
                return current_depth
            max_item_depth = current_depth
            for item in data:
                item_depth = self._calculate_toml_depth(item, current_depth + 1)
                max_item_depth = max(max_item_depth, item_depth)
            return max_item_depth
        else:
            return current_depth

    def _count_toml_keys(self, data: Any) -> int:
        """Count total number of keys in TOML structure."""
        if isinstance(data, dict):
            return len(data) + sum(self._count_toml_keys(v) for v in data.values())
        elif isinstance(data, list):
            return sum(self._count_toml_keys(item) for item in data)
        else:
            return 0

    def _extract_toml_comments(self, source: str) -> list[dict[str, Any]]:
        """Extract comments from TOML source (simple line-based extraction)."""
        comments = []
        lines = source.split("\n")

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                comments.append(
                    {
                        "line": line_num,
                        "content": stripped[1:].strip(),
                        "full_line": line,
                    }
                )

        return comments

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Data formats don't have imports."""
        return None

    def extract_constants(
        self,
        concept: Any,
        captures: dict[str, Any],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract TOML table names and key-value pairs with scalar values.

        Returns constant definitions for:
        - Table names as constants
        - Key-value pairs with scalar values

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

        if not HAS_TOMLLIB:
            return None

        try:
            data = tomllib.loads(source)

            if not isinstance(data, dict):
                return None

            constants = []

            def extract_from_dict(d: dict, prefix: str = "") -> None:
                """Recursively extract constants from nested dict."""
                for key, value in d.items():
                    full_key = f"{prefix}.{key}" if prefix else key

                    if isinstance(value, dict):
                        # Add table name as constant
                        constants.append({"name": full_key, "value": "[table]"})
                        # Recurse into nested table
                        extract_from_dict(value, full_key)
                    elif isinstance(value, list):
                        # Handle arrays of tables (list of dicts)
                        if value and isinstance(value[0], dict):
                            for idx, item in enumerate(value):
                                if isinstance(item, dict):
                                    # Add array element marker
                                    indexed_key = f"{full_key}[{idx}]"
                                    constants.append(
                                        {"name": indexed_key, "value": "[array_table]"}
                                    )
                                    # Recurse into array element
                                    extract_from_dict(item, indexed_key)
                    elif value is not None and isinstance(
                        value, (str, int, float, bool)
                    ):
                        # Add scalar value
                        value_str = str(value)
                        if len(value_str) > MAX_CONSTANT_VALUE_LENGTH:
                            value_str = value_str[:MAX_CONSTANT_VALUE_LENGTH]

                        constants.append({"name": full_key, "value": value_str})

            extract_from_dict(data)

            return constants if constants else None

        except Exception:
            return None
