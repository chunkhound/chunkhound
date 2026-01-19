"""Lua language mapping for unified parser architecture.

This module provides Lua-specific tree-sitter queries and extraction logic
for the universal concept system. It maps Lua's AST nodes to universal
semantic concepts used by the unified parser.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class LuaMapping(BaseMapping):
    """Lua-specific tree-sitter mapping for universal concepts."""

    def __init__(self) -> None:
        """Initialize Lua mapping."""
        super().__init__(Language.LUA)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions.

        Note: In tree-sitter-lua, both global and local functions use
        function_declaration node type. Local functions have a 'local'
        child node.
        """
        return """
        (function_declaration
            name: (identifier) @func_name
        ) @func_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions.

        Lua doesn't have native classes, but tables are often used as classes.
        We capture table assignments that look like class definitions.
        """
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments."""
        return """
        (comment) @comment
        """

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node."""
        if node is None:
            return self.get_fallback_name(node, "function")

        # Try to find the function name from various child types
        for child_type in ["identifier", "dot_index_expression", "method_index_expression"]:
            name_node = self.find_child_by_type(node, child_type)
            if name_node:
                return self.get_node_text(name_node, source).strip()

        # Try getting name directly
        name_node = self.find_child_by_type(node, "name")
        if name_node:
            return self.get_node_text(name_node, source).strip()

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not native in Lua)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Lua.

        Note: In tree-sitter-lua:
        - Both global and local functions use function_declaration
        - Local functions have a 'local' child node
        - variable_declaration contains assignment_statement
        """

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_declaration
                name: [
                    (identifier)
                    (dot_index_expression)
                    (method_index_expression)
                ] @name
            ) @definition

            (variable_declaration) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (block) @block

            (do_statement
                (block) @block
            )

            (if_statement
                (block) @block
            )

            (for_statement
                (block) @block
            )

            (while_statement
                (block) @block
            )

            (repeat_statement
                (block) @block
            )
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (function_call
                name: (identifier) @func_name
                (#match? @func_name "^require$")
            ) @definition

            (function_call
                name: (identifier) @func_name
                (#match? @func_name "^dofile$")
            ) @definition

            (function_call
                name: (identifier) @func_name
                (#match? @func_name "^loadfile$")
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (chunk) @definition
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
            # Try to get the name from capture groups
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()
                return name

            # For variable_declaration, extract name from nested structure
            if "definition" in captures:
                def_node = captures["definition"]
                if def_node.type == "variable_declaration":
                    # Navigate: variable_declaration -> assignment_statement -> variable_list -> identifier
                    for child in def_node.children:
                        if child.type == "assignment_statement":
                            for subchild in child.children:
                                if subchild.type == "variable_list":
                                    for var in subchild.children:
                                        if var.type == "identifier":
                                            return self.get_node_text(var, source).strip()

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                block_type = node.type
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
            if "definition" in captures:
                def_node = captures["definition"]
                def_text = self.get_node_text(def_node, source).strip()
                # Extract the module name from require("module") or require 'module'
                match = re.search(r'require\s*[\(\s]*["\']([^"\']+)["\']', def_text)
                if match:
                    module_name = match.group(1)
                    # Get just the module name for cleaner names
                    if "." in module_name:
                        module_name = module_name.split(".")[-1]
                    return f"require_{module_name}"

                # Try dofile/loadfile
                match = re.search(r'(?:dofile|loadfile)\s*[\(\s]*["\']([^"\']+)["\']', def_text)
                if match:
                    file_name = match.group(1)
                    if "/" in file_name:
                        file_name = file_name.split("/")[-1]
                    return f"load_{file_name}"

            return "unnamed_require"

        elif concept == UniversalConcept.STRUCTURE:
            return "lua_chunk"

        # All cases handled above
        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.BLOCK and "block" in captures:
            node = captures["block"]
            return self.get_node_text(node, source)
        elif "definition" in captures:
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
        """Extract Lua-specific metadata."""

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For functions, extract basic info
                if def_node.type == "function_declaration":
                    metadata["kind"] = "function"

                    # Extract function body size as a complexity metric
                    body_node = self.find_child_by_type(def_node, "block")
                    if body_node:
                        body_text = self.get_node_text(body_node, source)
                        metadata["body_lines"] = len(body_text.splitlines())

                # For variable declarations
                elif def_node.type == "variable_declaration":
                    metadata["kind"] = "variable"

                # For variable assignments
                elif def_node.type == "assignment_statement":
                    metadata["kind"] = "variable"

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                block_node = captures["block"]
                metadata["block_type"] = block_node.type

                # Count statements in the block for complexity
                statements = 0
                for child in self.walk_tree(block_node):
                    if child and child.type in [
                        "function_call",
                        "assignment_statement",
                        "if_statement",
                        "for_statement",
                        "while_statement",
                        "repeat_statement",
                        "return_statement",
                    ]:
                        statements += 1
                metadata["statement_count"] = statements

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_text = self.get_node_text(import_node, source).strip()

                # Extract the module being required
                match = re.search(r'require\s*[\(\s]*["\']([^"\']+)["\']', import_text)
                if match:
                    metadata["module"] = match.group(1)
                    metadata["import_type"] = "require"
                else:
                    match = re.search(r'(?:dofile|loadfile)\s*[\(\s]*["\']([^"\']+)["\']', import_text)
                    if match:
                        metadata["file"] = match.group(1)
                        metadata["import_type"] = "dofile" if "dofile" in import_text else "loadfile"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Clean and analyze comment
                clean_text = self.clean_comment_text(comment_text)

                # Detect special comment types
                is_doc = False
                comment_type = "regular"

                if clean_text:
                    upper_text = clean_text.upper()
                    if any(
                        prefix in upper_text
                        for prefix in ["TODO:", "FIXME:", "HACK:", "NOTE:", "WARNING:"]
                    ):
                        comment_type = "annotation"
                        is_doc = True
                    elif clean_text.startswith("#!/"):
                        comment_type = "shebang"
                        is_doc = True
                    elif len(clean_text) > 50 and any(
                        word in clean_text.lower()
                        for word in ["function", "parameter", "return", "usage", "@param", "@return"]
                    ):
                        comment_type = "documentation"
                        is_doc = True

                metadata["comment_type"] = comment_type
                if is_doc:
                    metadata["is_doc_comment"] = True

        return metadata

    def clean_comment_text(self, text: str) -> str:
        """Clean Lua comment text by removing comment markers.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Remove Lua single-line comment marker
        if cleaned.startswith("--"):
            # Check for multi-line comment
            if cleaned.startswith("--[["):
                # Remove --[[ and ]]
                cleaned = cleaned[4:]
                if cleaned.endswith("]]"):
                    cleaned = cleaned[:-2]
            elif cleaned.startswith("--[=["):
                # Long bracket comment --[=[ ... ]=]
                cleaned = cleaned[5:]
                if cleaned.endswith("]=]"):
                    cleaned = cleaned[:-3]
            else:
                # Simple single-line comment
                cleaned = cleaned[2:]

        return cleaned.strip()

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path from Lua require/dofile/loadfile.

        Args:
            import_text: The text of the import statement
            base_dir: Base directory of the indexed codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise
        """
        # require("module.submodule") -> module/submodule.lua
        match = re.search(r'require\s*[\(\s]*["\']([^"\']+)["\']', import_text)
        if match:
            module_path = match.group(1).replace(".", "/") + ".lua"

            # Try relative to source file first
            resolved = (source_file.parent / module_path).resolve()
            if resolved.exists():
                return resolved

            # Try relative to base directory
            full_path = base_dir / module_path
            if full_path.exists():
                return full_path

        # dofile/loadfile with direct path
        match = re.search(r'(?:dofile|loadfile)\s*[\(\s]*["\']([^"\']+)["\']', import_text)
        if match:
            path = match.group(1)

            # Try relative to source file first
            if path.startswith("./") or path.startswith("../"):
                resolved = (source_file.parent / path).resolve()
                if resolved.exists():
                    return resolved

            # Try relative to base directory
            full_path = base_dir / path
            if full_path.exists():
                return full_path

        return None

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Lua code.

        Identifies UPPER_SNAKE_CASE variable assignments as constants.
        Lua constants like: local MAX_RETRIES = 3, CONFIG_VALUE = "production"

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name' and 'value' keys, or None
        """
        if concept != UniversalConcept.DEFINITION:
            return None

        source = content.decode("utf-8")

        # Get the definition node to extract constant name and value
        def_node = captures.get("definition")
        if not def_node or def_node.type != "variable_declaration":
            return None

        # Navigate: variable_declaration -> assignment_statement -> variable_list -> identifier
        for child in def_node.children:
            if child.type == "assignment_statement":
                var_list = self.find_child_by_type(child, "variable_list")
                expr_list = self.find_child_by_type(child, "expression_list")

                if var_list:
                    for var in var_list.children:
                        if var.type == "identifier":
                            name = self.get_node_text(var, source).strip()
                            # Match UPPER_SNAKE_CASE pattern
                            if name and re.match(r"^_?[A-Z][A-Z0-9_]*$", name):
                                value = ""
                                if expr_list:
                                    value = self.get_node_text(expr_list, source).strip()
                                    if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                                        value = value[:MAX_CONSTANT_VALUE_LENGTH]
                                return [{"name": name, "value": value}]

        return None
