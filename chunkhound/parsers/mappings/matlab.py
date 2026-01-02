"""MATLAB language mapping for unified parser architecture.

This module provides MATLAB-specific tree-sitter queries and extraction logic
for mapping MATLAB AST nodes to semantic chunks.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class MatlabMapping(BaseMapping):
    """MATLAB-specific tree-sitter mapping implementation.

    Handles MATLAB's unique language features including:
    - Function definitions with multiple return values
    - Class definitions with properties and methods
    - Scripts (files without function definitions)
    - Comments (% and %%)
    - Section headers (%%)
    - Help text/docstrings
    - Function handles
    - Nested functions
    """

    def __init__(self) -> None:
        """Initialize MATLAB mapping."""
        super().__init__(Language.MATLAB)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB function definitions.

        Captures function definitions with their names and signature components.

        Returns:
            Tree-sitter query string for finding MATLAB function definitions
        """
        return """
            (function_definition
                (function_output)? @function_output
                (identifier) @function_name
                (function_arguments)? @function_arguments
            ) @function_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB class definitions.

        Captures classdef statements with inheritance.

        Returns:
            Tree-sitter query string for finding MATLAB class definitions
        """
        return """
            (class_definition
                (identifier) @class_name
                (superclasses)? @superclasses
            ) @class_def
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB method definitions.

        Methods are function definitions within class bodies.

        Returns:
            Tree-sitter query string for finding MATLAB method definitions
        """
        return """
            (class_definition
                (methods
                    (function_definition
                        (function_output)? @method_output
                        (identifier) @method_name
                        (function_arguments)? @method_arguments
                    ) @method_def
                )
            )
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB comments.

        Captures both single-line (%) and section (%%) comments.

        Returns:
            Tree-sitter query string for finding MATLAB comments
        """
        return """
            (comment) @comment
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB help text/docstrings.

        Captures section comments (%%) and help text blocks.

        Returns:
            Tree-sitter query string for finding MATLAB help text
        """
        return """
            (function_definition
                body: (
                    . (comment) @function_docstring
                )
            )
            (class_definition
                body: (
                    . (comment) @class_docstring
                )
            )
            (comment) @section_comment
        """

    def get_property_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB class properties.

        Returns:
            Tree-sitter query string for finding MATLAB property definitions
        """
        return """
            (properties
                (property
                    (identifier) @property_name
                ) @property_def
            )
        """

    def get_script_query(self) -> str:
        """Get tree-sitter query pattern for MATLAB script content.

        Returns:
            Tree-sitter query string for finding script-level content
        """
        return """
            (source_file) @script
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a MATLAB function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        # Look for the name child node
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a MATLAB class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "class")

        # Look for the name child node
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "class")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names from a MATLAB function/method node.

        Handles regular parameters and varargin.

        Args:
            node: Tree-sitter function/method definition node
            source: Source code string

        Returns:
            List of parameter names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters: list[str] = []

        # Find the function_arguments node
        args_node = self.find_child_by_type(node, "function_arguments")
        if not args_node:
            return parameters

        # Extract identifiers from arguments
        for identifier_node in self.find_children_by_type(args_node, "identifier"):
            param_name = self.get_node_text(identifier_node, source).strip()
            if param_name and param_name not in ("(", ")", ","):
                parameters.append(param_name)

        return parameters

    def extract_return_values(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract return value names from a MATLAB function definition.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            List of return value names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        return_values: list[str] = []

        # Find the function_output node
        output_node = self.find_child_by_type(node, "function_output")
        if not output_node:
            return return_values

        # Extract identifiers from output
        for identifier_node in self.find_children_by_type(output_node, "identifier"):
            return_name = self.get_node_text(identifier_node, source).strip()
            if return_name and return_name not in ("[", "]", ","):
                return_values.append(return_name)

        return return_values

    def extract_superclasses(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract superclass names from a MATLAB class definition.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            List of superclass names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        superclasses: list[str] = []

        # Find the superclasses node
        superclass_node = self.find_child_by_type(node, "superclasses")
        if not superclass_node:
            # Try alternative approach - parse class definition line
            class_text = self.get_node_text(node, source)
            lines = class_text.split("\n")
            if lines:
                first_line = lines[0].strip()
                if "<" in first_line:
                    # Extract inheritance from "classdef ClassName < BaseClass"
                    parts = first_line.split("<", 1)
                    if len(parts) > 1:
                        base_classes = parts[1].strip()
                        # Handle multiple inheritance separated by &
                        superclasses = [cls.strip() for cls in base_classes.split("&")]
            return superclasses

        # Extract identifiers from superclass list
        for identifier_node in self.find_children_by_type(
            superclass_node, "identifier"
        ):
            superclass_name = self.get_node_text(identifier_node, source).strip()
            if superclass_name and superclass_name not in ("&", ","):
                superclasses.append(superclass_name)

        return superclasses

    def extract_properties(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract property names from a MATLAB class definition.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            List of property names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        properties: list[str] = []

        # Find all properties nodes within the class
        for properties_block in self.find_nodes_by_type(node, "properties"):
            # Find property definitions within each block
            for property_def in self.find_nodes_by_type(properties_block, "property"):
                # Extract the property name
                name_node = self.find_child_by_type(property_def, "identifier")
                if name_node:
                    prop_name = self.get_node_text(name_node, source).strip()
                    if prop_name:
                        properties.append(prop_name)

        return properties

    def is_script_file(self, node: "TSNode | None", source: str) -> bool:
        """Determine if this is a MATLAB script file (no function definitions).

        Args:
            node: Tree-sitter root node
            source: Source code string

        Returns:
            True if this is a script file, False if it's a function file
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return True

        # Check for top-level function definitions
        for child in node.children:
            if child and child.type == "function_definition":
                return False

        return True

    def is_help_comment(self, node: "TSNode | None", source: str) -> bool:
        """Check if a comment node is MATLAB help text (starts with %%).

        Args:
            node: Tree-sitter comment node
            source: Source code string

        Returns:
            True if this is help text, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        comment_text = self.get_node_text(node, source).strip()
        return comment_text.startswith("%%")

    def is_function_handle(self, node: "TSNode | None", source: str) -> bool:
        """Check if a node represents a MATLAB function handle.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if this is a function handle, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Function handles start with @
        node_text = self.get_node_text(node, source).strip()
        return node_text.startswith("@")

    def clean_comment_text(self, text: str) -> str:
        """Clean MATLAB comment text by removing comment markers.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Remove MATLAB comment markers
        if cleaned.startswith("%%"):
            cleaned = cleaned[2:].strip()
        elif cleaned.startswith("%"):
            cleaned = cleaned[1:].strip()

        return cleaned

    def create_function_signature(
        self, name: str, parameters: list[str], return_values: list[str] | None = None
    ) -> str:
        """Create a MATLAB-style function signature string.

        Args:
            name: Function name
            parameters: List of parameter names
            return_values: List of return value names

        Returns:
            MATLAB-style function signature
        """
        param_str = ", ".join(parameters) if parameters else ""

        if return_values:
            if len(return_values) == 1:
                return f"{return_values[0]} = {name}({param_str})"
            else:
                return f"[{', '.join(return_values)}] = {name}({param_str})"
        else:
            return f"{name}({param_str})"

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for universal concept in MATLAB.

        Extends default DEFINITION query to include properties blocks and
        top-level assignments for constant extraction.

        Args:
            concept: The universal concept to query

        Returns:
            Tree-sitter query string or None if concept not supported
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return None

        if concept == UniversalConcept.DEFINITION:
            # Combine function, class, properties, and top-level assignment queries
            # Note: Only capture top-level assignments (children of source_file),
            # not assignments inside function bodies (those are extracted via function_definition)
            return """
                (function_definition
                    (function_output)? @function_output
                    (identifier) @function_name
                    (function_arguments)? @function_arguments
                ) @definition

                (class_definition
                    (identifier) @class_name
                    (superclasses)? @superclasses
                ) @definition

                (properties) @definition

                (source_file
                    (assignment
                        (identifier) @assignment_name
                    ) @definition
                )
            """

        return None

    def extract_name(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for MATLAB nodes.

        Handles properties blocks and top-level assignments for constant extraction.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted name string
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return "unnamed"

        if concept != UniversalConcept.DEFINITION:
            return "unnamed"

        source = content.decode("utf-8")

        def_node = captures.get("definition")
        if not def_node:
            return "unnamed"

        # Handle top-level assignments (check node type)
        if def_node.type == "assignment":
            # Get the identifier from assignment
            assignment_name = captures.get("assignment_name")
            if assignment_name:
                name = self.get_node_text(assignment_name, source).strip()
                # Only include UPPER_CASE assignments
                if re.match(r'^[A-Z][A-Z0-9_]*$', name):
                    return name
            # If not UPPER_CASE, skip by returning a dummy name
            return f"_skip_assignment_{def_node.start_point[0] + 1}"

        # Handle properties blocks
        if def_node.type == "properties":
            # Name based on line number
            line = def_node.start_point[0] + 1
            return f"properties_line_{line}"

        # Fall back to standard extraction
        if def_node.type == "function_definition":
            return self.extract_function_name(def_node, source)
        elif def_node.type == "class_definition":
            return self.extract_class_name(def_node, source)

        return self.get_fallback_name(def_node, "definition")

    def extract_content(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract content from captures for MATLAB nodes.

        Filters out non-UPPER_CASE assignments to avoid creating unnecessary chunks.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted content string, or empty string for filtered assignments
        """
        def_node = captures.get("definition")
        if not def_node:
            return ""

        source = content.decode("utf-8")

        # Filter out non-UPPER_CASE assignments by returning empty content
        if def_node.type == "assignment":
            assignment_name = captures.get("assignment_name")
            if assignment_name:
                name = self.get_node_text(assignment_name, source).strip()
                # Only include UPPER_CASE assignments
                if not re.match(r'^[A-Z][A-Z0-9_]*$', name):
                    return ""  # Empty content will filter out this chunk

        return self.get_node_text(def_node, source)

    def extract_metadata(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract metadata from captures for MATLAB nodes.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Dictionary of metadata
        """
        return {}

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a MATLAB node should be included as a chunk.

        Filters out very small nodes and empty function/class definitions.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Get the node text to check size
        text = self.get_node_text(node, source)

        # Skip very small nodes (less than 20 characters)
        if len(text.strip()) < 20:
            return False

        # For functions and methods, check if they're just empty definitions
        if node.type == "function_definition":
            # Look for actual body content beyond just 'end'
            lines = text.strip().split("\n")
            if len(lines) <= 2:  # Just function declaration and end
                return False

            # Check if body only contains 'end'
            body_lines = [
                line.strip() for line in lines[1:-1]
            ]  # Skip first and last line
            if all(not line or line.startswith("%") for line in body_lines):
                return False

        return True

    def extract_constants(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from MATLAB code.

        Identifies UPPER_CASE variable assignments as constants (MATLAB convention)
        from:
        1. Top-level script assignments
        2. Function body assignments
        3. Properties (Constant) blocks

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name' and 'value' keys, or None
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return None

        if concept != UniversalConcept.DEFINITION:
            return None

        source = content.decode("utf-8")

        # Get the definition node
        def_node = captures.get("definition")
        if not def_node:
            return None

        # Handle top-level assignments (captured as definition nodes)
        if def_node.type == "assignment":
            assignment_name = captures.get("assignment_name")
            if not assignment_name:
                return None

            var_name = self.get_node_text(assignment_name, source).strip()

            # Check UPPER_CASE convention
            if not re.match(r'^[A-Z][A-Z0-9_]*$', var_name):
                return None

            # Extract value from assignment
            assignment_text = self.get_node_text(def_node, source).strip()
            value_match = re.search(rf'^{re.escape(var_name)}\s*=\s*(.+)', assignment_text)
            if value_match:
                value = value_match.group(1).strip().rstrip(";")
                # Truncate long values
                if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                    value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."
                return [{"name": var_name, "value": value}]

            return None

        # Check if this is a properties block with (Constant) attribute
        if def_node.type == "properties":
            # Get the full properties block text
            props_text = self.get_node_text(def_node, source).strip()

            # Check if it has (Constant) attribute
            if not re.search(r'properties\s*\([^)]*\bConstant\b', props_text):
                return None

            # Extract constant properties from this block
            constants = []
            for property_def in self.find_nodes_by_type(def_node, "property"):
                # Extract property name
                name_node = self.find_child_by_type(property_def, "identifier")
                if not name_node:
                    continue

                prop_name = self.get_node_text(name_node, source).strip()
                if not prop_name:
                    continue

                # Extract value if present
                prop_text = self.get_node_text(property_def, source).strip()
                value_match = re.search(rf'{re.escape(prop_name)}\s*=\s*(.+?)(?:\s*;|\s*$)', prop_text)

                if value_match:
                    value = value_match.group(1).strip()
                    # Truncate long values
                    if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                        value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."
                    constants.append({"name": prop_name, "value": value})
                else:
                    # No value specified
                    constants.append({"name": prop_name, "value": ""})

            return constants if constants else None

        # MATLAB uses UPPER_CASE convention for constants
        # Look for variable assignments inside function bodies
        if def_node.type == "function_definition":
            if not TREE_SITTER_AVAILABLE:
                return None

            constants = []
            # Find all assignment nodes in the function body
            for assignment_node in self.find_nodes_by_type(def_node, "assignment"):
                # Get the left-hand side (identifier)
                identifier_node = self.find_child_by_type(assignment_node, "identifier")
                if not identifier_node:
                    continue

                var_name = self.get_node_text(identifier_node, source).strip()

                # Check if it matches UPPER_CASE convention
                if not re.match(r'^[A-Z][A-Z0-9_]*$', var_name):
                    continue

                # Extract the full assignment text to get the value
                assignment_text = self.get_node_text(assignment_node, source).strip()

                # Pattern: CONSTANT_NAME = value
                value_match = re.search(rf'^{re.escape(var_name)}\s*=\s*(.+)', assignment_text)
                if value_match:
                    value = value_match.group(1).strip().rstrip(";")
                    # Truncate long values
                    if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                        value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."
                    constants.append({"name": var_name, "value": value})

            return constants if constants else None

        return None

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve MATLAB import to file path.

        MATLAB imports map to +package directories:
        - import pkg.Class -> +pkg/Class.m
        - import pkg.subpkg.* -> +pkg/+subpkg/ (directory)

        Args:
            import_text: The raw import statement text
            base_dir: Project root directory
            source_file: File containing the import

        Returns:
            Resolved file path or None if external/unresolvable
        """
        # Match import statement
        match = re.search(r"import\s+([\w.]+)", import_text)
        if not match:
            return None

        import_path = match.group(1)
        parts = import_path.split(".")

        # Handle wildcard imports (pkg.*)
        if parts[-1] == "*":
            parts = parts[:-1]
            if not parts:
                return None
            # Convert to +pkg/+subpkg directory path
            dir_parts = [f"+{p}" for p in parts]
            rel_path = "/".join(dir_parts)
            full_path = base_dir / rel_path
            if full_path.is_dir():
                return full_path
            return None

        # Regular import (pkg.Class or pkg.func)
        # Last part is the class/function, rest are packages
        if len(parts) >= 1:
            class_or_func = parts[-1]
            pkg_parts = parts[:-1]

            # Build path: +pkg/+subpkg/Class.m
            dir_parts = [f"+{p}" for p in pkg_parts]
            if dir_parts:
                rel_path = "/".join(dir_parts) + f"/{class_or_func}.m"
            else:
                rel_path = f"{class_or_func}.m"

            full_path = base_dir / rel_path
            if full_path.exists():
                return full_path

            # Try as @ class directory
            if dir_parts:
                class_dir = "/".join(dir_parts) + f"/@{class_or_func}"
            else:
                class_dir = f"@{class_or_func}"
            class_path = base_dir / class_dir
            if class_path.is_dir():
                # Look for class file
                class_file = class_path / f"{class_or_func}.m"
                if class_file.exists():
                    return class_file

        return None
