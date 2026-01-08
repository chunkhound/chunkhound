"""Groovy language mapping for the unified parser architecture.

This module provides Groovy-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Groovy's dynamic features including
classes, traits, methods, closures, dynamic typing, GStrings, and Groovy-specific
syntax while building on the Java foundation.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalConcept

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode


class GroovyMapping(BaseMapping):
    """Groovy-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize Groovy mapping."""
        super().__init__(Language.GROOVY)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Groovy method definitions.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_declaration
            name: (identifier) @method_name
        ) @method_def

        (closure
        ) @closure_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Groovy class definitions.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_declaration
            name: (identifier) @class_name
        ) @class_def

        (interface_declaration
            name: (identifier) @interface_name
        ) @interface_def

        (enum_declaration
            name: (identifier) @enum_name
        ) @enum_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Groovy comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (line_comment) @comment
        (block_comment) @comment
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for Groovy method definitions.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_declaration
            name: (identifier) @method_name
        ) @method_def

        (constructor_declaration
            name: (identifier) @constructor_name
        ) @constructor_def

        (closure
        ) @closure_def
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for Groovydoc comments.

        Returns:
            Tree-sitter query string for finding Groovydoc comments
        """
        return """
        (block_comment) @groovydoc
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Groovy method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "method")

        try:
            # Handle closures specially
            if node.type == "closure":
                # Try to find if this closure is assigned to a variable
                parent = getattr(node, "parent", None)
                if parent and parent.type == "assignment":
                    var_node = self.find_child_by_type(parent, "identifier")
                    if var_node:
                        return self.get_node_text(var_node, source).strip()
                return self.get_fallback_name(node, "closure")

            # Find method name identifier
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source).strip()

            # Fallback: look through children for the identifier
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Groovy method name: {e}")

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a Groovy class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "class")

        try:
            # Find class name identifier
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source).strip()

            # Fallback: look through children for the identifier
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Groovy class name: {e}")

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Groovy method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        # Delegate to extract_function_name as Groovy methods are functions
        return self.extract_function_name(node, source)

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a Groovy method node.

        Args:
            node: Tree-sitter method/closure definition node
            source: Source code string

        Returns:
            List of parameter type strings (may be 'def' for dynamic typing)
        """
        if node is None:
            return []

        parameters: list[str] = []
        try:
            # Handle closures
            if node.type == "closure":
                # Look for closure parameters
                param_node = self.find_child_by_type(node, "closure_parameters")
                if param_node:
                    # Extract parameters from closure parameter list
                    for child in self.walk_tree(param_node):
                        if child and child.type == "identifier":
                            # In Groovy closures, parameters are often untyped
                            parameters.append("def")  # Default Groovy dynamic type
                return parameters

            # Find formal_parameters node for regular methods
            params_node = self.find_child_by_type(node, "formal_parameters")
            if not params_node:
                return parameters

            # Extract each formal_parameter
            for child in self.find_children_by_type(params_node, "formal_parameter"):
                # Look for the type of the parameter
                type_node = None
                for param_child in self.walk_tree(child):
                    if param_child and param_child.type in [
                        "type_identifier",
                        "integral_type",
                        "floating_point_type",
                        "boolean_type",
                        "generic_type",
                        "array_type",
                        "identifier",  # For Groovy dynamic typing
                    ]:
                        type_node = param_child
                        break

                if type_node:
                    param_type = self.get_node_text(type_node, source).strip()
                    # If no explicit type, Groovy defaults to dynamic typing
                    if not param_type or param_type == "def":
                        param_type = "def"
                    parameters.append(param_type)
                else:
                    # Fallback: use the first part of the parameter
                    param_text = self.get_node_text(child, source).strip()
                    if param_text:
                        # Try to extract type from "Type varName" or just "varName" format
                        parts = param_text.split()
                        if len(parts) >= 2:
                            parameters.append(parts[0])
                        else:
                            parameters.append("def")  # Dynamic typing

        except Exception as e:
            logger.error(f"Failed to extract Groovy method parameters: {e}")

        return parameters

    def extract_package_name(self, root_node: "TSNode | None", source: str) -> str:
        """Extract package name from Groovy file.

        Args:
            root_node: Root node of the Groovy AST
            source: Source code string

        Returns:
            Package name as string, or empty string if no package declaration found
        """
        if root_node is None:
            return ""

        try:
            # Look for package_declaration
            package_nodes = self.find_nodes_by_type(root_node, "package_declaration")
            if not package_nodes:
                return ""

            package_node = package_nodes[0]
            package_text = self.get_node_text(package_node, source)

            # Extract package name from "package com.example.demo"
            package_text = package_text.strip()
            if package_text.startswith("package ") and package_text.endswith(";"):
                return package_text[8:-1].strip()
            elif package_text.startswith("package "):
                return package_text[8:].strip()

        except Exception as e:
            logger.error(f"Failed to extract Groovy package name: {e}")

        return ""

    def extract_annotations(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract Groovy annotations from a node.

        Args:
            node: Tree-sitter node to extract annotations from
            source: Source code string

        Returns:
            List of annotation strings
        """
        if node is None:
            return []

        annotations = []
        try:
            # Look for modifiers node which contains annotations
            modifiers_node = self.find_child_by_type(node, "modifiers")
            if modifiers_node:
                annotation_nodes = self.find_children_by_type(
                    modifiers_node, "annotation"
                )
                annotation_nodes.extend(
                    self.find_children_by_type(modifiers_node, "marker_annotation")
                )

                for ann_node in annotation_nodes:
                    annotation_text = self.get_node_text(ann_node, source).strip()
                    if annotation_text:
                        annotations.append(annotation_text)

            # Also check direct children for annotations (fallback)
            annotation_nodes = self.find_children_by_type(node, "annotation")
            annotation_nodes.extend(
                self.find_children_by_type(node, "marker_annotation")
            )

            for ann_node in annotation_nodes:
                annotation_text = self.get_node_text(ann_node, source).strip()
                if annotation_text and annotation_text not in annotations:
                    annotations.append(annotation_text)

        except Exception as e:
            logger.error(f"Failed to extract Groovy annotations: {e}")

        return annotations

    def is_closure(self, node: "TSNode | None") -> bool:
        """Check if a node is a Groovy closure.

        Args:
            node: Tree-sitter node

        Returns:
            True if node is a closure, False otherwise
        """
        return node is not None and node.type == "closure"

    def is_trait(self, node: "TSNode | None") -> bool:
        """Check if a node is a Groovy trait.

        Args:
            node: Tree-sitter node

        Returns:
            True if node is a trait, False otherwise
        """
        return node is not None and node.type == "trait_declaration"

    def extract_gstring_content(self, node: "TSNode | None", source: str) -> str:
        """Extract content from a Groovy GString (interpolated string).

        Args:
            node: Tree-sitter GString node
            source: Source code string

        Returns:
            GString content with interpolation markers
        """
        if node is None:
            return ""

        try:
            if node.type in ["gstring", "string_literal"]:
                return self.get_node_text(node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Groovy GString content: {e}")

        return ""

    def extract_closure_parameters(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract parameter names from a Groovy closure.

        Args:
            node: Tree-sitter closure node
            source: Source code string

        Returns:
            List of parameter names
        """
        if node is None or node.type != "closure":
            return []

        parameters = []
        try:
            # Look for closure_parameters
            params_node = self.find_child_by_type(node, "closure_parameters")
            if params_node:
                for child in self.walk_tree(params_node):
                    if child and child.type == "identifier":
                        param_name = self.get_node_text(child, source).strip()
                        if param_name:
                            parameters.append(param_name)

        except Exception as e:
            logger.error(f"Failed to extract Groovy closure parameters: {e}")

        return parameters

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a Groovy node should be included as a chunk.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if node is None:
            return False

        try:
            # Skip empty nodes
            node_text = self.get_node_text(node, source).strip()
            if not node_text:
                return False

            # For comments, check if it's meaningful
            if node.type in ["line_comment", "block_comment"]:
                # Skip comments that are just separators or empty
                cleaned_text = self.clean_comment_text(node_text)
                if len(cleaned_text) < 5:  # Very short comments probably not useful
                    return False
                if cleaned_text.strip("=/-*+_ \t\n") == "":  # Only separator characters
                    return False

            # For Groovydoc, only include if it starts with /**
            if node.type == "block_comment":
                if self.get_node_text(node, source).strip().startswith("/**"):
                    # This is a Groovydoc comment
                    cleaned_text = self.clean_comment_text(node_text)
                    return len(cleaned_text) > 10  # Meaningful Groovydoc

            # Include closures if they have meaningful content
            if node.type == "closure":
                return len(node_text) > 10  # Skip very small closures

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate Groovy node inclusion: {e}")
            return False

    def clean_comment_text(self, text: str) -> str:
        """Clean Groovy comment text by removing comment markers and Groovydoc formatting.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Handle Groovydoc comments (same as Javadoc)
        if cleaned.startswith("/**") and cleaned.endswith("*/"):
            cleaned = cleaned[3:-2].strip()
            # Remove leading * from each line
            lines = cleaned.split("\n")
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("* "):
                    line = line[2:]
                elif line.startswith("*"):
                    line = line[1:]
                cleaned_lines.append(line)
            cleaned = "\n".join(cleaned_lines).strip()
        # Handle regular block comments
        elif cleaned.startswith("/*") and cleaned.endswith("*/"):
            cleaned = cleaned[2:-2].strip()
        # Handle line comments
        elif cleaned.startswith("//"):
            cleaned = cleaned[2:].strip()

        return cleaned

    def get_qualified_name(
        self,
        node: "TSNode | None",
        source: str,
        package_name: str = "",
        parent_name: str = "",
    ) -> str:
        """Get fully qualified name for a Groovy symbol.

        Args:
            node: Tree-sitter node
            source: Source code string
            package_name: Package name
            parent_name: Parent class/trait name

        Returns:
            Fully qualified symbol name
        """
        if node is None:
            return ""

        try:
            name = ""
            if node.type == "class_declaration":
                name = self.extract_class_name(node, source)
            elif node.type == "interface_declaration":
                name = self.extract_class_name(node, source)
            elif node.type == "trait_declaration":
                name = self.extract_class_name(node, source)
            elif node.type == "enum_declaration":
                name = self.extract_class_name(node, source)
            elif node.type in ["method_declaration", "constructor_declaration"]:
                name = self.extract_method_name(node, source)
            elif node.type == "closure":
                name = self.extract_function_name(node, source)
            else:
                # Try to get name from identifier
                name_node = self.find_child_by_type(node, "identifier")
                if name_node:
                    name = self.get_node_text(name_node, source).strip()

            if not name:
                return self.get_fallback_name(node, "symbol")

            # Build qualified name
            qualified_name = name
            if parent_name:
                qualified_name = f"{parent_name}.{name}"
            if package_name:
                qualified_name = f"{package_name}.{qualified_name}"

            return qualified_name

        except Exception as e:
            logger.error(f"Failed to get Groovy qualified name: {e}")
            return self.get_fallback_name(node, "symbol")

    def extract_field_declarations(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract field declaration names from a Groovy class.

        Args:
            node: Tree-sitter class node
            source: Source code string

        Returns:
            List of field names
        """
        if node is None:
            return []

        fields = []
        try:
            # Look for field_declaration nodes
            field_nodes = self.find_nodes_by_type(node, "field_declaration")
            for field_node in field_nodes:
                # Extract variable names from the field declaration
                var_nodes = self.find_children_by_type(
                    field_node, "variable_declarator"
                )
                for var_node in var_nodes:
                    name_node = self.find_child_by_type(var_node, "identifier")
                    if name_node:
                        field_name = self.get_node_text(name_node, source).strip()
                        if field_name:
                            fields.append(field_name)

        except Exception as e:
            logger.error(f"Failed to extract Groovy field declarations: {e}")

        return fields

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Groovy code.

        Identifies two types of constants:
        1. Static final fields: class-level constants (requires both 'static' and 'final')
        2. Local final variables: method-scoped constants (requires 'final' modifier)

        Supports multiple variable declarators in a single declaration.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            List of constant dictionaries with 'name', 'value', and optional 'type' keys,
            or None if no constants found

        Examples:
            - Static final field: "static final int MAX = 100"
            - Local final variable: "final String NAME = 'test'"
            - Multiple declarators: "final int A = 1, B = 2, C = 3"
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return None

        if concept != UniversalConcept.DEFINITION:
            return None

        # Get the definition node
        def_node = captures.get("definition")
        if not def_node or def_node.type not in (
            "field_declaration",
            "local_variable_declaration",
        ):
            return None

        source = content.decode("utf-8")

        # Check for required modifiers based on declaration type
        modifiers_node = self.find_child_by_type(def_node, "modifiers")
        if not modifiers_node:
            return None

        modifiers_text = self.get_node_text(modifiers_node, source)

        # Field declarations require both static and final
        # Local variable declarations only require final
        if def_node.type == "field_declaration":
            if not ("static" in modifiers_text and "final" in modifiers_text):
                return None
        elif def_node.type == "local_variable_declaration":
            if "final" not in modifiers_text:
                return None

        # Extract type from the field declaration
        type_node = None
        for child in self.walk_tree(def_node):
            if child and child.type in [
                "type_identifier",
                "integral_type",
                "floating_point_type",
                "boolean_type",
                "generic_type",
                "array_type",
            ]:
                type_node = child
                break

        const_type = (
            self.get_node_text(type_node, source).strip() if type_node else None
        )

        # Extract variable declarators
        constants = []
        var_nodes = self.find_children_by_type(def_node, "variable_declarator")
        for var_node in var_nodes:
            name_node = self.find_child_by_type(var_node, "identifier")
            if not name_node:
                continue

            name = self.get_node_text(name_node, source).strip()

            # Try to extract value
            value = ""
            for child in var_node.children:
                if child.type == "=":
                    # Next child is the value
                    idx = var_node.children.index(child)
                    if idx + 1 < len(var_node.children):
                        value_node = var_node.children[idx + 1]
                        value = self.get_node_text(value_node, source).strip()
                        break

            # Truncate long values
            if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."

            # Build result dict
            result: dict[str, str] = {"name": name, "value": value}
            if const_type is not None:
                result["type"] = const_type

            constants.append(result)

        return constants if constants else None

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve Groovy import to file path.

        Args:
            import_text: Import statement text (e.g., "import com.example.Foo;")
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            Path to the imported file, or None if not found
        """
        # Extract class path: import com.example.Foo; or import static com.example.Foo.bar;
        match = re.search(r"import\s+(?:static\s+)?([\w.]+);?", import_text)
        if not match:
            return None

        class_path = match.group(1)
        if not class_path:
            return None

        # Convert to file path (last part is class name)
        rel_path = class_path.replace(".", "/") + ".groovy"

        # Try common source directories
        for prefix in ["", "src/main/groovy/", "src/", "app/src/main/groovy/"]:
            full_path = base_dir / prefix / rel_path
            if full_path.exists():
                return full_path

        return None

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Groovy."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (method_declaration
                name: (identifier) @name
            ) @definition

            (constructor_declaration
                name: (identifier) @name
            ) @definition

            (class_declaration
                name: (identifier) @name
            ) @definition

            (interface_declaration
                name: (identifier) @name
            ) @definition

            (enum_declaration
                name: (identifier) @name
            ) @definition

            (closure
            ) @definition

            (field_declaration) @definition

            (local_variable_declaration) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (line_comment) @definition
            (block_comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (import_declaration) @definition
            (package_declaration) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (program
                (package_declaration)? @package
                (import_declaration)* @imports
            ) @definition
            """

        else:
            # BLOCK concept not supported for Groovy
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Use @name capture if available
            if "name" in captures:
                return self.get_node_text(captures["name"], source).strip()

            # For field_declaration and local_variable_declaration, extract variable name
            def_node = captures.get("definition")
            if def_node and def_node.type in (
                "field_declaration",
                "local_variable_declaration",
            ):
                declarator = self.find_child_by_type(def_node, "variable_declarator")
                if declarator:
                    var_name_node = self.find_child_by_type(declarator, "identifier")
                    if var_name_node:
                        return self.get_node_text(var_name_node, source).strip()
                line = def_node.start_point[0] + 1
                prefix = (
                    "field" if def_node.type == "field_declaration" else "local_var"
                )
                return f"{prefix}_line_{line}"

            # For closures, extract assigned variable name
            if def_node and def_node.type == "closure":
                return self.extract_function_name(def_node, source)

            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                comment_text = self.get_node_text(node, source)
                if comment_text.strip().startswith("/**"):
                    return f"groovydoc_line_{line}"
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                import_text = self.get_node_text(node, source).strip()

                # Package declaration
                if node.type == "package_declaration":
                    if import_text.startswith("package ") and import_text.endswith(";"):
                        pkg_name = import_text[8:-1].strip()
                        return f"package_{pkg_name.replace('.', '_')}"
                    elif import_text.startswith("package "):
                        pkg_name = import_text[8:].strip()
                        return f"package_{pkg_name.replace('.', '_')}"
                    return "package_unknown"

                # Import declaration
                if node.type == "import_declaration":
                    match = re.search(r"import\s+(?:static\s+)?([\w.]+);?", import_text)
                    if match:
                        import_path = match.group(1)
                        parts = import_path.split(".")
                        return f"import_{parts[-1]}"
                    return "import_unknown"

            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""
        source = content.decode("utf-8")

        if "definition" in captures:
            node = captures["definition"]
            return self.get_node_text(node, source)
        elif captures:
            node = list(captures.values())[0]
            return self.get_node_text(node, source)

        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, str]:
        """Extract Groovy-specific metadata."""
        source = content.decode("utf-8")
        metadata: dict[str, str] = {}

        def_node = captures.get("definition")
        if def_node:
            metadata["node_type"] = def_node.type

            # Extract annotations for DEFINITION concept
            if concept == UniversalConcept.DEFINITION:
                annotations = self.extract_annotations(def_node, source)
                if annotations:
                    metadata["annotations"] = ", ".join(annotations)

                # Extract parameters for methods/closures
                if def_node.type in ("method_declaration", "constructor_declaration"):
                    params = self.extract_parameters(def_node, source)
                    if params:
                        metadata["parameters"] = ", ".join(params)
                elif def_node.type == "closure":
                    params = self.extract_closure_parameters(def_node, source)
                    if params:
                        metadata["parameters"] = ", ".join(params)

        return metadata
