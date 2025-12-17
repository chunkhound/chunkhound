"""Java language mapping for the unified parser architecture.

This module provides Java-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Java's object-oriented features
including classes, interfaces, methods, constructors, annotations, and Javadoc.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from chunkhound.core.types.common import Language

from .base import BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode


class JavaMapping(BaseMapping):
    """Java-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize Java mapping."""
        super().__init__(Language.JAVA)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Java method definitions.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_declaration
            name: (identifier) @method_name
        ) @method_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Java class definitions.

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
        """Get tree-sitter query pattern for Java comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (line_comment) @comment
        (block_comment) @comment
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for Java method definitions.

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
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for Javadoc comments.

        Returns:
            Tree-sitter query string for finding Javadoc comments
        """
        return """
        (block_comment) @javadoc
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Java method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "method")

        try:
            # Find method name identifier
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source).strip()

            # Fallback: look for field_name in method_declaration
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    # Get the first identifier which should be the method name
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Java method name: {e}")

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a Java class definition node.

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
            logger.error(f"Failed to extract Java class name: {e}")

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Java method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        # Delegate to extract_function_name as Java methods are functions
        return self.extract_function_name(node, source)

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a Java method node.

        Args:
            node: Tree-sitter method/constructor definition node
            source: Source code string

        Returns:
            List of parameter type strings
        """
        if node is None:
            return []

        parameters: list[str] = []
        try:
            # Find formal_parameters node
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
                    ]:
                        type_node = param_child
                        break

                if type_node:
                    param_type = self.get_node_text(type_node, source).strip()
                    parameters.append(param_type)
                else:
                    # Fallback: use the first part of the parameter
                    param_text = self.get_node_text(child, source).strip()
                    if param_text:
                        # Try to extract type from "Type varName" format
                        parts = param_text.split()
                        if len(parts) >= 2:
                            parameters.append(parts[0])
                        else:
                            parameters.append(param_text)

        except Exception as e:
            logger.error(f"Failed to extract Java method parameters: {e}")

        return parameters

    def extract_package_name(self, root_node: "TSNode | None", source: str) -> str:
        """Extract package name from Java file.

        Args:
            root_node: Root node of the Java AST
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

            # Extract package name from "package com.example.demo;"
            package_text = package_text.strip()
            if package_text.startswith("package ") and package_text.endswith(";"):
                return package_text[8:-1].strip()

        except Exception as e:
            logger.error(f"Failed to extract Java package name: {e}")

        return ""

    def extract_annotations(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract Java annotations from a node.

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
            logger.error(f"Failed to extract Java annotations: {e}")

        return annotations

    def extract_type_parameters(self, node: "TSNode | None", source: str) -> str:
        """Extract generic type parameters from a Java node.

        Args:
            node: Tree-sitter node to extract type parameters from
            source: Source code string

        Returns:
            Type parameters string (e.g., "<T, U extends Comparable<U>>")
        """
        if node is None:
            return ""

        try:
            type_params_node = self.find_child_by_type(node, "type_parameters")
            if type_params_node:
                return self.get_node_text(type_params_node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Java type parameters: {e}")

        return ""

    def extract_return_type(self, node: "TSNode | None", source: str) -> str | None:
        """Extract return type from a Java method node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Return type string or None if not found/constructor
        """
        if node is None:
            return None

        try:
            # Constructor declarations don't have return types
            if node.type == "constructor_declaration":
                return None

            # Look for return type in method_declaration
            for child in self.walk_tree(node):
                if child and child.type in [
                    "type_identifier",
                    "integral_type",
                    "floating_point_type",
                    "boolean_type",
                    "void_type",
                    "generic_type",
                    "array_type",
                ]:
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Java return type: {e}")

        return None

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a Java node should be included as a chunk.

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

            # For comments, check if it's a meaningful comment
            if node.type in ["line_comment", "block_comment"]:
                # Skip comments that are just separators or empty
                cleaned_text = self.clean_comment_text(node_text)
                if len(cleaned_text) < 5:  # Very short comments probably not useful
                    return False
                if cleaned_text.strip("=/-*+_ \t\n") == "":  # Only separator characters
                    return False

            # For Javadoc, only include if it starts with /**
            if node.type == "block_comment":
                if self.get_node_text(node, source).strip().startswith("/**"):
                    # This is a Javadoc comment
                    cleaned_text = self.clean_comment_text(node_text)
                    return len(cleaned_text) > 10  # Meaningful Javadoc

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate Java node inclusion: {e}")
            return False

    def clean_comment_text(self, text: str) -> str:
        """Clean Java comment text by removing comment markers and Javadoc formatting.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Handle Javadoc comments
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
        """Get fully qualified name for a Java symbol.

        Args:
            node: Tree-sitter node
            source: Source code string
            package_name: Package name
            parent_name: Parent class/interface name

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
            elif node.type == "enum_declaration":
                name = self.extract_class_name(node, source)
            elif node.type in ["method_declaration", "constructor_declaration"]:
                name = self.extract_method_name(node, source)
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
            logger.error(f"Failed to get Java qualified name: {e}")
            return self.get_fallback_name(node, "symbol")

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve Java import to file path.

        Args:
            import_text: Import statement text (e.g., "import com.example.Foo;")
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            Path to the imported file, or None if not found
        """
        # Extract class path: import com.example.Foo; or import static com.example.Foo.bar;
        match = re.search(r"import\s+(?:static\s+)?([\w.]+);", import_text)
        if not match:
            return None

        class_path = match.group(1)
        if not class_path:
            return None

        # Convert to file path (last part is class name)
        rel_path = class_path.replace(".", "/") + ".java"

        # Try common source directories
        for prefix in ["", "src/main/java/", "src/", "app/src/main/java/"]:
            full_path = base_dir / prefix / rel_path
            if full_path.exists():
                return full_path

        return None
