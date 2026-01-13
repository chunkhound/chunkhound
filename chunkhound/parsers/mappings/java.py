"""Java language mapping for the unified parser architecture.

This module provides Java-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Java's object-oriented features
including classes, interfaces, methods, constructors, annotations, and Javadoc.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalConcept

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

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
        # Extract class path: import com.example.Foo;
        # or import static com.example.Foo.bar;
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

    def extract_constants(
        self,
        concept: UniversalConcept,
        captures: dict[str, Node],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Java code.

        Detects static final field declarations.

        Args:
            concept: The universal concept being processed
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dictionaries with "name", "value", and "type" keys, or None
        """
        source = content.decode("utf-8")

        # Only process DEFINITION concept
        if concept != UniversalConcept.DEFINITION:
            return None

        def_node = captures.get("definition")
        if not def_node:
            return None

        # Handle enum declarations
        if def_node.type == "enum_declaration":
            enum_body = self.find_child_by_type(def_node, "enum_body")
            if not enum_body:
                return None

            enum_constants = self.find_children_by_type(enum_body, "enum_constant")
            if not enum_constants:
                return None

            results = []
            for enum_const in enum_constants:
                name_node = self.find_child_by_type(enum_const, "identifier")
                if name_node:
                    const_name = self.get_node_text(name_node, source).strip()
                    results.append({"name": const_name, "value": const_name})

            return results if results else None

        # Handle local variable declarations with final modifier
        if def_node.type == "local_variable_declaration":
            # Check for final modifier
            modifiers_node = self.find_child_by_type(def_node, "modifiers")
            if not modifiers_node:
                return None

            modifiers_text = self.get_node_text(modifiers_node, source)
            has_final = "final" in modifiers_text

            if not has_final:
                return None

        # Handle static final field declarations
        elif def_node.type == "field_declaration":
            # Check for both static and final modifiers
            modifiers_node = self.find_child_by_type(def_node, "modifiers")
            if not modifiers_node:
                return None

            modifiers_text = self.get_node_text(modifiers_node, source)
            has_static = "static" in modifiers_text
            has_final = "final" in modifiers_text

            if not (has_static and has_final):
                return None

        else:
            # Not a node type we handle for constant extraction
            return None

        # Extract field name and value
        # Look for variable_declarator which contains identifier and optional value
        declarator = self.find_child_by_type(def_node, "variable_declarator")
        if not declarator:
            return None

        # Get field name
        name_node = self.find_child_by_type(declarator, "identifier")
        if not name_node:
            return None

        const_name = self.get_node_text(name_node, source).strip()

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

        # Extract value by finding the assignment after "="
        const_value: str | None = None
        found_equals = False
        for i in range(declarator.child_count):
            child_node = declarator.child(i)
            if child_node is None:
                continue

            if self.get_node_text(child_node, source).strip() == "=":
                found_equals = True
            elif found_equals and child_node.type != ";":
                # Extract the value, truncating to 50 chars if longer
                value_text = self.get_node_text(child_node, source).strip()
                if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                    const_value = value_text[:MAX_CONSTANT_VALUE_LENGTH]
                else:
                    const_value = value_text
                break

        # Build result dict
        result: dict[str, str] = {"name": const_name}

        if const_value is not None:
            result["value"] = const_value

        if const_type is not None:
            result["type"] = const_type

        return [result]

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Java."""

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

            (annotation_type_declaration
                name: (identifier) @name
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
            # BLOCK concept not supported for Java
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Use @name capture if available
            if "name" in captures:
                return self.get_node_text(captures["name"], source).strip()

            # For field_declaration and local_variable_declaration, extract variable name
            def_node = captures.get("definition")
            if def_node and def_node.type in ["field_declaration", "local_variable_declaration"]:
                declarator = self.find_child_by_type(def_node, "variable_declarator")
                if declarator:
                    var_name_node = self.find_child_by_type(declarator, "identifier")
                    if var_name_node:
                        return self.get_node_text(var_name_node, source).strip()
                line = def_node.start_point[0] + 1
                node_type = "local" if def_node.type == "local_variable_declaration" else "field"
                return f"{node_type}_line_{line}"

            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                comment_text = self.get_node_text(node, source)
                if comment_text.strip().startswith("/**"):
                    return f"javadoc_line_{line}"
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                import_text = self.get_node_text(node, source).strip()

                # Package declaration
                if node.type == "package_declaration":
                    # Extract: package com.example.foo;
                    if import_text.startswith("package ") and import_text.endswith(";"):
                        pkg_name = import_text[8:-1].strip()
                        return f"package_{pkg_name.replace('.', '_')}"
                    return "package_unknown"

                # Import declaration
                if node.type == "import_declaration":
                    # Extract: import com.example.Foo; or import static ...
                    match = re.search(r"import\s+(?:static\s+)?([\w.]+);", import_text)
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
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
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
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract Java-specific metadata."""
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # Determine kind
                kind_map = {
                    "method_declaration": "method",
                    "constructor_declaration": "constructor",
                    "class_declaration": "class",
                    "interface_declaration": "interface",
                    "enum_declaration": "enum",
                    "annotation_type_declaration": "annotation",
                    "field_declaration": "field",
                    "local_variable_declaration": "local_variable",
                }
                metadata["kind"] = kind_map.get(def_node.type, "unknown")

                # Extract modifiers and visibility
                modifiers_node = self.find_child_by_type(def_node, "modifiers")
                if modifiers_node:
                    modifiers_text = self.get_node_text(modifiers_node, source)

                    # Visibility
                    if "public" in modifiers_text:
                        metadata["visibility"] = "public"
                    elif "private" in modifiers_text:
                        metadata["visibility"] = "private"
                    elif "protected" in modifiers_text:
                        metadata["visibility"] = "protected"
                    else:
                        metadata["visibility"] = "package"

                    # Other modifiers
                    modifiers = []
                    modifier_keywords = [
                        "static", "final", "abstract", "synchronized",
                        "native", "volatile", "transient",
                    ]
                    for mod in modifier_keywords:
                        if mod in modifiers_text:
                            modifiers.append(mod)
                    if modifiers:
                        metadata["modifiers"] = modifiers

                # For methods and constructors
                if def_node.type in ["method_declaration", "constructor_declaration"]:
                    params = self.extract_parameters(def_node, source)
                    if params:
                        metadata["parameters"] = params

                    if def_node.type == "method_declaration":
                        return_type = self.extract_return_type(def_node, source)
                        if return_type:
                            metadata["return_type"] = return_type

                # Extract annotations
                annotations = self.extract_annotations(def_node, source)
                if annotations:
                    metadata["annotations"] = annotations

                # Extract type parameters (generics)
                type_params = self.extract_type_parameters(def_node, source)
                if type_params:
                    metadata["type_parameters"] = type_params

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                import_text = self.get_node_text(node, source).strip()

                if node.type == "import_declaration":
                    metadata["import_type"] = "import"
                    if "static" in import_text:
                        metadata["is_static_import"] = True

                    # Extract full import path
                    match = re.search(r"import\s+(?:static\s+)?([\w.]+);", import_text)
                    if match:
                        metadata["import_path"] = match.group(1)

                elif node.type == "package_declaration":
                    metadata["import_type"] = "package"
                    if import_text.startswith("package ") and import_text.endswith(";"):
                        metadata["package_name"] = import_text[8:-1].strip()

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                comment_text = self.get_node_text(node, source)

                if node.type == "line_comment":
                    metadata["comment_type"] = "line"
                elif node.type == "block_comment":
                    if comment_text.strip().startswith("/**"):
                        metadata["comment_type"] = "javadoc"
                        metadata["is_javadoc"] = True
                    else:
                        metadata["comment_type"] = "block"

        return metadata
