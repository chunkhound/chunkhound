"""Kotlin language mapping for the unified parser architecture.

This module provides Kotlin-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Kotlin's modern features including
classes, data classes, sealed classes, functions, extension functions, interfaces,
properties, coroutines, and KDoc comments.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalConcept

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode


class KotlinMapping(BaseMapping):
    """Kotlin-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize Kotlin mapping."""
        super().__init__(Language.KOTLIN)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Kotlin function definitions.

        Returns:
            Tree-sitter query string for finding function definitions
        """
        return """
        (function_declaration
            name: (identifier) @function_name
        ) @function_def

        (property_declaration
            (variable_declaration
                (identifier) @property_name
            )
        ) @property_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Kotlin class definitions.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_declaration
            name: (identifier) @class_name
        ) @class_def

        (object_declaration
            name: (identifier) @object_name
        ) @object_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Kotlin comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (line_comment) @comment
        (block_comment) @comment
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for Kotlin method definitions.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (function_declaration
            name: (identifier) @function_name
        ) @function_def

        (property_declaration
            (variable_declaration
                (identifier) @property_name
            )
        ) @property_def

        (getter) @getter_def

        (setter) @setter_def
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for KDoc comments.

        Returns:
            Tree-sitter query string for finding KDoc comments
        """
        return """
        (block_comment) @kdoc
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a Kotlin function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "function")

        try:
            # Find function name identifier
            name_node = self.find_child_by_type(node, "identifier")
            if name_node:
                return self.get_node_text(name_node, source).strip()

            # Fallback: look through children for identifier
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Kotlin function name: {e}")

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a Kotlin class definition node.

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

            # Fallback: look through children
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Kotlin class name: {e}")

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a Kotlin method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        # In Kotlin, methods are functions, so delegate to function name extraction
        return self.extract_function_name(node, source)

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a Kotlin function node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            List of parameter type strings
        """
        if node is None:
            return []

        parameters: list[str] = []
        try:
            # Find function_value_parameters node
            params_node = self.find_child_by_type(node, "function_value_parameters")
            if not params_node:
                return parameters

            # Extract each parameter
            param_nodes = self.find_children_by_type(params_node, "parameter")
            for param_node in param_nodes:
                # Look for type annotation
                type_node = self.find_child_by_type(param_node, "user_type")
                if not type_node:
                    type_node = self.find_child_by_type(param_node, "type_reference")

                if type_node:
                    param_type = self.get_node_text(type_node, source).strip()
                    parameters.append(param_type)
                else:
                    # Try to extract from the parameter text
                    param_text = self.get_node_text(param_node, source).strip()
                    if ":" in param_text:
                        # Format: "name: Type" - extract type part
                        parts = param_text.split(":", 1)
                        if len(parts) == 2:
                            param_type = parts[1].strip()
                            # Remove default value if present
                            if "=" in param_type:
                                param_type = param_type.split("=")[0].strip()
                            parameters.append(param_type)

        except Exception as e:
            logger.error(f"Failed to extract Kotlin function parameters: {e}")

        return parameters

    def extract_package_name(self, root_node: "TSNode | None", source: str) -> str:
        """Extract package name from Kotlin file.

        Args:
            root_node: Root node of the Kotlin AST
            source: Source code string

        Returns:
            Package name as string, or empty string if no package declaration found
        """
        if root_node is None:
            return ""

        try:
            # Look for package_header
            package_nodes = self.find_nodes_by_type(root_node, "package_header")
            if not package_nodes:
                return ""

            package_node = package_nodes[0]
            package_text = self.get_node_text(package_node, source)

            # Extract package name from "package com.example.demo"
            package_text = package_text.strip()
            if package_text.startswith("package "):
                return package_text[8:].strip()

        except Exception as e:
            logger.error(f"Failed to extract Kotlin package name: {e}")

        return ""

    def extract_annotations(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract Kotlin annotations from a node.

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
            # Look for modifiers which contain annotations
            modifiers_node = self.find_child_by_type(node, "modifiers")
            if modifiers_node:
                annotation_nodes = self.find_children_by_type(
                    modifiers_node, "annotation"
                )
                for ann_node in annotation_nodes:
                    annotation_text = self.get_node_text(ann_node, source).strip()
                    if annotation_text:
                        annotations.append(annotation_text)

            # Also check direct children for annotations (fallback)
            annotation_nodes = self.find_children_by_type(node, "annotation")
            for ann_node in annotation_nodes:
                annotation_text = self.get_node_text(ann_node, source).strip()
                if annotation_text and annotation_text not in annotations:
                    annotations.append(annotation_text)

        except Exception as e:
            logger.error(f"Failed to extract Kotlin annotations: {e}")

        return annotations

    def extract_type_parameters(self, node: "TSNode | None", source: str) -> str:
        """Extract generic type parameters from a Kotlin node.

        Args:
            node: Tree-sitter node to extract type parameters from
            source: Source code string

        Returns:
            Type parameters string (e.g., "<T : Comparable<T>>")
        """
        if node is None:
            return ""

        try:
            type_params_node = self.find_child_by_type(node, "type_parameters")
            if type_params_node:
                return self.get_node_text(type_params_node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Kotlin type parameters: {e}")

        return ""

    def extract_return_type(self, node: "TSNode | None", source: str) -> str | None:
        """Extract return type from a Kotlin function node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Return type string or None if not found
        """
        if node is None:
            return None

        try:
            # Look for type_reference after the colon
            type_ref_node = self.find_child_by_type(node, "type_reference")
            if type_ref_node:
                return self.get_node_text(type_ref_node, source).strip()

            # Look for user_type
            user_type_node = self.find_child_by_type(node, "user_type")
            if user_type_node:
                return self.get_node_text(user_type_node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Kotlin return type: {e}")

        return None

    def is_suspend_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a Kotlin function is a suspend function (coroutine).

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if function is marked with suspend, False otherwise
        """
        if node is None:
            return False

        try:
            # Look for modifiers containing suspend
            modifiers_node = self.find_child_by_type(node, "modifiers")
            if modifiers_node:
                modifiers_text = self.get_node_text(modifiers_node, source)
                return "suspend" in modifiers_text

        except Exception as e:
            logger.error(f"Failed to check Kotlin suspend function: {e}")

        return False

    def is_extension_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a Kotlin function is an extension function.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if function is an extension function, False otherwise
        """
        if node is None:
            return False

        try:
            # Extension functions have a receiver type before the function name
            # Look for receiver_type in the function signature
            receiver_node = self.find_child_by_type(node, "receiver_type")
            return receiver_node is not None

        except Exception as e:
            logger.error(f"Failed to check Kotlin extension function: {e}")

        return False

    def extract_class_modifiers(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract Kotlin class modifiers (data, sealed, abstract, etc.).

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            List of modifier strings
        """
        if node is None:
            return []

        modifiers = []
        try:
            modifiers_node = self.find_child_by_type(node, "modifiers")
            if modifiers_node:
                modifiers_text = self.get_node_text(modifiers_node, source)
                # Common Kotlin class modifiers
                kotlin_modifiers = [
                    "data",
                    "sealed",
                    "abstract",
                    "final",
                    "open",
                    "inner",
                    "enum",
                    "annotation",
                    "companion",
                ]
                for modifier in kotlin_modifiers:
                    if modifier in modifiers_text:
                        modifiers.append(modifier)

        except Exception as e:
            logger.error(f"Failed to extract Kotlin class modifiers: {e}")

        return modifiers

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a Kotlin node should be included as a chunk.

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

            # For KDoc, only include if it starts with /**
            if node.type == "block_comment":
                if self.get_node_text(node, source).strip().startswith("/**"):
                    # This is a KDoc comment
                    cleaned_text = self.clean_comment_text(node_text)
                    return len(cleaned_text) > 10  # Meaningful KDoc

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate Kotlin node inclusion: {e}")
            return False

    def clean_comment_text(self, text: str) -> str:
        """Clean Kotlin comment text by removing comment markers and KDoc formatting.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Handle KDoc comments
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
        # Handle regular multiline comments
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
        """Get fully qualified name for a Kotlin symbol.

        Args:
            node: Tree-sitter node
            source: Source code string
            package_name: Package name
            parent_name: Parent class/object name

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
            elif node.type == "object_declaration":
                name = self.extract_class_name(node, source)
            elif node.type == "function_declaration":
                name = self.extract_function_name(node, source)
            elif node.type == "property_declaration":
                name = self.extract_function_name(
                    node, source
                )  # Properties handled like functions
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
            logger.error(f"Failed to get Kotlin qualified name: {e}")
            return self.get_fallback_name(node, "symbol")

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve Kotlin import to file path.

        Args:
            import_text: Import statement text (e.g., "import com.example.Foo")
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            Path to the imported file, or None if not found
        """
        # Extract class path: import com.example.Foo or import com.example.Foo.bar
        match = re.search(r"import\s+([\w.]+)", import_text)
        if not match:
            return None

        class_path = match.group(1)
        if not class_path:
            return None

        # Convert to file path (last part is class name)
        rel_path = class_path.replace(".", "/") + ".kt"

        # Try common source directories
        for prefix in ["", "src/main/kotlin/", "src/", "app/src/main/kotlin/"]:
            full_path = base_dir / prefix / rel_path
            if full_path.exists():
                return full_path

        return None

    def extract_constants(
        self,
        concept: UniversalConcept,
        captures: dict[str, "TSNode"],
        content: bytes,
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Kotlin code.

        Detects immutable val declarations at any scope (top-level, class-level,
        or local function scope). This includes both const val and regular val
        declarations. Note that var (mutable) declarations are excluded.

        In Kotlin's tree-sitter grammar, all val/var declarations are represented
        as property_declaration nodes, whether they are top-level, inside classes,
        or inside function bodies.

        Args:
            concept: The universal concept being processed
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dictionaries with "name" and "value" keys, or None
        """
        source = content.decode("utf-8")

        # Only process DEFINITION concept
        if concept != UniversalConcept.DEFINITION:
            return None

        def_node = captures.get("definition")
        if not def_node or def_node.type != "property_declaration":
            return None

        # Check if this is a val declaration (immutable)
        # Note: Both "const val" and "val" have a val child node, while "var" has var
        has_val = False
        for child in def_node.children:
            if child and child.type == "val":
                has_val = True
                break

        # Accept val declarations (immutable), reject var declarations (mutable)
        if not has_val:
            return None

        # Extract property name
        # Look for variable_declaration which contains identifier
        var_decl = self.find_child_by_type(def_node, "variable_declaration")
        if not var_decl:
            return None

        name_node = self.find_child_by_type(var_decl, "identifier")
        if not name_node:
            return None

        const_name = self.get_node_text(name_node, source).strip()

        # Extract type (optional) from type_reference
        const_type = None
        type_ref_node = self.find_child_by_type(def_node, "type_reference")
        if type_ref_node:
            const_type = self.get_node_text(type_ref_node, source).strip()

        # Extract value by finding the assignment after "="
        const_value = None
        found_equals = False
        for i in range(def_node.child_count):
            child = def_node.child(i)
            if child is None:
                continue

            if self.get_node_text(child, source).strip() == "=":
                found_equals = True
            elif found_equals:
                # Extract the value
                value_text = self.get_node_text(child, source).strip()
                const_value = value_text
                break

        # Build result dict
        result: dict[str, str] = {"name": const_name}

        if const_value is not None:
            result["value"] = const_value[:MAX_CONSTANT_VALUE_LENGTH]
        if const_type is not None:
            result["type"] = const_type

        return [result]

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Kotlin."""
        if concept == UniversalConcept.DEFINITION:
            # Note: interfaces are represented as class_declaration in Kotlin's tree-sitter
            return """
            (function_declaration
                name: (identifier) @name
            ) @definition

            (class_declaration
                name: (identifier) @name
            ) @definition

            (object_declaration
                name: (identifier) @name
            ) @definition

            (property_declaration) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (line_comment) @definition
            (block_comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (import) @definition
            (package_header) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (source_file
                (package_header)? @package
                (import)* @imports
            ) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Any], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Use @name capture if available
            if "name" in captures:
                return self.get_node_text(captures["name"], source).strip()

            # For property_declaration, extract variable name
            def_node = captures.get("definition")
            if def_node:
                if def_node.type == "property_declaration":
                    var_decl = self.find_child_by_type(def_node, "variable_declaration")
                    if var_decl:
                        name_node = self.find_child_by_type(var_decl, "identifier")
                        if name_node:
                            return self.get_node_text(name_node, source).strip()
                    line = def_node.start_point[0] + 1
                    return f"property_line_{line}"
                elif def_node.type == "function_declaration":
                    name_node = self.find_child_by_type(def_node, "identifier")
                    if name_node:
                        return self.get_node_text(name_node, source).strip()
                elif def_node.type in ("class_declaration", "object_declaration"):
                    name_node = self.find_child_by_type(def_node, "identifier")
                    if name_node:
                        return self.get_node_text(name_node, source).strip()

            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                comment_text = self.get_node_text(node, source)
                if comment_text.strip().startswith("/**"):
                    return f"kdoc_line_{line}"
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                import_text = self.get_node_text(node, source).strip()

                # Package header
                if node.type == "package_header":
                    if import_text.startswith("package "):
                        pkg_name = import_text[8:].strip()
                        return f"package_{pkg_name.replace('.', '_')}"
                    return "package_unknown"

                # Import statement
                if node.type == "import":
                    match = re.search(r"import\s+([\w.]+)", import_text)
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
        self, concept: UniversalConcept, captures: dict[str, Any], content: bytes
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
        self, concept: UniversalConcept, captures: dict[str, Any], content: bytes
    ) -> dict[str, Any]:
        """Extract Kotlin-specific metadata."""
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # Determine kind
                kind_map = {
                    "function_declaration": "function",
                    "class_declaration": "class",
                    "object_declaration": "object",
                    "property_declaration": "property",
                }
                metadata["kind"] = kind_map.get(def_node.type, "unknown")

                # Extract modifiers
                modifiers_node = self.find_child_by_type(def_node, "modifiers")
                if modifiers_node:
                    modifiers_text = self.get_node_text(modifiers_node, source)
                    metadata["modifiers"] = modifiers_text.strip()

        return metadata
