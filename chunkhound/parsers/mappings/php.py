"""PHP language mapping for unified parser architecture.

This module provides PHP-specific tree-sitter queries and extraction logic
for mapping PHP AST nodes to semantic chunks.

## Tree-sitter-php Node Types

Based on inspection of tree-sitter-php (verified via scripts/inspect_php_ast.py):
- Functions: function_definition (name child is "name" node)
- Methods: method_declaration (name child is "name" node)
- Classes: class_declaration (name child is "name" node)
- Interfaces: interface_declaration (name child is "name" node)
- Traits: trait_declaration (name child is "name" node)
- Namespaces: namespace_definition (namespace_name child contains "name" nodes)
- Comments: comment (includes //, /* */, and /** */ styles)
- Names: name (not "identifier")

Example AST structure for a function:
```
function_definition
├── name: "myFunction"
├── formal_parameters
│   ├── simple_parameter
│   │   └── variable_name: "$param1"
│   └── simple_parameter
│       └── variable_name: "$param2"
└── compound_statement
    └── return_statement
```

Example AST structure for a class with method:
```
class_declaration
├── name: "MyClass"
└── declaration_list
    └── method_declaration
        ├── visibility_modifier: "public"
        ├── name: "myMethod"
        ├── formal_parameters
        └── compound_statement
```
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class PHPMapping(BaseMapping):
    """PHP-specific tree-sitter mapping implementation.

    Handles PHP's unique language features including:
    - Namespaces with backslash separators
    - Classes, interfaces, traits
    - Visibility modifiers (public, private, protected)
    - Type hints and return types
    - PHPDoc comments
    - Static, abstract, and final modifiers
    """

    def __init__(self) -> None:
        """Initialize PHP mapping."""
        super().__init__(Language.PHP)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for PHP function definitions.

        Captures both standalone functions and methods within classes.
        Note: Methods are also captured here since they're semantically functions.
        """
        return """
            (function_definition
                name: (name) @function_name
            ) @function_def

            (method_declaration
                name: (name) @method_name
            ) @method_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for PHP class-like definitions.

        Captures classes, interfaces, and traits since they're all
        similar structural concepts in PHP.
        """
        return """
            (class_declaration
                name: (name) @class_name
            ) @class_def

            (interface_declaration
                name: (name) @interface_name
            ) @interface_def

            (trait_declaration
                name: (name) @trait_name
            ) @trait_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for PHP comments."""
        return """
            (comment) @comment
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a PHP function definition node."""
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        name_node = self.find_child_by_type(node, "name")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a PHP class definition node."""
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "class")

        name_node = self.find_child_by_type(node, "name")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "class")

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in PHP.

        This method maps PHP AST nodes to ChunkHound's universal semantic concepts.
        """

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_definition
                name: (name) @name
            ) @definition

            (method_declaration
                name: (name) @name
            ) @definition

            (class_declaration
                name: (name) @name
            ) @definition

            (interface_declaration
                name: (name) @name
            ) @definition

            (trait_declaration
                name: (name) @name
            ) @definition

            (const_declaration) @definition

            (expression_statement
                (function_call_expression
                    function: (name) @func_name
                    (#eq? @func_name "define")
                )
            ) @definition

            ;; Fallback for config-style files: top-level return statement
            (program
                (return_statement) @definition
            )
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (if_statement
                body: (compound_statement) @block
            )

            (while_statement
                body: (compound_statement) @block
            )

            (for_statement
                body: (compound_statement) @block
            )

            (foreach_statement
                body: (compound_statement) @block
            )

            (switch_statement
                body: (switch_block) @block
            )
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (namespace_use_declaration) @definition

            (namespace_definition
                name: (namespace_name) @namespace_name
            ) @definition
            """

        # PHP doesn't have a clear document-level structure concept like
        # Go's package or Python's module. The (program) node captures the
        # entire file which overlaps with all other chunks, causing them to
        # be merged into one large chunk. Return None for STRUCTURE and any
        # other unknown concepts.
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted name string
        """
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Try to get the name from the name capture
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()
                if name:
                    return name

            # Fallback based on node type
            if "definition" in captures:
                def_node = captures["definition"]
                # Name top-level return statements by line for discoverability
                if def_node.type == "return_statement":
                    line = def_node.start_point[0] + 1
                    return f"return_line_{line}"
                if def_node.type == "method_declaration":
                    return self.get_fallback_name(def_node, "method")
                elif def_node.type == "class_declaration":
                    return self.get_fallback_name(def_node, "class")
                elif def_node.type == "interface_declaration":
                    return self.get_fallback_name(def_node, "interface")
                elif def_node.type == "trait_declaration":
                    return self.get_fallback_name(def_node, "trait")
                elif def_node.type == "function_definition":
                    return self.get_fallback_name(def_node, "function")
                elif def_node.type == "const_declaration":
                    line = def_node.start_point[0] + 1
                    return f"const_line_{line}"
                elif def_node.type == "expression_statement":
                    line = def_node.start_point[0] + 1
                    return f"define_line_{line}"

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            if "block" in captures:
                node = captures["block"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"
            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                comment_text = self.get_node_text(node, source).strip()

                # Detect PHPDoc comments
                if comment_text.startswith("/**"):
                    return f"phpdoc_line_{line}"
                else:
                    return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "namespace_name" in captures:
                ns_node = captures["namespace_name"]
                ns_name = self.get_node_text(ns_node, source).strip()
                # Replace backslashes with underscores for safer symbol names
                safe_name = ns_name.replace("\\", "_")
                return f"namespace_{safe_name}"
            elif "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                text = self.get_node_text(node, source).strip()
                if "use " in text:
                    return f"use_line_{line}"
                elif "namespace " in text:
                    return f"namespace_line_{line}"
            return "unnamed_import"

        # For STRUCTURE or any unknown concept
        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract content from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Extracted content as string
        """
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
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract PHP-specific metadata from captures.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of capture names to tree-sitter nodes
            content: Source code as bytes

        Returns:
            Dictionary of metadata
        """
        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # Function or method
                if def_node.type in ("function_definition", "method_declaration"):
                    if def_node.type == "method_declaration":
                        metadata["kind"] = "method"
                        # Extract visibility
                        visibility = self._extract_visibility(def_node, source)
                        if visibility:
                            metadata["visibility"] = visibility
                        # Extract static modifier
                        if self._is_static(def_node, source):
                            metadata["is_static"] = True
                    else:
                        metadata["kind"] = "function"

                    # Extract parameters with type hints
                    parameters = self._extract_parameters(def_node, source)
                    if parameters:
                        metadata["parameters"] = parameters

                    # Extract return type
                    return_type = self._extract_return_type(def_node, source)
                    if return_type:
                        metadata["return_type"] = return_type

                # Class, interface, or trait
                elif def_node.type in (
                    "class_declaration",
                    "interface_declaration",
                    "trait_declaration",
                ):
                    if def_node.type == "class_declaration":
                        metadata["kind"] = "class"
                        # Check for abstract/final modifiers
                        if self._is_abstract(def_node, source):
                            metadata["is_abstract"] = True
                        if self._is_final(def_node, source):
                            metadata["is_final"] = True
                    elif def_node.type == "interface_declaration":
                        metadata["kind"] = "interface"
                    elif def_node.type == "trait_declaration":
                        metadata["kind"] = "trait"

                # Config returns (e.g., return [ 'key' => env('FOO') ])
                elif def_node.type == "return_statement":
                    metadata["kind"] = "return"
                    # Hint chunk type so downstream mapping doesn't default to FUNCTION
                    if self._contains_array_expression(def_node, source):
                        metadata["chunk_type_hint"] = "array"
                    else:
                        metadata["chunk_type_hint"] = "block"

                # Constants
                elif def_node.type == "const_declaration":
                    metadata["kind"] = "const"
                elif def_node.type == "expression_statement":
                    # Check if it's a define() call
                    for child in self.walk_tree(def_node):
                        if child and child.type == "function_call_expression":
                            name_node = self.find_child_by_type(child, "name")
                            if name_node:
                                func_name = self.get_node_text(name_node, source).strip()
                                if func_name == "define":
                                    metadata["kind"] = "define"
                                    break

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Detect comment type
                if comment_text.strip().startswith("/**"):
                    metadata["comment_type"] = "phpdoc"
                    metadata["is_docblock"] = True
                elif comment_text.strip().startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.strip().startswith("/*"):
                    metadata["comment_type"] = "block"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]

                if import_node.type == "namespace_use_declaration":
                    metadata["import_type"] = "use"
                elif import_node.type == "namespace_definition":
                    metadata["import_type"] = "namespace"

                if "namespace_name" in captures:
                    ns_node = captures["namespace_name"]
                    metadata["namespace"] = self.get_node_text(ns_node, source).strip()

        return metadata

    # PHP-specific helper methods for detailed metadata extraction

    def _extract_parameters(
        self, func_node: "TSNode", source: str
    ) -> list[dict[str, str]]:
        """Extract parameter names and types from a PHP function/method node.

        Args:
            func_node: Tree-sitter function/method node
            source: Source code string

        Returns:
            List of parameter dictionaries with 'name' and optionally 'type' keys
        """
        if not TREE_SITTER_AVAILABLE or func_node is None:
            return []

        parameters: list[dict[str, str]] = []

        # Find the formal_parameters node
        params_node = self.find_child_by_type(func_node, "formal_parameters")
        if not params_node:
            return parameters

        # Walk through and find parameter nodes
        for child in self.walk_tree(params_node):
            if child and child.type == "simple_parameter":
                param_info = {}

                # Extract parameter name (variable_name starts with $)
                for param_child in child.children:
                    if param_child.type == "variable_name":
                        param_info["name"] = self.get_node_text(
                            param_child, source
                        ).strip()
                    elif param_child.type in (
                        "primitive_type",
                        "named_type",
                        "optional_type",
                    ):
                        # Type hint
                        param_info["type"] = self.get_node_text(
                            param_child, source
                        ).strip()

                if param_info:
                    parameters.append(param_info)

        return parameters

    def _extract_return_type(self, func_node: "TSNode", source: str) -> str | None:
        """Extract return type hint from a PHP function/method node.

        Args:
            func_node: Tree-sitter function/method node
            source: Source code string

        Returns:
            Return type string or None if not specified
        """
        if not TREE_SITTER_AVAILABLE or func_node is None:
            return None

        # Look for return type (appears after : in function signature)
        for child in func_node.children:
            if child and child.type in (
                "primitive_type",
                "named_type",
                "optional_type",
                "union_type",
            ):
                # Check if this is after the parameter list (return type, not
                # parameter type). Return type appears after formal_parameters
                params_idx = -1
                for i, c in enumerate(func_node.children):
                    if c.type == "formal_parameters":
                        params_idx = i
                        break

                # If this type node comes after parameters, it's the return type
                child_idx = list(func_node.children).index(child)
                if params_idx >= 0 and child_idx > params_idx:
                    return self.get_node_text(child, source).strip()

        return None

    def _extract_visibility(self, node: "TSNode", source: str) -> str | None:
        """Extract visibility modifier (public, private, protected).

        Args:
            node: Tree-sitter node (method or property)
            source: Source code string

        Returns:
            Visibility string or None (defaults to public in PHP)
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return None

        # Look for visibility modifier nodes
        for child in node.children:
            if child and child.type == "visibility_modifier":
                return self.get_node_text(child, source).strip()

        return "public"  # Default visibility in PHP

    def _is_static(self, node: "TSNode", source: str) -> bool:
        """Check if method/property has static modifier.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if static, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Look for static_modifier node
        for child in node.children:
            if child and child.type == "static_modifier":
                return True

        return False

    def _is_abstract(self, node: "TSNode", source: str) -> bool:
        """Check if class/method has abstract modifier.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if abstract, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Look for abstract_modifier node
        for child in node.children:
            if child and child.type == "abstract_modifier":
                return True

        return False

    def _is_final(self, node: "TSNode", source: str) -> bool:
        """Check if class/method has final modifier.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if final, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Look for final_modifier node
        for child in node.children:
            if child and child.type == "final_modifier":
                return True

        return False

    def _contains_array_expression(self, node: "TSNode", source: str) -> bool:
        """Return True if the node subtree contains an array creation expression.

        Handles both short array syntax `[ ... ]` and classic `array(...)` forms.
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        try:
            for n in self.walk_tree(node):
                if n and n.type in (
                    "array_creation_expression",
                    "array_element_initializer",
                ):
                    return True
            # Defensive text check in case of grammar differences
            text = self.get_node_text(node, source)
            if text is None:
                return False
            if "[" in text and "]" in text:
                return True
            if "array(" in text:
                return True
        except Exception:
            return False
        return False

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from PHP code.

        Detects:
        - const declarations (const X = 5)
        - define() function calls (define('X', 5))

        Args:
            concept: The universal concept being processed
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dictionaries with "name" and "value" keys, or None
        """
        if not TREE_SITTER_AVAILABLE:
            return None

        source = content.decode("utf-8")
        constants: list[dict[str, str]] = []

        # Only process DEFINITION concept
        if concept != UniversalConcept.DEFINITION:
            return None

        def_node = captures.get("definition")
        if not def_node:
            return None

        # Handle const declarations (const X = 5;)
        if def_node.type == "const_declaration":
            # Walk through const_element nodes
            for child in self.walk_tree(def_node):
                if child and child.type == "const_element":
                    # Get the name
                    name_node = self.find_child_by_type(child, "name")
                    if name_node:
                        name = self.get_node_text(name_node, source).strip()

                        # Get the value
                        value_text = ""
                        for value_child in child.children:
                            if value_child.type in (
                                "integer",
                                "float",
                                "string",
                                "boolean",
                                "null",
                            ):
                                value_text = self.get_node_text(value_child, source).strip()
                                break

                        # Truncate to 50 chars if longer
                        if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                            value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                        constants.append({"name": name, "value": value_text})

        # Handle define() function calls
        elif def_node.type == "expression_statement":
            # Look for function_call_expression with name "define"
            for child in self.walk_tree(def_node):
                if child and child.type == "function_call_expression":
                    # Check if function name is "define"
                    name_node = self.find_child_by_type(child, "name")
                    if name_node:
                        func_name = self.get_node_text(name_node, source).strip()
                        if func_name == "define":
                            # Extract arguments from arguments node
                            args_node = self.find_child_by_type(child, "arguments")
                            if args_node:
                                # Get the first two arguments (name and value)
                                arg_nodes = [
                                    n
                                    for n in args_node.children
                                    if n.type != "," and n.type != "(" and n.type != ")"
                                ]

                                if len(arg_nodes) >= 2:
                                    # First argument is the constant name (usually a string)
                                    const_name_text = self.get_node_text(
                                        arg_nodes[0], source
                                    ).strip()
                                    # Remove quotes from string
                                    const_name = const_name_text.strip("\"'")

                                    # Second argument is the value
                                    const_value = self.get_node_text(
                                        arg_nodes[1], source
                                    ).strip()

                                    # Truncate to 50 chars if longer
                                    if len(const_value) > MAX_CONSTANT_VALUE_LENGTH:
                                        const_value = const_value[:MAX_CONSTANT_VALUE_LENGTH]

                                    constants.append(
                                        {"name": const_name, "value": const_value}
                                    )

        return constants if constants else None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path for PHP.

        Attempts to resolve relative require/include statements.

        Args:
            import_text: The import statement text
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            Path to the imported file if resolvable, None otherwise
        """
        match = re.search(
            r'(?:require|include)(?:_once)?\s*\(\s*[\'"](.+?)[\'"]\s*\)', import_text
        )
        if not match:
            return None

        path = match.group(1)
        if path.startswith("./") or path.startswith("../"):
            resolved = (source_file.parent / path).resolve()
            if resolved.exists():
                return resolved

        return None
