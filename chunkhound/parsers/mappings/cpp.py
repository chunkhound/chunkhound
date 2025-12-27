"""C++ language mapping for unified parser architecture.

This module provides C++-specific tree-sitter queries and extraction logic
for mapping C++ AST nodes to semantic chunks.
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
    TSNode = Any


class CppMapping(BaseMapping):
    """C++-specific tree-sitter mapping implementation.

    Handles C++ language features including:
    - Functions, methods, and function templates
    - Classes and class templates with inheritance
    - Namespaces and nested namespaces
    - Templates and template specialization
    - Constructors, destructors, and operator overloading
    - Access specifiers (public, private, protected)
    - Preprocessor directives
    - Comments (// and /* */)
    - All C features (structs, unions, enums, typedefs)
    """

    def __init__(self) -> None:
        """Initialize C++ mapping."""
        super().__init__(Language.CPP)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for C++ function definitions.

        Returns:
            Tree-sitter query string for finding C++ function definitions
        """
        return """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @function_name
                )
            ) @function_def

            (function_definition
                declarator: (pointer_declarator
                    declarator: (function_declarator
                        declarator: (identifier) @pointer_function_name
                    )
                )
            ) @pointer_function_def

            (template_declaration
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @template_function_name
                    )
                ) @template_function_def
            )
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for C++ class definitions.

        Returns:
            Tree-sitter query string for finding C++ class definitions
        """
        return """
            (class_specifier
                name: (type_identifier) @class_name
            ) @class_def

            (struct_specifier
                name: (type_identifier) @struct_name
            ) @struct_def

            (union_specifier
                name: (type_identifier) @union_name
            ) @union_def

            (enum_specifier
                name: (type_identifier) @enum_name
            ) @enum_def

            (template_declaration
                (class_specifier
                    name: (type_identifier) @template_class_name
                ) @template_class_def
            )
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for C++ method definitions.

        Methods are function definitions within class bodies or defined outside
        with scope resolution.

        Returns:
            Tree-sitter query string for finding C++ method definitions
        """
        return """
            (class_specifier
                body: (field_declaration_list
                    (function_definition
                        declarator: (function_declarator
                            declarator: (identifier) @method_name
                        )
                    ) @method_def
                )
            )

            (function_definition
                declarator: (function_declarator
                    declarator: (qualified_identifier
                        scope: (namespace_identifier) @class_scope
                        name: (identifier) @scoped_method_name
                    )
                )
            ) @scoped_method_def

            (template_declaration
                (function_definition
                    declarator: (function_declarator
                        declarator: (qualified_identifier
                            scope: (namespace_identifier) @template_class_scope
                            name: (identifier) @template_method_name
                        )
                    )
                ) @template_method_def
            )
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for C++ comments.

        Returns:
            Tree-sitter query string for finding C++ comments
        """
        return """
            (comment) @comment
        """

    def get_namespace_query(self) -> str:
        """Get tree-sitter query pattern for C++ namespace definitions.

        Returns:
            Tree-sitter query string for finding C++ namespace definitions
        """
        return """
            (namespace_definition
                (namespace_identifier) @namespace_name
            ) @namespace_def

            (namespace_definition
                (nested_namespace_specifier
                    (namespace_identifier) @nested_namespace_name
                )
            ) @nested_namespace_def
        """

    def get_template_query(self) -> str:
        """Get tree-sitter query pattern for C++ template definitions.

        Returns:
            Tree-sitter query string for finding C++ template definitions
        """
        return """
            (template_declaration
                (class_specifier
                    name: (type_identifier) @template_class_name
                )
            ) @template_class_def

            (template_declaration
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @template_function_name
                    )
                )
            ) @template_function_def

            (template_type
                name: (type_identifier) @template_instantiation_name
            ) @template_instantiation
        """

    def get_constructor_query(self) -> str:
        """Get tree-sitter query pattern for C++ constructor definitions.

        Returns:
            Tree-sitter query string for finding C++ constructor definitions
        """
        return """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @constructor_name
                )
                body: (compound_statement)
            ) @constructor_def
        """

    def get_destructor_query(self) -> str:
        """Get tree-sitter query pattern for C++ destructor definitions.

        Returns:
            Tree-sitter query string for finding C++ destructor definitions
        """
        return """
            (function_definition
                declarator: (function_declarator
                    declarator: (destructor_name
                        (identifier) @destructor_name
                    )
                )
            ) @destructor_def
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a C++ function definition node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "function")

        # Look for identifier nodes in the declarator hierarchy
        for child in self.walk_tree(node):
            if child and child.type == "identifier":
                # Make sure this identifier is part of the function declarator
                parent = child.parent
                while parent and parent != node:
                    if parent.type == "function_declarator":
                        name = self.get_node_text(child, source).strip()
                        if name:
                            return name
                        break
                    parent = parent.parent

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a C++ class definition node.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            Class name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "class")

        # Look for the type_identifier child node
        name_node = self.find_child_by_type(node, "type_identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        # Try identifier as fallback
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "class")

    def extract_namespace_name(self, node: "TSNode | None", source: str) -> str:
        """Extract namespace name from a C++ namespace definition.

        Args:
            node: Tree-sitter namespace definition node
            source: Source code string

        Returns:
            Namespace name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "namespace")

        # Look for the identifier child node
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        # Handle nested namespace specifier
        nested_node = self.find_child_by_type(node, "nested_namespace_specifier")
        if nested_node:
            # Get all identifiers in the nested namespace
            identifiers = self.find_children_by_type(nested_node, "identifier")
            if identifiers:
                names = [
                    self.get_node_text(id_node, source).strip()
                    for id_node in identifiers
                ]
                return "::".join(names)

        return self.get_fallback_name(node, "namespace")

    def extract_template_parameters(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract template parameters from a template declaration.

        Args:
            node: Tree-sitter template declaration node
            source: Source code string

        Returns:
            List of template parameter names
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters: list[str] = []

        # Find template_parameter_list
        param_list = self.find_child_by_type(node, "template_parameter_list")
        if param_list:
            # Look for type_parameter_declaration and template_template_parameter_declaration
            for child in self.walk_tree(param_list):
                if child and child.type in (
                    "type_parameter_declaration",
                    "template_template_parameter_declaration",
                ):
                    # Find identifier within the parameter declaration
                    name_node = self.find_child_by_type(child, "type_identifier")
                    if name_node:
                        param_name = self.get_node_text(name_node, source).strip()
                        if param_name:
                            parameters.append(param_name)

        return parameters

    def extract_inheritance(
        self, node: "TSNode | None", source: str
    ) -> list[dict[str, str]]:
        """Extract inheritance information from a C++ class definition.

        Args:
            node: Tree-sitter class definition node
            source: Source code string

        Returns:
            List of dictionaries with inheritance information
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        inheritance_info: list[dict[str, str]] = []

        # Find base_class_clause
        base_clause = self.find_child_by_type(node, "base_class_clause")
        if base_clause:
            # Walk through access_specifier and type_identifier pairs
            access_spec = "public"  # Default for struct, private for class
            if node.type == "class_specifier":
                access_spec = "private"

            for child in self.walk_tree(base_clause):
                if child and child.type == "access_specifier":
                    access_spec = self.get_node_text(child, source).strip()
                elif child and child.type in (
                    "type_identifier",
                    "qualified_identifier",
                ):
                    base_class = self.get_node_text(child, source).strip()
                    if base_class:
                        inheritance_info.append(
                            {"base_class": base_class, "access_specifier": access_spec}
                        )

        return inheritance_info

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a C++ function node.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            List of parameter declarations
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return []

        parameters: list[str] = []

        # Find the parameter_list node
        for child in self.walk_tree(node):
            if child and child.type == "parameter_list":
                # Walk through parameter declarations
                for param_node in self.find_children_by_type(
                    child, "parameter_declaration"
                ):
                    param_text = self.get_node_text(param_node, source).strip()
                    if param_text and param_text not in ("(", ")", ","):
                        parameters.append(param_text)
                break

        return parameters

    def is_constructor(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function node represents a constructor.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function is a constructor, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check if function name matches class name
        func_name = self.extract_function_name(node, source)

        # Find the parent class if this is defined inside a class
        parent = node.parent
        while parent:
            if parent.type == "class_specifier":
                class_name = self.extract_class_name(parent, source)
                return func_name == class_name
            parent = parent.parent

        return False

    def is_destructor(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function node represents a destructor.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            True if the function is a destructor, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Look for destructor_name in the declarator
        for child in self.walk_tree(node):
            if child and child.type == "destructor_name":
                return True

        return False

    def is_template(self, node: "TSNode | None") -> bool:
        """Check if a node is part of a template declaration.

        Args:
            node: Tree-sitter node

        Returns:
            True if the node is templated, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Check if parent is template_declaration
        parent = node.parent
        while parent:
            if parent.type == "template_declaration":
                return True
            parent = parent.parent

        return False

    # Universal Concept System Integration

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in C++."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name
                )
            ) @definition

            (class_specifier
                name: (type_identifier) @name
            ) @definition

            (struct_specifier
                name: (type_identifier) @name
            ) @definition

            (union_specifier
                name: (type_identifier) @name
            ) @definition

            (enum_specifier
                name: (type_identifier) @name
            ) @definition

            (namespace_definition
                (namespace_identifier) @name
            ) @definition

            (template_declaration
                (class_specifier
                    name: (type_identifier) @name
                )
            ) @definition

            (template_declaration
                (function_definition
                    declarator: (function_declarator
                        declarator: (identifier) @name
                    )
                )
            ) @definition

            ; Capture all declarations with const/constexpr at any scope
            ; This includes top-level, function-scoped, and nested block declarations
            (declaration
                (type_qualifier)
            ) @definition
            """

        elif concept == UniversalConcept.BLOCK:
            return """
            (compound_statement) @definition

            (if_statement
                consequence: (compound_statement) @block
            ) @definition

            (while_statement
                body: (compound_statement) @block
            ) @definition

            (for_statement
                body: (compound_statement) @block
            ) @definition

            (switch_statement
                body: (compound_statement) @block
            ) @definition

            (try_statement
                body: (compound_statement) @try_block
                (catch_clause
                    body: (compound_statement) @catch_block
                )*
            ) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (preproc_include
                path: (_) @include_path
            ) @definition

            (preproc_def
                name: (identifier) @define_name
            ) @definition

            (using_declaration
                (qualified_identifier) @using_name
            ) @definition

            (namespace_alias_definition
                (namespace_identifier) @alias_name
                (nested_namespace_specifier) @alias_value
            ) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (translation_unit) @definition
            """

        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for processing
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Try to get the name from the capture
            if "name" in captures:
                name_node = captures["name"]
                name = self.get_node_text(name_node, source).strip()
                if name:
                    # Check if this is a template
                    def_node = captures.get("definition")
                    if def_node and self.is_template(def_node):
                        # Add template parameters
                        template_params = self.extract_template_parameters(
                            def_node.parent, source
                        )
                        if template_params:
                            return f"{name}<{','.join(template_params)}>"
                    return name

            return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Use location-based naming for blocks
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"block_line_{line}"

            return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use location-based naming for comments
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"

            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "include_path" in captures:
                path_node = captures["include_path"]
                path = self.get_node_text(path_node, source).strip()
                # Remove quotes and brackets
                path = path.strip('"<>')
                # Extract filename from path
                filename = path.split("/")[-1].replace(".", "_")
                return f"include_{filename}"
            elif "define_name" in captures:
                define_node = captures["define_name"]
                define_name = self.get_node_text(define_node, source).strip()
                return f"define_{define_name}"
            elif "using_name" in captures:
                using_node = captures["using_name"]
                using_name = self.get_node_text(using_node, source).strip()
                # Simplify qualified names
                parts = using_name.split("::")
                return f"using_{parts[-1]}"
            elif "alias_name" in captures:
                alias_node = captures["alias_name"]
                alias_name = self.get_node_text(alias_node, source).strip()
                return f"namespace_alias_{alias_name}"

            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
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
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract C++-specific metadata."""

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            # Extract definition-specific metadata
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For functions, extract parameters and return type
                if def_node.type == "function_definition":
                    metadata["kind"] = "function"
                    params = self.extract_parameters(def_node, source)
                    metadata["parameters"] = params
                    return_type = self._extract_return_type(def_node, source)
                    if return_type:
                        metadata["return_type"] = return_type

                    # Check if it's a constructor or destructor
                    if self.is_constructor(def_node, source):
                        metadata["kind"] = "constructor"
                    elif self.is_destructor(def_node, source):
                        metadata["kind"] = "destructor"

                    # Check if it's templated
                    if self.is_template(def_node):
                        metadata["is_template"] = True
                        template_params = self.extract_template_parameters(
                            def_node.parent, source
                        )
                        metadata["template_parameters"] = template_params

                # For classes, extract inheritance and template info
                elif def_node.type in ("class_specifier", "struct_specifier"):
                    metadata["kind"] = def_node.type.replace("_specifier", "")
                    inheritance = self.extract_inheritance(def_node, source)
                    if inheritance:
                        metadata["inheritance"] = inheritance

                    if self.is_template(def_node):
                        metadata["is_template"] = True
                        template_params = self.extract_template_parameters(
                            def_node.parent, source
                        )
                        metadata["template_parameters"] = template_params

                # For namespaces, extract nested namespace info
                elif def_node.type == "namespace_definition":
                    metadata["kind"] = "namespace"

                # For unions and enums
                elif def_node.type in ("union_specifier", "enum_specifier"):
                    metadata["kind"] = def_node.type.replace("_specifier", "")

        elif concept == UniversalConcept.IMPORT:
            if "include_path" in captures:
                path_node = captures["include_path"]
                include_path = (
                    self.get_node_text(path_node, source).strip().strip('"<>')
                )
                metadata["include_path"] = include_path

                # Determine if it's a system or local include
                if include_path.startswith("<") or not include_path.startswith('"'):
                    metadata["include_type"] = "system"
                else:
                    metadata["include_type"] = "local"

            elif "define_name" in captures:
                define_node = captures["define_name"]
                metadata["define_name"] = self.get_node_text(
                    define_node, source
                ).strip()
                metadata["directive_type"] = "define"

            elif "using_name" in captures:
                using_node = captures["using_name"]
                metadata["using_name"] = self.get_node_text(using_node, source).strip()
                metadata["directive_type"] = "using"

            elif "alias_name" in captures and "alias_value" in captures:
                alias_name_node = captures["alias_name"]
                alias_value_node = captures["alias_value"]
                metadata["alias_name"] = self.get_node_text(
                    alias_name_node, source
                ).strip()
                metadata["alias_value"] = self.get_node_text(
                    alias_value_node, source
                ).strip()
                metadata["directive_type"] = "namespace_alias"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Determine comment type
                if comment_text.startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.startswith("/*"):
                    metadata["comment_type"] = "block"

                # Check for doxygen-style comments
                if comment_text.startswith("///") or comment_text.startswith("/**"):
                    metadata["is_doc_comment"] = True

        return metadata

    def _extract_return_type(self, func_node: "TSNode", source: str) -> str | None:
        """Extract return type from a C++ function node."""
        if not TREE_SITTER_AVAILABLE or func_node is None:
            return None

        # Look for type specifiers before the function declarator
        for i in range(func_node.child_count):
            child = func_node.child(i)
            if child and child.type in (
                "primitive_type",
                "type_identifier",
                "qualified_identifier",
                "template_type",
            ):
                return self.get_node_text(child, source).strip()

        return None

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a C++ node should be included as a chunk.

        Filters out very small nodes, forward declarations, and access specifiers.

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

        # Skip very small nodes (less than 10 characters)
        if len(text.strip()) < 10:
            return False

        # Skip access specifiers alone
        if node.type == "access_specifier":
            return False

        # For function definitions, skip forward declarations (those without body)
        if node.type == "function_definition":
            # Check if there's a compound_statement (function body)
            body_node = self.find_child_by_type(node, "compound_statement")
            if not body_node:
                return False

        # For class definitions, include even if they're just declarations
        # since they provide valuable type information
        return True

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from C++ code.

        Detects:
        - const qualified variable declarations (at any scope)
        - constexpr declarations (at any scope)

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

        # Only process declaration nodes
        if def_node.type != "declaration":
            return None

        # Check if declaration has const or constexpr qualifier
        has_const = False
        has_constexpr = False
        for child in self.walk_tree(def_node):
            if child and child.type == "type_qualifier":
                qualifier_text = self.get_node_text(child, source).strip()
                if qualifier_text == "const":
                    has_const = True
                elif qualifier_text == "constexpr":
                    has_constexpr = True

        if not (has_const or has_constexpr):
            return None

        # Extract variable name and initializer value
        for child in self.walk_tree(def_node):
            if child and child.type == "init_declarator":
                # Get the identifier (variable name) - may be nested in pointer_declarator
                name = ""
                for id_child in self.walk_tree(child):
                    if id_child and id_child.type == "identifier":
                        name = self.get_node_text(id_child, source).strip()
                        break

                if not name:
                    continue

                # Get the initializer value
                value_text = ""
                for init_child in self.walk_tree(child):
                    if init_child and init_child.type in (
                        "number_literal",
                        "string_literal",
                        "char_literal",
                        "true",
                        "false",
                        "null",  # nullptr is wrapped in a 'null' node
                    ):
                        value_text = self.get_node_text(init_child, source).strip()
                        break

                # Truncate to 50 chars if longer
                if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                    value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                if value_text:
                    constants.append({"name": name, "value": value_text})

        return constants if constants else None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve C++ include to file path.

        Args:
            import_text: The import statement text (e.g., '#include "file.h"')
            base_dir: Base directory of the codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise (system includes)
        """
        # Local includes: #include "file.h"
        local_match = re.search(r'#include\s*"(.+?)"', import_text)
        if local_match:
            include_path = local_match.group(1)

            # Try relative to source file
            resolved = (source_file.parent / include_path).resolve()
            if resolved.exists():
                return resolved

            # Try relative to base_dir and common include paths
            for prefix in ["", "include/", "src/", "inc/"]:
                full_path = base_dir / prefix / include_path
                if full_path.exists():
                    return full_path

        # System includes (#include <...>) - external, return None
        return None
