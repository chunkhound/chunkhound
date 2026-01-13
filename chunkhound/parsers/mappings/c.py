"""C language mapping for unified parser architecture.

This module provides C-specific tree-sitter queries and extraction logic
for mapping C AST nodes to semantic chunks.
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


class CMapping(BaseMapping):
    """C-specific tree-sitter mapping implementation.

    Handles C's language features including:
    - Functions and function declarations
    - Structs, unions, and enums
    - Typedefs and type aliases
    - Preprocessor directives (#include, #define, #ifdef, etc.)
    - Global variables and static variables
    - Comments (// and /* */)
    """

    def __init__(self) -> None:
        """Initialize C mapping."""
        super().__init__(Language.C)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for C function definitions.

        Returns:
            Tree-sitter query string for finding C function definitions
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
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for C struct definitions.

        C doesn't have classes, but structs serve a similar purpose.

        Returns:
            Tree-sitter query string for finding C struct definitions
        """
        return """
            (struct_specifier
                name: (type_identifier) @struct_name
            ) @struct_def

            (union_specifier
                name: (type_identifier) @union_name
            ) @union_def

            (enum_specifier
                name: (type_identifier) @enum_name
            ) @enum_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for C comments.

        Returns:
            Tree-sitter query string for finding C comments
        """
        return """
            (comment) @comment
        """

    def get_typedef_query(self) -> str:
        """Get tree-sitter query pattern for C typedef statements.

        Returns:
            Tree-sitter query string for finding C typedef statements
        """
        return """
            (type_definition
                declarator: (type_identifier) @typedef_name
            ) @typedef_def
        """

    def get_preprocessor_query(self) -> str:
        """Get tree-sitter query pattern for C preprocessor directives.

        Returns:
            Tree-sitter query string for finding C preprocessor directives
        """
        return """
            (preproc_include
                path: (_) @include_path
            ) @include_directive

            (preproc_def
                name: (identifier) @define_name
            ) @define_directive

            (preproc_ifdef
                name: (identifier) @ifdef_name
            ) @ifdef_directive

            (preproc_ifndef
                name: (identifier) @ifndef_name
            ) @ifndef_directive
        """

    def get_variable_query(self) -> str:
        """Get tree-sitter query pattern for C variable declarations.

        Returns:
            Tree-sitter query string for finding C variable declarations
        """
        return """
            (declaration
                declarator: (identifier) @var_name
            ) @var_declaration

            (declaration
                declarator: (init_declarator
                    declarator: (identifier) @init_var_name
                )
            ) @init_var_declaration
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a C function definition node.

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
        """Extract struct/union/enum name from a C definition node.

        Args:
            node: Tree-sitter struct/union/enum definition node
            source: Source code string

        Returns:
            Struct/union/enum name or fallback name if extraction fails
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "struct")

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

        return self.get_fallback_name(node, "struct")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a C function node.

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

    def extract_preprocessor_name(self, node: "TSNode | None", source: str) -> str:
        """Extract name from a preprocessor directive.

        Args:
            node: Tree-sitter preprocessor directive node
            source: Source code string

        Returns:
            Preprocessor directive name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "preprocessor")

        # For #include directives, extract the path
        if node.type == "preproc_include":
            path_node = None
            for child in self.walk_tree(node):
                if child and child.type in ("string_literal", "system_lib_string"):
                    path_node = child
                    break
            if path_node:
                path = self.get_node_text(path_node, source).strip()
                # Clean up quotes and brackets
                path = path.strip('"<>')
                return f"include_{path.replace('/', '_').replace('.', '_')}"

        # For #define, #ifdef, #ifndef, extract the identifier
        name_node = self.find_child_by_type(node, "identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                directive_type = node.type.replace("preproc_", "")
                return f"{directive_type}_{name}"

        return self.get_fallback_name(node, "preprocessor")

    def extract_variable_name(self, node: "TSNode | None", source: str) -> str:
        """Extract variable name from a C variable declaration.

        Args:
            node: Tree-sitter variable declaration node
            source: Source code string

        Returns:
            Variable name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "variable")

        # Look for identifier nodes in the declarator
        for child in self.walk_tree(node):
            if child and child.type == "identifier":
                # Check if this is the variable name (not a type)
                name = self.get_node_text(child, source).strip()
                if (
                    name and not name.isupper()
                ):  # Avoid type names which are often uppercase
                    return name

        return self.get_fallback_name(node, "variable")

    def extract_typedef_name(self, node: "TSNode | None", source: str) -> str:
        """Extract typedef name from a C typedef definition.

        Args:
            node: Tree-sitter typedef node
            source: Source code string

        Returns:
            Typedef name or fallback
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "typedef")

        # Look for the type_identifier (the new type name)
        name_node = self.find_child_by_type(node, "type_identifier")
        if name_node:
            name = self.get_node_text(name_node, source).strip()
            if name:
                return name

        return self.get_fallback_name(node, "typedef")

    # Universal Concept System Integration

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in C."""

        if concept == UniversalConcept.DEFINITION:
            return """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @name
                )
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

            (type_definition
                declarator: (type_identifier) @name
            ) @definition

            (preproc_def
                name: (identifier) @name
            ) @definition

            (declaration
                declarator: (identifier) @name
            ) @definition

            (declaration
                declarator: (init_declarator
                    declarator: (identifier) @name
                )
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
        """Extract C-specific metadata."""

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

                # For struct/union/enum, extract type kind
                elif def_node.type in (
                    "struct_specifier",
                    "union_specifier",
                    "enum_specifier",
                ):
                    metadata["kind"] = def_node.type.replace("_specifier", "")

                # For typedefs, extract underlying type
                elif def_node.type == "type_definition":
                    metadata["kind"] = "typedef"
                    underlying_type = self._extract_typedef_underlying_type(
                        def_node, source
                    )
                    if underlying_type:
                        metadata["underlying_type"] = underlying_type

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

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Determine comment type
                if comment_text.startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.startswith("/*"):
                    metadata["comment_type"] = "block"

        return metadata

    def _extract_return_type(self, func_node: "TSNode", source: str) -> str | None:
        """Extract return type from a C function node."""
        if not TREE_SITTER_AVAILABLE or func_node is None:
            return None

        # Look for type specifiers before the function declarator
        for i in range(func_node.child_count):
            child = func_node.child(i)
            if child and child.type in (
                "primitive_type",
                "type_identifier",
                "struct_specifier",
                "union_specifier",
            ):
                return self.get_node_text(child, source).strip()

        return None

    def _extract_typedef_underlying_type(
        self, typedef_node: "TSNode", source: str
    ) -> str | None:
        """Extract underlying type from a typedef definition."""
        if not TREE_SITTER_AVAILABLE or typedef_node is None:
            return None

        # Look for the type being aliased (before the declarator)
        for i in range(typedef_node.child_count):
            child = typedef_node.child(i)
            if child and child.type in (
                "primitive_type",
                "type_identifier",
                "struct_specifier",
                "union_specifier",
                "enum_specifier",
            ):
                return self.get_node_text(child, source).strip()

        return None

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a C node should be included as a chunk.

        Filters out very small nodes and forward declarations.

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

        # For function definitions, skip forward declarations (those without body)
        if node.type == "function_definition":
            # Check if there's a compound_statement (function body)
            body_node = self.find_child_by_type(node, "compound_statement")
            if not body_node:
                return False

        return True

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from C code.

        Detects:
        - #define macros (preprocessor definitions)
        - const qualified variable declarations

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

        # Handle #define macros (preproc_def)
        if def_node.type == "preproc_def":
            # Skip function-like macros (e.g., #define SQUARE(x) ((x) * (x)))
            if self.find_child_by_type(def_node, "preproc_params"):
                return None

            # Extract the macro name
            name_node = self.find_child_by_type(def_node, "identifier")
            if name_node:
                name = self.get_node_text(name_node, source).strip()

                # Extract the macro value from preproc_arg
                value_node = self.find_child_by_type(def_node, "preproc_arg")
                if value_node:
                    value = self.get_node_text(value_node, source).strip()
                    # Truncate to 50 chars if longer
                    if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                        value = value[:MAX_CONSTANT_VALUE_LENGTH]
                    constants.append({"name": name, "value": value})
                else:
                    # Macro without value (e.g., #define FLAG)
                    constants.append({"name": name, "value": ""})

        # Handle const qualified declarations
        elif def_node.type == "declaration":
            # Check if declaration has const qualifier
            has_const = False
            for child in self.walk_tree(def_node):
                if child and child.type == "type_qualifier":
                    qualifier_text = self.get_node_text(child, source).strip()
                    if qualifier_text == "const":
                        has_const = True
                        break

            if has_const:
                # Extract variable name and initializer value
                for child in self.walk_tree(def_node):
                    if child and child.type == "init_declarator":
                        # Get the identifier (variable name)
                        name_node = self.find_child_by_type(child, "identifier")
                        if name_node:
                            name = self.get_node_text(name_node, source).strip()

                            # Get the initializer value
                            value_text = ""
                            for init_child in self.walk_tree(child):
                                if init_child and init_child.type in (
                                    "number_literal",
                                    "string_literal",
                                    "char_literal",
                                    "true",
                                    "false",
                                ):
                                    value_text = self.get_node_text(
                                        init_child, source
                                    ).strip()
                                    break

                            # Truncate to 50 chars if longer
                            if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                                value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                            if value_text:
                                constants.append({"name": name, "value": value_text})

        # Handle enum constants
        elif def_node.type == "enum_specifier":
            # Find the enumerator_list node
            enum_list_node = self.find_child_by_type(def_node, "enumerator_list")
            if enum_list_node:
                # Extract each enumerator
                for child in self.walk_tree(enum_list_node):
                    if child and child.type == "enumerator":
                        # Get the identifier (enum member name)
                        name_node = self.find_child_by_type(child, "identifier")
                        if name_node:
                            name = self.get_node_text(name_node, source).strip()
                            # Enum constants use their name as the value
                            constants.append({"name": name, "value": name})

        return constants if constants else None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve C include to file path.

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
