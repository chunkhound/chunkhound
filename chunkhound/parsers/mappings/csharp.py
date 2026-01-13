"""C# language mapping for the unified parser architecture.

This module provides C#-specific tree-sitter queries and extraction logic
for the unified parser system. It handles C#'s object-oriented features
including classes, interfaces, methods, properties, namespaces, attributes,
and XML documentation comments.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import Language

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode


class CSharpMapping(BaseMapping):
    """C#-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize C# mapping."""
        super().__init__(Language.CSHARP)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for C# method definitions.

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_declaration
            name: (identifier) @method_name
        ) @method_def

        (local_function_statement
            name: (identifier) @local_function_name
        ) @local_function_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for C# class definitions.

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

        (struct_declaration
            name: (identifier) @struct_name
        ) @struct_def

        (enum_declaration
            name: (identifier) @enum_name
        ) @enum_def

        (record_declaration
            name: (identifier) @record_name
        ) @record_def

        (delegate_declaration
            name: (identifier) @delegate_name
        ) @delegate_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for C# comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (comment) @comment
        """

    def get_method_query(self) -> str:
        """Get tree-sitter query pattern for C# method definitions.

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

        (destructor_declaration
            name: (identifier) @destructor_name
        ) @destructor_def

        (operator_declaration
            type: (_) @operator_type
        ) @operator_def

        (conversion_operator_declaration
            type: (_) @conversion_type
        ) @conversion_def

        (property_declaration
            name: (identifier) @property_name
        ) @property_def

        (indexer_declaration) @indexer_def

        (event_declaration
            name: (identifier) @event_name
        ) @event_def
        """

    def get_docstring_query(self) -> str:
        """Get tree-sitter query pattern for XML documentation comments.

        Returns:
            Tree-sitter query string for finding XML doc comments
        """
        return """
        (comment) @xml_doc
        """

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for universal concept in C#."""
        from chunkhound.parsers.universal_engine import UniversalConcept

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

            (struct_declaration
                name: (identifier) @name
            ) @definition

            (enum_declaration
                name: (identifier) @name
            ) @definition

            (record_declaration
                name: (identifier) @name
            ) @definition

            (delegate_declaration
                name: (identifier) @name
            ) @definition

            (property_declaration
                name: (identifier) @name
            ) @definition

            (field_declaration) @definition

            (local_declaration_statement) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (using_directive) @definition
            """

        elif concept == UniversalConcept.STRUCTURE:
            return """
            (compilation_unit
                (using_directive)* @imports
                (namespace_declaration)? @namespace
            ) @definition
            """

        else:
            return None

    def extract_name(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        from chunkhound.parsers.universal_engine import UniversalConcept

        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Use @name capture if available
            if "name" in captures:
                return self.get_node_text(captures["name"], source).strip()

            # For field_declaration and local_declaration_statement,
            # extract variable name
            def_node = captures.get("definition")
            if def_node and def_node.type in (
                "field_declaration",
                "local_declaration_statement",
            ):
                declarator = self.find_child_by_type(def_node, "variable_declarator")
                if declarator:
                    var_name_node = self.find_child_by_type(declarator, "identifier")
                    if var_name_node:
                        return self.get_node_text(var_name_node, source).strip()
                line = def_node.start_point[0] + 1
                node_prefix = (
                    "local"
                    if def_node.type == "local_declaration_statement"
                    else "field"
                )
                return f"{node_prefix}_line_{line}"

            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"
            return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                node = captures["definition"]
                return self.get_node_text(node, source).strip()
            return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "file_structure"

        return "unnamed"

    def extract_content(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
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
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract C#-specific metadata."""
        from chunkhound.parsers.universal_engine import UniversalConcept

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
                    "struct_declaration": "struct",
                    "enum_declaration": "enum",
                    "record_declaration": "record",
                    "delegate_declaration": "delegate",
                    "property_declaration": "property",
                    "field_declaration": "field",
                    "local_declaration_statement": "local_constant",
                }
                metadata["kind"] = kind_map.get(def_node.type, "unknown")

                # Extract modifiers
                modifiers = self.extract_access_modifiers(def_node, source)
                if modifiers:
                    metadata["modifiers"] = modifiers

        return metadata

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a C# method definition node.

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

            # Special handling for different node types
            if node.type == "operator_declaration":
                # Extract operator type (e.g., "+", "==", "implicit")
                type_nodes = []
                for child in self.walk_tree(node):
                    if child and child.type in [
                        "binary_operator_token",
                        "unary_operator_token",
                        "implicit_keyword",
                        "explicit_keyword",
                    ]:
                        type_nodes.append(child)

                if type_nodes:
                    operator_type = self.get_node_text(type_nodes[0], source).strip()
                    return f"operator {operator_type}"

            elif node.type == "conversion_operator_declaration":
                # Get conversion target type
                for child in self.walk_tree(node):
                    if child and child.type in [
                        "predefined_type",
                        "identifier_name",
                        "generic_name",
                    ]:
                        target_type = self.get_node_text(child, source).strip()
                        return f"operator {target_type}"

            elif node.type == "indexer_declaration":
                return "this[]"

            # Fallback: look for the first identifier
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract C# method name: {e}")

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from a C# class definition node.

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
            logger.error(f"Failed to extract C# class name: {e}")

        return self.get_fallback_name(node, "class")

    def extract_method_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from a C# method definition node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name or fallback name if extraction fails
        """
        # Delegate to extract_function_name as C# methods are functions
        return self.extract_function_name(node, source)

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names and types from a C# method node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            List of parameter type strings
        """
        if node is None:
            return []

        parameters: list[str] = []
        try:
            # Find parameter_list node
            params_node = self.find_child_by_type(node, "parameter_list")
            if not params_node:
                return parameters

            # Extract each parameter
            for child in self.find_children_by_type(params_node, "parameter"):
                param_text = self.get_node_text(child, source).strip()
                if param_text:
                    parameters.append(param_text)

        except Exception as e:
            logger.error(f"Failed to extract C# method parameters: {e}")

        return parameters

    def extract_namespace_name(self, root_node: "TSNode | None", source: str) -> str:
        """Extract namespace name from C# file.

        Args:
            root_node: Root node of the C# AST
            source: Source code string

        Returns:
            Namespace name as string, or empty string if no namespace declaration found
        """
        if root_node is None:
            return ""

        try:
            # Look for namespace_declaration or file_scoped_namespace_declaration
            namespace_nodes = self.find_nodes_by_type(
                root_node, "namespace_declaration"
            )
            if not namespace_nodes:
                namespace_nodes = self.find_nodes_by_type(
                    root_node, "file_scoped_namespace_declaration"
                )

            if not namespace_nodes:
                return ""

            namespace_node = namespace_nodes[0]

            # Find the qualified_name or identifier
            qualified_name_node = self.find_child_by_type(
                namespace_node, "qualified_name"
            )
            if qualified_name_node:
                return self.get_node_text(qualified_name_node, source).strip()

            # Fallback to identifier
            identifier_node = self.find_child_by_type(namespace_node, "identifier")
            if identifier_node:
                return self.get_node_text(identifier_node, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract C# namespace name: {e}")

        return ""

    def extract_using_statements(
        self, root_node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract using statements from C# file.

        Args:
            root_node: Root node of the C# AST
            source: Source code string

        Returns:
            List of using statement namespaces
        """
        if root_node is None:
            return []

        using_statements = []
        try:
            # Look for using_directive nodes
            using_nodes = self.find_nodes_by_type(root_node, "using_directive")

            for using_node in using_nodes:
                using_text = self.get_node_text(using_node, source).strip()
                if using_text.startswith("using ") and using_text.endswith(";"):
                    namespace = using_text[6:-1].strip()
                    # Skip using aliases (contains =)
                    if "=" not in namespace:
                        using_statements.append(namespace)

        except Exception as e:
            logger.error(f"Failed to extract C# using statements: {e}")

        return using_statements

    def extract_attributes(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract C# attributes from a node.

        Args:
            node: Tree-sitter node to extract attributes from
            source: Source code string

        Returns:
            List of attribute strings
        """
        if node is None:
            return []

        attributes = []
        try:
            # Look for attribute_list nodes
            attr_list_nodes = self.find_children_by_type(node, "attribute_list")

            for attr_list_node in attr_list_nodes:
                # Find individual attributes within the list
                attr_nodes = self.find_children_by_type(attr_list_node, "attribute")

                for attr_node in attr_nodes:
                    attr_text = self.get_node_text(attr_node, source).strip()
                    if attr_text:
                        attributes.append(f"[{attr_text}]")

        except Exception as e:
            logger.error(f"Failed to extract C# attributes: {e}")

        return attributes

    def extract_type_parameters(self, node: "TSNode | None", source: str) -> str:
        """Extract generic type parameters from a C# node.

        Args:
            node: Tree-sitter node to extract type parameters from
            source: Source code string

        Returns:
            Type parameters string (e.g., "<T, U> where T : class")
        """
        if node is None:
            return ""

        try:
            # Find type_parameter_list
            type_params_node = self.find_child_by_type(node, "type_parameter_list")
            if type_params_node:
                type_params = self.get_node_text(type_params_node, source).strip()

                # Look for type parameter constraints
                constraints_nodes = self.find_children_by_type(
                    node, "type_parameter_constraints_clause"
                )
                if constraints_nodes:
                    constraints_text = " ".join(
                        self.get_node_text(constraint_node, source).strip()
                        for constraint_node in constraints_nodes
                    )
                    return f"{type_params} {constraints_text}"

                return type_params

        except Exception as e:
            logger.error(f"Failed to extract C# type parameters: {e}")

        return ""

    def extract_return_type(self, node: "TSNode | None", source: str) -> str | None:
        """Extract return type from a C# method node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Return type string or None if not found/constructor
        """
        if node is None:
            return None

        try:
            # Constructor and destructor declarations don't have return types
            if node.type in ["constructor_declaration", "destructor_declaration"]:
                return None

            # Look for return type - it's typically the first type-related node
            for child in self.walk_tree(node):
                if child and child.type in [
                    "predefined_type",
                    "identifier_name",
                    "generic_name",
                    "array_type",
                    "nullable_type",
                    "pointer_type",
                    "qualified_name",
                ]:
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract C# return type: {e}")

        return None

    def extract_access_modifiers(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract access modifiers from a C# declaration.

        Args:
            node: Tree-sitter declaration node
            source: Source code string

        Returns:
            List of access modifiers (public, private, protected, internal, etc.)
        """
        if node is None:
            return []

        modifiers = []
        try:
            # Look for modifier tokens
            # Tree-sitter C# grammar uses bare keyword names as node types
            # (e.g., "static", "readonly")
            # NOT suffixed with "_keyword" (e.g., NOT "static_keyword")
            for child in self.walk_tree(node):
                if child and child.type in [
                    "public",
                    "private",
                    "protected",
                    "internal",
                    "static",
                    "readonly",
                    "abstract",
                    "virtual",
                    "override",
                    "sealed",
                    "async",
                    "extern",
                    "partial",
                    "unsafe",
                    "const",
                ]:
                    modifier_text = self.get_node_text(child, source).strip()
                    if modifier_text:
                        modifiers.append(modifier_text)

        except Exception as e:
            logger.error(f"Failed to extract C# access modifiers: {e}")

        return modifiers

    def is_xml_doc_comment(self, node: "TSNode | None", source: str) -> bool:
        """Check if a comment node is an XML documentation comment.

        Args:
            node: Tree-sitter comment node
            source: Source code string

        Returns:
            True if the comment is XML doc-style (starts with ///)
        """
        if node is None:
            return False

        try:
            comment_text = self.get_node_text(node, source).strip()
            return comment_text.startswith("///")
        except Exception as e:
            logger.error(f"Failed to check XML doc comment: {e}")
            return False

    def clean_comment_text(self, text: str) -> str:
        """Clean C# comment text by removing comment markers and XML doc formatting.

        Args:
            text: Raw comment text

        Returns:
            Cleaned comment text
        """
        cleaned = text.strip()

        # Handle XML doc comments (///)
        if cleaned.startswith("///"):
            lines = cleaned.split("\n")
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith("/// "):
                    line = line[4:]
                elif line.startswith("///"):
                    line = line[3:]

                # Remove common XML tags for readability
                line = line.replace("<summary>", "").replace("</summary>", "")
                line = line.replace("<param name=", "Parameter ")
                line = line.replace("<returns>", "Returns: ").replace("</returns>", "")
                line = line.replace("<remarks>", "").replace("</remarks>", "")
                line = line.replace("<example>", "Example: ").replace("</example>", "")

                if line.strip():
                    cleaned_lines.append(line)

            cleaned = "\n".join(cleaned_lines).strip()
        # Handle regular single-line comments
        elif cleaned.startswith("//"):
            cleaned = cleaned[2:].strip()
        # Handle multi-line comments
        elif cleaned.startswith("/*") and cleaned.endswith("*/"):
            cleaned = cleaned[2:-2].strip()

        return cleaned

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a C# node should be included as a chunk.

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
            if node.type == "comment":
                # Include XML documentation comments
                if self.is_xml_doc_comment(node, source):
                    cleaned_text = self.clean_comment_text(node_text)
                    return len(cleaned_text) > 10  # Meaningful XML doc

                # Skip very short comments
                cleaned_text = self.clean_comment_text(node_text)
                if len(cleaned_text) < 5:
                    return False

                # Skip comments that are just separators
                if cleaned_text.strip("=/-*+_ \t\n") == "":
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate C# node inclusion: {e}")
            return False

    def get_qualified_name(
        self,
        node: "TSNode | None",
        source: str,
        namespace_name: str = "",
        parent_name: str = "",
    ) -> str:
        """Get fully qualified name for a C# symbol.

        Args:
            node: Tree-sitter node
            source: Source code string
            namespace_name: Namespace name
            parent_name: Parent class/interface name

        Returns:
            Fully qualified symbol name
        """
        if node is None:
            return ""

        try:
            name = ""
            if node.type in [
                "class_declaration",
                "interface_declaration",
                "struct_declaration",
                "enum_declaration",
                "record_declaration",
            ]:
                name = self.extract_class_name(node, source)
            elif node.type in [
                "method_declaration",
                "constructor_declaration",
                "destructor_declaration",
            ]:
                name = self.extract_method_name(node, source)
            elif node.type == "property_declaration":
                name_node = self.find_child_by_type(node, "identifier")
                if name_node:
                    name = self.get_node_text(name_node, source).strip()
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
            if namespace_name:
                qualified_name = f"{namespace_name}.{qualified_name}"

            return qualified_name

        except Exception as e:
            logger.error(f"Failed to get C# qualified name: {e}")
            return self.get_fallback_name(node, "symbol")

    def extract_constants(
        self, concept: "Any", captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from C# code.

        Detects:
        - const field declarations (class-level)
        - readonly static fields (class-level)
        - const local declarations (method-level)

        Args:
            concept: The universal concept being processed
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dictionaries with "name", "value", and optionally "type"
            keys, or None
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return None

        source = content.decode("utf-8")
        constants: list[dict[str, str]] = []

        # Only process DEFINITION concept
        if concept != UniversalConcept.DEFINITION:
            return None

        def_node = captures.get("definition")
        if not def_node:
            return None

        # Handle enum declarations
        if def_node.type == "enum_declaration":
            # Extract enum members
            for child in self.walk_tree(def_node):
                if child and child.type == "enum_member_declaration":
                    # Get the identifier (enum member name)
                    name_node = self.find_child_by_type(child, "identifier")
                    if name_node:
                        name = self.get_node_text(name_node, source).strip()
                        constants.append({"name": name, "value": name})

        # Handle field declarations with const or readonly static modifiers
        if def_node.type == "field_declaration":
            # Check for const or (readonly + static) modifiers
            modifiers = self.extract_access_modifiers(def_node, source)
            has_const = "const" in modifiers
            has_readonly = "readonly" in modifiers
            has_static = "static" in modifiers

            if has_const or (has_readonly and has_static):
                # Extract variable declarator
                for child in self.walk_tree(def_node):
                    if child and child.type == "variable_declarator":
                        # Get the identifier (field name)
                        name_node = self.find_child_by_type(child, "identifier")
                        if name_node:
                            name = self.get_node_text(name_node, source).strip()

                            # Get the initializer value
                            value_text = ""
                            for init_child in self.walk_tree(child):
                                if init_child and init_child.type in (
                                    "integer_literal",
                                    "real_literal",
                                    "string_literal",
                                    "character_literal",
                                    "boolean_literal",
                                    "null_literal",
                                ):
                                    value_text = self.get_node_text(
                                        init_child, source
                                    ).strip()
                                    break

                            # Truncate to 50 chars if longer
                            if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                                value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                            # Extract type
                            type_text = ""
                            for type_child in self.walk_tree(def_node):
                                if type_child and type_child.type in (
                                    "predefined_type",
                                    "identifier_name",
                                    "generic_name",
                                    "array_type",
                                    "nullable_type",
                                ):
                                    type_text = self.get_node_text(
                                        type_child, source
                                    ).strip()
                                    break

                            const_info: dict[str, str] = {
                                "name": name,
                                "value": value_text,
                            }
                            if type_text:
                                const_info["type"] = type_text

                            constants.append(const_info)

        # Handle local const declarations (method-level)
        if def_node.type == "local_declaration_statement":
            # Check for const modifier
            modifiers = self.extract_access_modifiers(def_node, source)
            has_const = "const" in modifiers

            if has_const:
                # Extract variable declarator
                for child in self.walk_tree(def_node):
                    if child and child.type == "variable_declarator":
                        # Get the identifier (local const name)
                        name_node = self.find_child_by_type(child, "identifier")
                        if name_node:
                            name = self.get_node_text(name_node, source).strip()

                            # Get the initializer value
                            value_text = ""
                            for init_child in self.walk_tree(child):
                                if init_child and init_child.type in (
                                    "integer_literal",
                                    "real_literal",
                                    "string_literal",
                                    "character_literal",
                                    "boolean_literal",
                                    "null_literal",
                                ):
                                    value_text = self.get_node_text(
                                        init_child, source
                                    ).strip()
                                    break

                            # Truncate to 50 chars if longer
                            if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                                value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                            # Extract type
                            type_text = ""
                            for type_child in self.walk_tree(def_node):
                                if type_child and type_child.type in (
                                    "predefined_type",
                                    "identifier_name",
                                    "generic_name",
                                    "array_type",
                                    "nullable_type",
                                ):
                                    type_text = self.get_node_text(
                                        type_child, source
                                    ).strip()
                                    break

                            const_info: dict[str, str] = {
                                "name": name,
                                "value": value_text,
                            }
                            if type_text:
                                const_info["type"] = type_text

                            constants.append(const_info)

        return constants if constants else None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path for C#.

        C# using directives map to assemblies, not files.

        Args:
            import_text: The import statement text
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            None (C# using directives are namespace-based, not file-based)
        """
        # C# using directives map to assemblies, not files
        return None
