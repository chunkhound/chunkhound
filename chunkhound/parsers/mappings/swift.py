"""Swift language mapping for unified parser architecture.

This module provides Swift-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Swift's unique features including:
- Classes and structs (value/reference types)
- Protocols (Swift's interface mechanism)
- Extensions (type augmentation)
- Actors (modern concurrency primitives)
- Property wrappers (@State, @Published, etc.)
- Functions and methods
- Enums with associated values
"""

from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.types.common import Language
from chunkhound.parsers.universal_engine import UniversalConcept

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class SwiftMapping(BaseMapping):
    """Swift-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize Swift mapping."""
        super().__init__(Language.SWIFT)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Swift function definitions.

        Captures functions, methods, initializers, and computed properties.

        Returns:
            Tree-sitter query string for finding function definitions
        """
        return """
        (function_declaration) @function_def

        (init_declaration) @function_def

        (deinit_declaration) @function_def

        (subscript_declaration) @function_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Swift class definitions.

        Captures classes, structs, protocols, enums, actors, and extensions.
        Note: Swift tree-sitter uses class_declaration for all type declarations
        except protocols. The actual type is determined by keyword child nodes.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_declaration) @class_def

        (protocol_declaration) @protocol_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Swift comments.

        Captures both line comments (//) and block comments (/* */).

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (comment) @comment

        (multiline_comment) @comment
        """

    def get_extension_query(self) -> str:
        """Get tree-sitter query pattern for Swift extensions.

        Extensions should be separate chunks per user requirements.

        Returns:
            Tree-sitter query string for finding extension declarations
        """
        return """
        (extension_declaration) @extension_def
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract function name from a Swift function definition node.

        Handles functions, methods, initializers, and deinit.

        Args:
            node: Tree-sitter function definition node
            source: Source code string

        Returns:
            Function name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "function")

        try:
            # Handle initializers
            if node.type == "init_declaration":
                # Check for failable initializer
                if self.find_child_by_type(node, "?"):
                    return "init?"
                return "init"

            # Handle deinitializers
            if node.type == "deinit_declaration":
                return "deinit"

            # Handle subscripts
            if node.type == "subscript_declaration":
                # Extract parameter types for subscript signature
                params = self.extract_parameter_types(node, source)
                if params:
                    return f"subscript({', '.join(params)})"
                return "subscript"

            # Handle regular functions and methods
            name_node = self.find_child_by_type(node, "simple_identifier")
            if name_node:
                func_name = self.get_node_text(name_node, source).strip()

                # Check for generic parameters
                generic_params = self.extract_generic_parameters(node, source)
                if generic_params:
                    func_name += f"<{', '.join(generic_params)}>"

                return func_name

        except Exception as e:
            logger.error(f"Failed to extract Swift function name: {e}")

        return self.get_fallback_name(node, "function")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract type name from a Swift type definition node.

        Handles classes, structs, protocols, enums, and actors.
        Note: Swift tree-sitter uses class_declaration for all except protocols.

        Args:
            node: Tree-sitter type definition node
            source: Source code string

        Returns:
            Type name or fallback name if extraction fails
        """
        if node is None:
            return self.get_fallback_name(node, "type")

        try:
            # Find type name identifier
            name_node = self.find_child_by_type(node, "type_identifier")
            if name_node:
                type_name = self.get_node_text(name_node, source).strip()

                # Add prefix for protocols
                if node.type == "protocol_declaration":
                    type_name = f"protocol {type_name}"
                # For class_declaration, check the keyword child to determine actual type
                elif node.type == "class_declaration":
                    # Only add prefix for actors (to distinguish from classes)
                    # Other types (struct, enum, class) don't get prefixes in symbol names
                    for child in node.children:
                        if child.type == "actor":
                            type_name = f"actor {type_name}"
                            break

                # Check for generic parameters
                generic_params = self.extract_generic_parameters(node, source)
                if generic_params:
                    type_name += f"<{', '.join(generic_params)}>"

                return type_name

        except Exception as e:
            logger.error(f"Failed to extract Swift type name: {e}")

        return self.get_fallback_name(node, "type")

    def extract_extension_name(self, node: "TSNode | None", source: str) -> str:
        """Extract extended type name from a Swift extension node.

        Args:
            node: Tree-sitter extension declaration node
            source: Source code string

        Returns:
            Extended type name (e.g., "extension Array<Element>")
        """
        if node is None:
            return self.get_fallback_name(node, "extension")

        try:
            # Find the type being extended
            type_node = self.find_child_by_type(node, "user_type")
            if not type_node:
                type_node = self.find_child_by_type(node, "type_identifier")

            if type_node:
                type_name = self.get_node_text(type_node, source).strip()

                # Check for protocol conformances in extension
                protocols = self.extract_extension_protocols(node, source)
                if protocols:
                    return f"extension {type_name}: {', '.join(protocols)}"

                return f"extension {type_name}"

        except Exception as e:
            logger.error(f"Failed to extract Swift extension name: {e}")

        return self.get_fallback_name(node, "extension")

    def extract_generic_parameters(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract generic parameter names from a Swift type or function.

        Args:
            node: Tree-sitter node (function or type declaration)
            source: Source code string

        Returns:
            List of generic parameter names (e.g., ["T", "U"])
        """
        if node is None:
            return []

        generic_params: list[str] = []

        try:
            # Find type_parameters node
            type_params_node = self.find_child_by_type(node, "type_parameters")
            if type_params_node:
                # Extract type_identifier children
                for identifier in self.find_children_by_type(
                    type_params_node, "type_identifier"
                ):
                    param_name = self.get_node_text(identifier, source).strip()
                    if param_name:
                        generic_params.append(param_name)

        except Exception as e:
            logger.error(f"Failed to extract generic parameters: {e}")

        return generic_params

    def extract_parameter_types(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter types from a function or subscript.

        Args:
            node: Tree-sitter function/subscript node
            source: Source code string

        Returns:
            List of parameter type names
        """
        if node is None:
            return []

        param_types: list[str] = []

        try:
            # Find parameter list
            params_node = self.find_child_by_type(node, "parameter")
            if params_node:
                # Extract type annotations
                for param in self.find_children_by_type(params_node, "parameter"):
                    type_node = self.find_child_by_type(param, "type_annotation")
                    if type_node:
                        type_text = self.get_node_text(type_node, source).strip()
                        # Remove leading colon
                        if type_text.startswith(":"):
                            type_text = type_text[1:].strip()
                        if type_text:
                            param_types.append(type_text)

        except Exception as e:
            logger.error(f"Failed to extract parameter types: {e}")

        return param_types

    def extract_extension_protocols(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract protocol names from an extension's conformance list.

        Args:
            node: Tree-sitter extension declaration node
            source: Source code string

        Returns:
            List of protocol names
        """
        if node is None:
            return []

        protocols: list[str] = []

        try:
            # Find type_inheritance_clause
            inheritance_node = self.find_child_by_type(node, "type_inheritance_clause")
            if inheritance_node:
                # Extract type identifiers
                for identifier in self.find_children_by_type(
                    inheritance_node, "type_identifier"
                ):
                    protocol_name = self.get_node_text(identifier, source).strip()
                    if protocol_name:
                        protocols.append(protocol_name)

        except Exception as e:
            logger.error(f"Failed to extract extension protocols: {e}")

        return protocols

    def extract_inherited_types(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract inherited types (superclasses and protocols) from a type declaration.

        Args:
            node: Tree-sitter type declaration node
            source: Source code string

        Returns:
            List of inherited type names
        """
        if node is None:
            return []

        inherited: list[str] = []

        try:
            # Find type_inheritance_clause
            inheritance_node = self.find_child_by_type(node, "type_inheritance_clause")
            if inheritance_node:
                # Extract all type identifiers
                for identifier in self.find_children_by_type(
                    inheritance_node, "type_identifier"
                ):
                    type_name = self.get_node_text(identifier, source).strip()
                    if type_name:
                        inherited.append(type_name)

        except Exception as e:
            logger.error(f"Failed to extract inherited types: {e}")

        return inherited

    def extract_access_modifier(self, node: "TSNode | None", source: str) -> str | None:
        """Extract access modifier from a Swift declaration.

        Args:
            node: Tree-sitter declaration node
            source: Source code string

        Returns:
            Access modifier string (public, private, internal, fileprivate, open) or None
        """
        if node is None:
            return None

        try:
            # Find modifiers node
            modifiers_node = self.find_child_by_type(node, "modifiers")
            if modifiers_node:
                modifier_text = self.get_node_text(modifiers_node, source).strip()

                # Check for access modifiers
                access_modifiers = ["open", "public", "internal", "fileprivate", "private"]
                for modifier in access_modifiers:
                    if modifier in modifier_text:
                        return modifier

        except Exception as e:
            logger.error(f"Failed to extract access modifier: {e}")

        return None

    def is_async_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function is async.

        Args:
            node: Tree-sitter function declaration node
            source: Source code string

        Returns:
            True if function is async, False otherwise
        """
        if node is None:
            return False

        try:
            # Find function_modifiers or check for "async" keyword
            modifiers_node = self.find_child_by_type(node, "function_modifiers")
            if modifiers_node:
                modifier_text = self.get_node_text(modifiers_node, source).strip()
                return "async" in modifier_text

        except Exception:
            pass

        return False

    def is_throwing_function(self, node: "TSNode | None", source: str) -> bool:
        """Check if a function can throw errors.

        Args:
            node: Tree-sitter function declaration node
            source: Source code string

        Returns:
            True if function can throw, False otherwise
        """
        if node is None:
            return False

        try:
            # Check for "throws" or "rethrows" keywords
            node_text = self.get_node_text(node, source)
            return "throws" in node_text or "rethrows" in node_text

        except Exception:
            pass

        return False

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a Swift node should be included as a chunk.

        Filters out very small nodes and empty definitions.

        Args:
            node: Tree-sitter node
            source: Source code string

        Returns:
            True if node should be included, False otherwise
        """
        if node is None:
            return False

        # Get the node text to check size
        text = self.get_node_text(node, source)

        # Skip very small nodes (e.g., empty protocols, forward declarations)
        if len(text.strip()) < 20:
            return False

        # For functions/methods, ensure they have actual implementation
        if node.type in ("function_declaration", "init_declaration"):
            # Check if there's a function body
            body_node = self.find_child_by_type(node, "function_body")
            if not body_node:
                # No body means it's just a protocol requirement or declaration
                return False

        return True

    # UniversalConcept interface methods

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for universal concept in Swift.

        Args:
            concept: The universal concept to query

        Returns:
            Tree-sitter query string or None if concept not supported
        """
        if concept == UniversalConcept.DEFINITION:
            return """
            (class_declaration) @definition

            (protocol_declaration) @definition

            (function_declaration) @definition

            (init_declaration) @definition

            (deinit_declaration) @definition

            (property_declaration) @definition

            (typealias_declaration) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition

            (multiline_comment) @definition
            """

        elif concept == UniversalConcept.IMPORT:
            return """
            (import_declaration) @definition
            """

        return None

    def extract_name(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of captured nodes from tree-sitter query
            content: Source code as bytes

        Returns:
            Extracted name string
        """
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            if "definition" in captures:
                def_node = captures["definition"]

                # Functions, initializers, deinitializers
                if def_node.type in (
                    "function_declaration",
                    "init_declaration",
                    "deinit_declaration",
                ):
                    return self.extract_function_name(def_node, source)

                # Classes, structs, protocols, enums, actors, extensions
                # Note: class_declaration is used for classes, structs, enums, actors, and extensions
                elif def_node.type in (
                    "class_declaration",
                    "protocol_declaration",
                ):
                    # Check if it's an extension (uses class_declaration with extension keyword child)
                    if def_node.type == "class_declaration":
                        for child in def_node.children:
                            if child.type == "extension":
                                return self.extract_extension_name(def_node, source)
                    return self.extract_class_name(def_node, source)

                # Properties
                elif def_node.type == "property_declaration":
                    name_node = self.find_child_by_type(def_node, "pattern_binding")
                    if name_node:
                        identifier = self.find_child_by_type(
                            name_node, "simple_identifier"
                        )
                        if identifier:
                            return self.get_node_text(identifier, source).strip()

                # Type aliases
                elif def_node.type == "typealias_declaration":
                    name_node = self.find_child_by_type(def_node, "type_identifier")
                    if name_node:
                        return self.get_node_text(name_node, source).strip()

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
                # Extract imported module name
                node_text = self.get_node_text(node, source).strip()
                # Remove "import " prefix
                if node_text.startswith("import"):
                    module_name = node_text[6:].strip()
                    return f"import {module_name}"
            return "unnamed_import"

        return "unnamed"

    def extract_content(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract content from captures for this concept.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of captured nodes from tree-sitter query
            content: Source code as bytes

        Returns:
            Extracted content string
        """
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
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract Swift-specific metadata.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of captured nodes from tree-sitter query
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

                # Extract access modifier for all definitions
                access = self.extract_access_modifier(def_node, source)
                if access:
                    metadata["access"] = access

                # For function declarations
                if def_node.type == "function_declaration":
                    metadata["kind"] = "function"

                    # Check for async/await
                    if self.is_async_function(def_node, source):
                        metadata["async"] = True

                    # Check for throws
                    if self.is_throwing_function(def_node, source):
                        metadata["throws"] = True

                    # Extract generic parameters
                    generic_params = self.extract_generic_parameters(def_node, source)
                    if generic_params:
                        metadata["generic_parameters"] = generic_params

                # For initializers
                elif def_node.type == "init_declaration":
                    metadata["kind"] = "initializer"

                    # Check for failable initializer
                    if self.find_child_by_type(def_node, "?"):
                        metadata["failable"] = True

                # For class_declaration (handles classes, structs, enums, actors, extensions)
                elif def_node.type == "class_declaration":
                    # Determine actual kind by checking keyword child
                    declaration_kind = "class"  # default
                    for child in def_node.children:
                        if child.type == "struct":
                            declaration_kind = "struct"
                            metadata["value_type"] = True  # Structs are value types
                            break
                        elif child.type == "enum":
                            declaration_kind = "enum"
                            break
                        elif child.type == "actor":
                            declaration_kind = "actor"
                            metadata["concurrency"] = True  # Mark as concurrency-related
                            break
                        elif child.type == "extension":
                            declaration_kind = "extension"
                            break

                    metadata["kind"] = declaration_kind

                    # Extract inherited types/protocols (or protocols for extensions)
                    if declaration_kind == "extension":
                        protocols = self.extract_extension_protocols(def_node, source)
                        if protocols:
                            metadata["adds_conformance"] = protocols
                    elif declaration_kind == "class":
                        inherited = self.extract_inherited_types(def_node, source)
                        if inherited:
                            metadata["inherits"] = inherited
                    else:
                        # For structs, enums, actors
                        inherited = self.extract_inherited_types(def_node, source)
                        if inherited:
                            metadata["conforms_to"] = inherited

                    # Extract generic parameters
                    generic_params = self.extract_generic_parameters(def_node, source)
                    if generic_params:
                        metadata["generic_parameters"] = generic_params

                # For protocols
                elif def_node.type == "protocol_declaration":
                    metadata["kind"] = "protocol"

                    # Extract protocol inheritance
                    inherited = self.extract_inherited_types(def_node, source)
                    if inherited:
                        metadata["inherits"] = inherited

                # For properties
                elif def_node.type == "property_declaration":
                    metadata["kind"] = "property"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                comment_node = captures["definition"]
                comment_text = self.get_node_text(comment_node, source)

                # Determine comment type
                if comment_text.startswith("///"):
                    metadata["comment_type"] = "doc"
                    metadata["is_doc_comment"] = True
                elif comment_text.startswith("//"):
                    metadata["comment_type"] = "line"
                elif comment_text.startswith("/*") and comment_text.endswith("*/"):
                    metadata["comment_type"] = "block"

        elif concept == UniversalConcept.IMPORT:
            if "definition" in captures:
                import_node = captures["definition"]
                import_text = self.get_node_text(import_node, source).strip()

                # Extract module name
                if import_text.startswith("import"):
                    module_name = import_text[6:].strip()
                    metadata["module"] = module_name

        return metadata

    def extract_constants(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Swift code.

        Detects:
        - let declarations at module/type level
        - static let properties

        Args:
            concept: The universal concept being processed
            captures: Tree-sitter query captures
            content: Source file content as bytes

        Returns:
            List of dictionaries with "name", "value", and optionally "type" keys, or None
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

        # Handle property declarations
        if def_node.type == "property_declaration":
            # Check if it's a let declaration (immutable) using AST node types
            is_let = False

            # Check for let keyword using AST structure (not text matching)
            # Swift AST: property_declaration -> value_binding_pattern -> let|var
            value_binding = self.find_child_by_type(def_node, "value_binding_pattern")
            if value_binding:
                let_node = self.find_child_by_type(value_binding, "let")
                if let_node:
                    is_let = True

            # Extract all let bindings (Swift let is immutable)
            # Includes module-level, static, and instance properties
            if is_let:
                # Extract from pattern node (contains property name)
                pattern_node = self.find_child_by_type(def_node, "pattern")
                if pattern_node:
                    # Get the identifier (property name)
                    identifier = self.find_child_by_type(pattern_node, "simple_identifier")
                    if identifier:
                        name = self.get_node_text(identifier, source).strip()

                        # Get the value from property_declaration's children
                        value_text = ""
                        for value_child in self.walk_tree(def_node):
                            if value_child and value_child.type in (
                                "integer_literal",
                                "real_literal",
                                "string_literal",
                                "boolean_literal",
                                "nil",
                            ):
                                value_text = self.get_node_text(value_child, source).strip()
                                break

                        # Truncate to 50 chars if longer
                        if len(value_text) > MAX_CONSTANT_VALUE_LENGTH:
                            value_text = value_text[:MAX_CONSTANT_VALUE_LENGTH]

                        # Extract type annotation from property_declaration
                        type_text = ""
                        type_annotation = self.find_child_by_type(
                            def_node, "type_annotation"
                        )
                        if type_annotation:
                            type_text = self.get_node_text(type_annotation, source).strip()
                            # Remove leading colon
                            if type_text.startswith(":"):
                                type_text = type_text[1:].strip()

                        const_info: dict[str, str] = {"name": name, "value": value_text}
                        if type_text:
                            const_info["type"] = type_text

                        constants.append(const_info)

        return constants if constants else None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve import path for Swift.

        Swift imports are typically framework imports, not file paths.

        Args:
            import_text: The import statement text
            base_dir: Base directory of the project
            source_file: Path to the file containing the import

        Returns:
            None (Swift imports are framework-based, not file-based)
        """
        # Swift imports are typically framework imports, return None
        return None
