"""Objective-C language mapping for unified parser architecture.

This module provides Objective-C-specific tree-sitter queries and extraction logic
for the unified parser system. It handles Objective-C's unique features including:
- @interface / @implementation (class definitions)
- Instance methods (-) and class methods (+)
- @property declarations
- @protocol definitions
- Categories and extensions
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from chunkhound.core.types.common import Language

from .base import MAX_CONSTANT_VALUE_LENGTH, BaseMapping

if TYPE_CHECKING:
    from chunkhound.parsers.universal_engine import UniversalConcept

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class ObjCMapping(BaseMapping):
    """Objective-C-specific tree-sitter mapping for semantic code extraction."""

    def __init__(self) -> None:
        """Initialize Objective-C mapping."""
        super().__init__(Language.OBJC)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Objective-C method definitions.

        Captures both instance methods (-) and class methods (+).

        Returns:
            Tree-sitter query string for finding method definitions
        """
        return """
        (method_definition) @method_def
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Objective-C class definitions.

        Captures @interface, @implementation, and protocol declarations.

        Returns:
            Tree-sitter query string for finding class definitions
        """
        return """
        (class_interface) @class_def

        (class_implementation) @class_def

        (protocol_declaration) @class_def
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Objective-C comments.

        Returns:
            Tree-sitter query string for finding comments
        """
        return """
        (comment) @comment
        """

    def get_property_query(self) -> str:
        """Get tree-sitter query pattern for @property declarations.

        Returns:
            Tree-sitter query string for finding property declarations
        """
        return """
        (property_declaration) @property_def
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract method name from an Objective-C method definition node.

        Handles both instance methods (-) and class methods (+).

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            Method name with scope prefix (e.g., "-setName:age:" or "+sharedInstance")
        """
        if node is None:
            return self.get_fallback_name(node, "method")

        try:
            # Extract scope (- or +) from direct child
            scope = ""
            for child in node.children:
                if child.type in ("-", "+"):
                    scope = self.get_node_text(child, source).strip()
                    break

            # Collect identifiers and count parameters
            identifiers = []
            param_count = 0
            for child in node.children:
                if child.type == "identifier":
                    identifiers.append(self.get_node_text(child, source).strip())
                elif child.type == "method_parameter":
                    param_count += 1
                elif child.type == "compound_statement":
                    break

            # Build selector
            if param_count > 0 and identifiers:
                selector = ":".join(identifiers) + ":"
            elif identifiers:
                selector = identifiers[0]
            else:
                return self.get_fallback_name(node, "method")

            return f"{scope}{selector}"

        except Exception as e:
            logger.error(f"Failed to extract Objective-C method name: {e}")

        return self.get_fallback_name(node, "method")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract class name from an Objective-C class definition node.

        Handles @interface, @implementation, categories, and protocols.

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
                class_name = self.get_node_text(name_node, source).strip()

                # Check if this is a category
                if node.type in ("category_interface", "category_implementation"):
                    # Find category name (second identifier)
                    identifiers = self.find_children_by_type(node, "identifier")
                    if len(identifiers) >= 2:
                        category_name = self.get_node_text(
                            identifiers[1], source
                        ).strip()
                        return f"{class_name}+{category_name}"

                # Check if this is a protocol
                if node.type == "protocol_declaration":
                    return f"@protocol {class_name}"

                return class_name

            # Fallback: search through children
            for child in self.walk_tree(node):
                if child and child.type == "identifier":
                    return self.get_node_text(child, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract Objective-C class name: {e}")

        return self.get_fallback_name(node, "class")

    def extract_parameters(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract parameter names from an Objective-C method node.

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            List of parameter names (e.g., ["name", "age"])
        """
        if node is None:
            return []

        parameters: list[str] = []

        try:
            # Each method_parameter node has structure: ":" + method_type + identifier
            # We extract the identifier (child 2) from each method_parameter
            for child in node.children:
                if child.type == "method_parameter":
                    # Get the identifier child (index 2)
                    if child.child_count >= 3:
                        param_identifier = child.child(2)
                        if param_identifier and param_identifier.type == "identifier":
                            param_name = self.get_node_text(
                                param_identifier, source
                            ).strip()
                            if param_name:
                                parameters.append(param_name)

        except Exception as e:
            logger.error(f"Failed to extract Objective-C parameters: {e}")

        return parameters

    def extract_property_attributes(
        self, node: "TSNode | None", source: str
    ) -> list[str]:
        """Extract property attributes from an @property declaration.

        Attributes include: nonatomic, strong, weak, copy, readonly, etc.

        Args:
            node: Tree-sitter property declaration node
            source: Source code string

        Returns:
            List of property attributes
        """
        if node is None:
            return []

        attributes: list[str] = []

        try:
            # Find property_attributes node
            attrs_node = self.find_child_by_type(node, "property_attributes")
            if attrs_node:
                attr_text = self.get_node_text(attrs_node, source).strip()
                # Remove parentheses and split by comma
                attr_text = attr_text.strip("()")
                if attr_text:
                    # Split by comma and clean up each attribute
                    attributes = [attr.strip() for attr in attr_text.split(",")]

        except Exception as e:
            logger.error(f"Failed to extract property attributes: {e}")

        return attributes

    def extract_superclass(self, node: "TSNode | None", source: str) -> str | None:
        """Extract superclass name from an Objective-C class interface.

        Args:
            node: Tree-sitter class interface node
            source: Source code string

        Returns:
            Superclass name or None if no superclass specified
        """
        if node is None or node.type != "class_interface":
            return None

        try:
            # Look for superclass specification (: ClassName)
            superclass_node = self.find_child_by_type(node, "superclass")
            if superclass_node:
                # Get the identifier from the superclass node
                identifier = self.find_child_by_type(superclass_node, "identifier")
                if identifier:
                    return self.get_node_text(identifier, source).strip()

        except Exception as e:
            logger.error(f"Failed to extract superclass: {e}")

        return None

    def extract_protocols(self, node: "TSNode | None", source: str) -> list[str]:
        """Extract protocol names that a class conforms to.

        Args:
            node: Tree-sitter class interface node
            source: Source code string

        Returns:
            List of protocol names
        """
        if node is None:
            return []

        protocols: list[str] = []

        try:
            # Look for protocol_qualifiers
            protocol_quals = self.find_child_by_type(node, "protocol_qualifiers")
            if protocol_quals:
                # Extract identifiers from protocol qualifiers
                for identifier in self.find_children_by_type(
                    protocol_quals, "identifier"
                ):
                    protocol_name = self.get_node_text(identifier, source).strip()
                    if protocol_name:
                        protocols.append(protocol_name)

        except Exception as e:
            logger.error(f"Failed to extract protocols: {e}")

        return protocols

    def is_class_method(self, node: "TSNode | None", source: str) -> bool:
        """Check if a method is a class method (+ prefix) vs instance method (- prefix).

        Args:
            node: Tree-sitter method definition node
            source: Source code string

        Returns:
            True if class method, False if instance method
        """
        if node is None:
            return False

        try:
            scope_node = self.find_child_by_type(node, "method_scope")
            if scope_node:
                scope = self.get_node_text(scope_node, source).strip()
                return scope == "+"

        except Exception:
            pass

        return False

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if an Objective-C node should be included as a chunk.

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

        # Skip very small nodes to filter out trivial/empty declarations
        # 20 characters filters stub methods, forward declarations, empty blocks
        # while preserving real implementations (e.g., "- (void)method { }")
        if len(text.strip()) < 20:
            return False

        # For methods, check if they have actual body content
        if node.type == "method_definition":
            # Look for actual implementation (more than just the signature)
            lines = text.strip().split("\n")
            if len(lines) <= 2:  # Just declaration line
                return False

        return True

    # UniversalConcept interface methods

    def get_query_for_concept(self, concept: "UniversalConcept") -> str | None:
        """Get tree-sitter query for universal concept in Objective-C.

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
            return """
            (class_interface) @definition

            (protocol_declaration) @definition

            (method_definition) @definition

            (property_declaration) @definition

            ; Capture #define macros at any scope
            ; Note: Tree-sitter queries search entire tree by default (top-level, function-scoped, nested blocks)
            (preproc_def) @definition

            ; Capture const declarations at any scope
            ; This pattern matches declarations at ALL nesting levels (top-level, methods, nested blocks, etc.)
            (declaration
                (type_qualifier)
            ) @definition
            """

        elif concept == UniversalConcept.COMMENT:
            return """
            (comment) @definition
            """

        # IMPORT not supported by tree-sitter-objc grammar
        # elif concept == UniversalConcept.IMPORT:
        #     return """
        #     (preproc_import) @definition
        #     """

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
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return "unnamed"

        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            if "definition" in captures:
                def_node = captures["definition"]

                # For method definitions, extract full selector with scope
                if def_node.type == "method_definition":
                    name = self.extract_function_name(def_node, source)
                    return name

                # For class interface/implementation
                elif def_node.type in ("class_interface", "class_implementation"):
                    name = self.extract_class_name(def_node, source)
                    return name

                # For protocol declarations
                elif def_node.type == "protocol_declaration":
                    name = self.extract_class_name(def_node, source)
                    # Add @protocol prefix if not already present
                    if not name.startswith("@protocol"):
                        return f"@protocol {name}"
                    return name

                # For property declarations
                elif def_node.type == "property_declaration":
                    # Find property name identifier
                    name_node = self.find_child_by_type(def_node, "identifier")
                    if name_node:
                        return self.get_node_text(name_node, source).strip()

            return "unnamed_definition"

        elif concept == UniversalConcept.COMMENT:
            if "definition" in captures:
                node = captures["definition"]
                line = node.start_point[0] + 1
                return f"comment_line_{line}"
            return "unnamed_comment"

        # IMPORT concept not supported (get_query_for_concept returns None)

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
        """Extract Objective-C-specific metadata.

        Args:
            concept: The universal concept being extracted
            captures: Dictionary of captured nodes from tree-sitter query
            content: Source code as bytes

        Returns:
            Dictionary of metadata
        """
        try:
            from chunkhound.parsers.universal_engine import UniversalConcept
        except ImportError:
            return {}

        source = content.decode("utf-8")
        metadata: dict[str, Any] = {}

        if concept == UniversalConcept.DEFINITION:
            def_node = captures.get("definition")
            if def_node:
                metadata["node_type"] = def_node.type

                # For method definitions
                if def_node.type == "method_definition":
                    metadata["kind"] = "method"

                    # Extract method scope (instance - or class +)
                    scope_node = self.find_child_by_type(def_node, "method_scope")
                    if scope_node:
                        scope = self.get_node_text(scope_node, source).strip()
                        metadata["scope"] = "class" if scope == "+" else "instance"

                    # Extract parameters
                    selector_node = self.find_child_by_type(def_node, "method_selector")
                    if selector_node:
                        params = []
                        for child in selector_node.children:
                            if child and child.type == "identifier":
                                param_name = self.get_node_text(child, source).strip()
                                if param_name:
                                    params.append(param_name)
                        if params:
                            metadata["parameters"] = params

                # For class interface
                elif def_node.type == "class_interface":
                    metadata["kind"] = "class"
                    metadata["declaration_type"] = "interface"

                    # Extract superclass
                    superclass = self.extract_superclass(def_node, source)
                    if superclass:
                        metadata["superclass"] = superclass

                    # Extract protocols
                    protocols = self.extract_protocols(def_node, source)
                    if protocols:
                        metadata["protocols"] = protocols

                # For class implementation
                elif def_node.type == "class_implementation":
                    metadata["kind"] = "class"
                    metadata["declaration_type"] = "implementation"

                # For protocol declaration
                elif def_node.type == "protocol_declaration":
                    # Protocols map to INTERFACE chunk type
                    metadata["kind"] = "interface"
                    metadata["protocol"] = True  # Mark as protocol

                    # Extract adopted protocols
                    protocols = self.extract_protocols(def_node, source)
                    if protocols:
                        metadata["adopted_protocols"] = protocols

                # For property declaration
                elif def_node.type == "property_declaration":
                    metadata["kind"] = "property"

                    # Extract property attributes
                    attributes = self.extract_property_attributes(def_node, source)
                    if attributes:
                        metadata["attributes"] = attributes

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

        # IMPORT concept not supported (get_query_for_concept returns None)

        return metadata

    def extract_constants(
        self, concept: "UniversalConcept", captures: dict[str, "TSNode"], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constant definitions from Objective-C code.

        Identifies #define macros and const declarations as constants.

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

        # Get the definition node
        def_node = captures.get("definition")
        if not def_node:
            return None

        source = content.decode("utf-8")

        # Handle #define macros (preprocessor directives)
        if def_node.type == "preproc_def":
            # Extract #define NAME VALUE
            node_text = self.get_node_text(def_node, source).strip()
            match = re.match(r'#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)', node_text)
            if not match:
                return None

            name = match.group(1)
            value = match.group(2).strip()

            # Truncate long values
            if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."

            return [{"name": name, "value": value}]

        # Handle const declarations
        if def_node.type == "declaration":
            # Look for 'const' keyword in type specifier
            has_const = False
            for child in self.walk_tree(def_node):
                if child and child.type == "type_qualifier":
                    qualifier_text = self.get_node_text(child, source).strip()
                    if qualifier_text == "const":
                        has_const = True
                        break

            if not has_const:
                return None

            # Extract variable name and value
            constants = []
            for child in self.walk_tree(def_node):
                if child and child.type == "init_declarator":
                    # Find identifier and value
                    name = ""
                    value = ""

                    for subchild in child.children:
                        if subchild.type == "identifier":
                            name = self.get_node_text(subchild, source).strip()
                        elif subchild.type not in ["identifier", "="]:
                            # This is the initializer
                            value = self.get_node_text(subchild, source).strip()

                    if name:
                        # Truncate long values
                        if len(value) > MAX_CONSTANT_VALUE_LENGTH:
                            value = value[:MAX_CONSTANT_VALUE_LENGTH] + "..."
                        constants.append({"name": name, "value": value})

            return constants if constants else None

        return None

    def resolve_import_path(
        self, import_text: str, base_dir: Path, source_file: Path
    ) -> Path | None:
        """Resolve Objective-C include to file path.

        Args:
            import_text: The import statement text (e.g., '#include "file.h"' or '#import "file.h"')
            base_dir: Base directory of the codebase
            source_file: Path to the file containing the import

        Returns:
            Resolved absolute path if found, None otherwise (system includes)
        """
        # Local includes: #include "file.h" or #import "file.h"
        local_match = re.search(r'#(?:include|import)\s*"(.+?)"', import_text)
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

        # System includes (#include <...> or #import <...>) - external, return None
        return None
