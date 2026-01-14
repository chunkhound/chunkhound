"""Adapter that bridges BaseMapping to LanguageMapping protocol.

This adapter allows existing BaseMapping implementations to work with
the ConceptExtractor's LanguageMapping protocol by converting traditional
queries (get_function_query, get_class_query, etc.) to universal concepts.
"""

from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.parsers.concept_extractor import LanguageMapping
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class MappingAdapter(LanguageMapping):
    """Adapter that converts BaseMapping to LanguageMapping protocol.

    This adapter maps universal concepts to traditional mapping methods:
    - DEFINITION -> function_query + class_query
    - BLOCK -> method_query (methods are code blocks within classes)
    - COMMENT -> comment_query
    - IMPORT -> (not supported by BaseMapping, returns None)
    - STRUCTURE -> (not supported by BaseMapping, returns None)
    """

    def __init__(self, base_mapping: BaseMapping):
        """Initialize adapter with a BaseMapping instance.

        Args:
            base_mapping: BaseMapping implementation to adapt
        """
        self.base_mapping = base_mapping

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept using BaseMapping methods.

        Args:
            concept: Universal concept to get query for

        Returns:
            Tree-sitter query string or None if concept not supported
        """
        # Check if base_mapping implements LanguageMapping protocol directly
        if hasattr(self.base_mapping, "get_query_for_concept") and callable(
            getattr(self.base_mapping, "get_query_for_concept")
        ):
            return self.base_mapping.get_query_for_concept(concept)

        # Fallback to adapter behavior
        if concept == UniversalConcept.DEFINITION:
            # Combine function, class, and definition queries
            function_query = self.base_mapping.get_function_query()
            class_query = self.base_mapping.get_class_query()

            # Combine both queries if they exist
            queries = []
            if function_query.strip():
                queries.append(function_query.strip())
            if class_query.strip():
                queries.append(class_query.strip())

            # Check for additional definition patterns (e.g., field_declaration for Java)
            if hasattr(self.base_mapping, "get_definition_query"):
                definition_query = self.base_mapping.get_definition_query()
                if definition_query and definition_query.strip():
                    queries.append(definition_query.strip())

            if queries:
                return "\n\n".join(queries)
            else:
                return None

        elif concept == UniversalConcept.BLOCK:
            # Use method query if available, otherwise empty
            method_query = self.base_mapping.get_method_query()
            if method_query.strip():
                return method_query.strip()
            else:
                return None

        elif concept == UniversalConcept.COMMENT:
            # Use comment query
            comment_query = self.base_mapping.get_comment_query()
            if comment_query.strip():
                return comment_query.strip()
            else:
                return None

        elif concept == UniversalConcept.IMPORT:
            # BaseMapping doesn't typically have import queries
            return None

        elif concept == UniversalConcept.STRUCTURE:
            # BaseMapping doesn't typically have structure queries
            return None

        else:
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures using BaseMapping methods.

        Args:
            concept: Universal concept being extracted
            captures: Dictionary of captured nodes from query
            content: Source code as bytes

        Returns:
            Extracted name or fallback name
        """
        # Check if base_mapping implements LanguageMapping protocol directly
        if hasattr(self.base_mapping, "extract_name") and callable(
            getattr(self.base_mapping, "extract_name")
        ):
            return self.base_mapping.extract_name(concept, captures, content)

        # Fallback to adapter behavior
        source = content.decode("utf-8", errors="replace")

        if concept == UniversalConcept.DEFINITION:
            # Try to find the main definition node
            def_node = self._find_definition_node(captures)

            if def_node:
                # Determine if it's a function or class based on captures or node type
                if self._is_function_node(captures, def_node):
                    return self.base_mapping.extract_function_name(def_node, source)
                elif self._is_class_node(captures, def_node):
                    return self.base_mapping.extract_class_name(def_node, source)
                else:
                    # Default to function extraction
                    return self.base_mapping.extract_function_name(def_node, source)
            else:
                return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Try method extraction
            method_node = self._find_definition_node(captures)
            if method_node and hasattr(self.base_mapping, "extract_method_name"):
                return self.base_mapping.extract_method_name(method_node, source)
            elif method_node:
                return self.base_mapping.extract_function_name(method_node, source)
            else:
                return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Use line-based naming for comments
            comment_node = self._find_definition_node(captures)
            if comment_node:
                line = comment_node.start_point[0] + 1
                return f"comment_line_{line}"
            else:
                return "unnamed_comment"

        else:
            return f"unnamed_{concept.value}"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures.

        Args:
            concept: Universal concept being extracted
            captures: Dictionary of captured nodes from query
            content: Source code as bytes

        Returns:
            Extracted content
        """
        # Check if base_mapping implements LanguageMapping protocol directly
        if hasattr(self.base_mapping, "extract_content") and callable(
            getattr(self.base_mapping, "extract_content")
        ):
            return self.base_mapping.extract_content(concept, captures, content)

        # Fallback to adapter behavior
        def_node = self._find_definition_node(captures)
        if def_node:
            source = content.decode("utf-8", errors="replace")
            return self.base_mapping.get_node_text(def_node, source)
        else:
            return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract metadata from captures.

        Args:
            concept: Universal concept being extracted
            captures: Dictionary of captured nodes from query
            content: Source code as bytes

        Returns:
            Dictionary of metadata
        """
        # Check if base_mapping implements LanguageMapping protocol directly
        if hasattr(self.base_mapping, "extract_metadata") and callable(
            getattr(self.base_mapping, "extract_metadata")
        ):
            return self.base_mapping.extract_metadata(concept, captures, content)

        # Fallback to adapter behavior
        def_node = self._find_definition_node(captures)
        metadata = {
            "concept": concept.value,
            "language": self.base_mapping.language.value,
            "adapter_used": True,
            "capture_names": list(captures.keys()),
        }

        if def_node:
            metadata["node_type"] = def_node.type

        # Add concept-specific metadata
        if concept == UniversalConcept.DEFINITION and def_node:
            source = content.decode("utf-8", errors="replace")

            # Try to extract parameters if it's a function
            if hasattr(self.base_mapping, "extract_parameters"):
                try:
                    params = self.base_mapping.extract_parameters(def_node, source)
                    if params:
                        metadata["parameters"] = params
                except:
                    pass  # Ignore extraction errors

        return metadata

    def _find_definition_node(self, captures: dict[str, Node]) -> Node | None:
        """Find the main definition node from captures.

        Args:
            captures: Dictionary of captured nodes

        Returns:
            Main definition node or None
        """
        # Common patterns for definition nodes in queries
        definition_keys = [
            "definition",
            "def",
            "function_def",
            "function.def",
            "async_function_def",
            "async_function.def",
            "class_def",
            "class.def",
            "method_def",
            "method.def",
            "async_method_def",
            "async_method.def",
            "comment",
        ]

        # File-level and large container nodes that should never be returned as definitions
        # These would create oversized chunks containing entire files or large code sections
        FILE_LEVEL_NODES = {
            # File-level nodes
            "module",
            "source_file",
            "program",
            "translation_unit",
            "compilation_unit",
            # Large container nodes that can contain massive amounts of code
            "class_body",
            "struct_body",
            "interface_body",
            "namespace_body",
            "enum_body",
            "field_declaration_list",
            "declaration_list",
            "member_declaration_list",
            "compound_statement",  # When it's a large top-level block
        }

        # Try to find a definition node
        for key in definition_keys:
            if key in captures:
                return captures[key]

        # Fallback: return first non-file-level capture
        for node in captures.values():
            if node.type not in FILE_LEVEL_NODES:
                return node

        # If only file-level captures found, return None to skip this concept
        # This prevents extracting entire files as single chunks
        return None

    def _is_function_node(self, captures: dict[str, Node], node: Node) -> bool:
        """Check if a node represents a function definition.

        Args:
            captures: Dictionary of captured nodes
            node: Node to check

        Returns:
            True if node is a function definition
        """
        # Check capture names for function indicators
        function_indicators = [
            "function",
            "function_def",
            "function.def",
            "async_function",
            "async_function_def",
            "async_function.def",
            "function_name",
            "function.name",
            "async_function_name",
            "async_function.name",
        ]

        for indicator in function_indicators:
            if indicator in captures:
                return True

        # Check node type
        if node.type in ("function_definition", "async_function_definition"):
            return True

        return False

    def _is_class_node(self, captures: dict[str, Node], node: Node) -> bool:
        """Check if a node represents a class definition.

        Args:
            captures: Dictionary of captured nodes
            node: Node to check

        Returns:
            True if node is a class definition
        """
        # Check capture names for class indicators
        class_indicators = [
            "class",
            "class_def",
            "class.def",
            "class_name",
            "class.name",
        ]

        for indicator in class_indicators:
            if indicator in captures:
                return True

        # Check node type
        if node.type in ("class_definition", "class_declaration"):
            return True

        return False

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constants from captures by delegating to base_mapping.

        Args:
            concept: Universal concept being extracted
            captures: Dictionary of captured nodes from query
            content: Source code as bytes

        Returns:
            List of constant dictionaries or None
        """
        if hasattr(self.base_mapping, "extract_constants"):
            return self.base_mapping.extract_constants(concept, captures, content)
        return None

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve import statement to file path by delegating to base mapping.

        Args:
            import_text: The import statement text to resolve
            base_dir: Base directory for resolution
            source_file: Path to the source file containing the import

        Returns:
            Resolved path or None if unresolvable
        """
        return self.base_mapping.resolve_import_path(
            import_text, base_dir, source_file
        )
