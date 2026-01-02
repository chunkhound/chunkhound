"""Universal concept extraction using language-specific mappings."""

from pathlib import Path
from typing import Any, Protocol

from tree_sitter import Node, QueryCursor

from chunkhound.utils.normalization import normalize_content

from .universal_engine import (
    QueryCompilationError,
    TreeSitterEngine,
    UniversalChunk,
    UniversalConcept,
)


class LanguageMapping(Protocol):
    """Protocol for language-specific tree-sitter mappings."""

    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in this language."""
        ...

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""
        ...

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""
        ...

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract language-specific metadata."""
        ...

    def extract_constants(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> list[dict[str, str]] | None:
        """Extract constants from captures.

        Returns list of constant definitions with structure:
            [{"name": "CONST_NAME", "value": "value_str", "type": "optional_type"}, ...]

        Returns None if no constants found or not applicable for this concept.
        """
        ...

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path,
    ) -> Path | None:
        """Resolve import statement to file path.

        Each language implements its own import resolution logic.
        Returns None for external/unresolvable imports.

        Args:
            import_text: The raw import statement text
            base_dir: Project root directory
            source_file: File containing the import

        Returns:
            Resolved file path or None if external/unresolvable
        """
        ...


class ConceptExtractor:
    """Extracts universal concepts using language-specific mappings."""

    def __init__(self, engine: TreeSitterEngine, mapping: LanguageMapping):
        self.engine = engine
        self.mapping = mapping
        self._compiled_queries: dict[UniversalConcept, Any] = {}
        self._compile_all_queries()

    def _compile_all_queries(self) -> None:
        """Compile all concept queries for this language - fail fast."""
        for concept in UniversalConcept:
            query_template = self.mapping.get_query_for_concept(concept)
            if query_template:
                try:
                    self._compiled_queries[concept] = self.engine.compile_query(
                        query_template
                    )
                except Exception as e:
                    raise QueryCompilationError(
                        concept=concept,
                        language=self.engine.language_name,
                        query=query_template,
                        error=str(e),
                    )

    def extract_concept(
        self, ast_root: Node, content: bytes, concept: UniversalConcept
    ) -> list[UniversalChunk]:
        """Extract all instances of a universal concept."""
        if concept not in self._compiled_queries:
            return []

        query = self._compiled_queries[concept]
        chunks = []

        query_cursor = QueryCursor(query)
        for match in query_cursor.matches(ast_root):
            # match is a tuple: (pattern_index, captures_dict)
            _, captures_dict = match
            # Flatten the captures dict (values are lists, take first element)
            captures = {
                name: nodes[0] for name, nodes in captures_dict.items() if nodes
            }
            chunk = self._build_universal_chunk(concept, captures, content)
            if chunk:  # Some extractions may return None
                chunks.append(chunk)

        return chunks

    def extract_all_concepts(
        self, ast_root: Node, content: bytes
    ) -> list[UniversalChunk]:
        """Extract all universal concepts from the AST."""
        all_chunks = []
        for concept in UniversalConcept:
            concept_chunks = self.extract_concept(ast_root, content, concept)
            all_chunks.extend(concept_chunks)
        return all_chunks

    def _build_universal_chunk(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> UniversalChunk | None:
        """Convert language-specific captures to universal chunk."""
        # Use mapping to extract universal fields from language-specific captures
        name = self.mapping.extract_name(concept, captures, content)
        chunk_content = self.mapping.extract_content(concept, captures, content)
        metadata = self.mapping.extract_metadata(concept, captures, content)

        # Add constants to metadata
        constants = self.mapping.extract_constants(concept, captures, content)
        if constants:
            metadata["constants"] = constants

        # Skip empty or whitespace-only chunks
        normalized_content = normalize_content(chunk_content or "")
        if not normalized_content:
            return None

        # Get position from definition node
        def_node = (
            captures.get("definition")
            or captures.get("node")
            or list(captures.values())[0]
            if captures
            else None
        )
        if not def_node:
            return None

        start_line = def_node.start_point[0] + 1
        end_line = def_node.end_point[0] + 1

        return UniversalChunk(
            concept=concept,
            name=name,
            content=chunk_content,
            start_line=start_line,
            end_line=end_line,
            metadata=metadata,
            language_node_type=def_node.type,
        )
