"""Markdown language mapping for unified parser architecture.

This module provides Markdown-specific tree-sitter queries and extraction logic
for mapping Markdown AST nodes to semantic chunks. Supports CommonMark syntax
and GitHub Flavored Markdown extensions.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class MarkdownMapping(BaseMapping):
    """Markdown-specific tree-sitter mapping implementation.

    Handles Markdown language features including:
    - ATX and Setext style headers
    - Fenced and indented code blocks
    - Ordered and unordered lists
    - Paragraphs and text content
    - Links and images
    - Blockquotes
    - Tables (GitHub Flavored Markdown)
    - HTML blocks
    """

    def __init__(self) -> None:
        """Initialize Markdown mapping."""
        super().__init__(Language.MARKDOWN)

    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for Markdown code blocks.

        In Markdown, code blocks are the closest equivalent to functions.
        Captures both fenced and indented code blocks.

        Returns:
            Tree-sitter query string for finding Markdown code blocks
        """
        return """
            (fenced_code_block
                (code_fence_content) @code_content
            ) @fenced_code

            (indented_code_block) @indented_code
        """

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for Markdown sections.

        In Markdown, sections (defined by headers) are the closest equivalent to classes.
        Captures both ATX and Setext style headers.

        Returns:
            Tree-sitter query string for finding Markdown headers/sections
        """
        return """
            (atx_heading
                (atx_h1_marker)? @h1_marker
                (atx_h2_marker)? @h2_marker
                (atx_h3_marker)? @h3_marker
                (atx_h4_marker)? @h4_marker
                (atx_h5_marker)? @h5_marker
                (atx_h6_marker)? @h6_marker
            ) @atx_heading

            (setext_heading) @setext_heading
        """

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for Markdown comments.

        HTML-style comments in Markdown documents.

        Returns:
            Tree-sitter query string for finding Markdown comments
        """
        return """
            (html_block) @html_comment
        """

    def get_heading_query(self) -> str:
        """Get tree-sitter query pattern for all Markdown headings.

        Returns:
            Tree-sitter query string for finding all heading types
        """
        return """
            (atx_heading) @atx_heading
            (setext_heading) @setext_heading
        """

    def get_list_query(self) -> str:
        """Get tree-sitter query pattern for Markdown lists.

        Returns:
            Tree-sitter query string for finding lists and list items
        """
        return """
            (list) @list
            (list_item) @list_item
        """

    def get_paragraph_query(self) -> str:
        """Get tree-sitter query pattern for Markdown paragraphs.

        Returns:
            Tree-sitter query string for finding paragraphs
        """
        return """
            (paragraph) @paragraph
        """

    def get_blockquote_query(self) -> str:
        """Get tree-sitter query pattern for Markdown blockquotes.

        Returns:
            Tree-sitter query string for finding blockquotes
        """
        return """
            (block_quote) @blockquote
        """

    def get_table_query(self) -> str:
        """Get tree-sitter query pattern for Markdown tables (GFM).

        Returns:
            Tree-sitter query string for finding pipe tables
        """
        return """
            (pipe_table) @pipe_table
        """

    def get_link_query(self) -> str:
        """Get tree-sitter query pattern for Markdown links and images.

        Note: tree-sitter-markdown doesn't have dedicated 'link' or 'image' nodes.
        Inline links/images are parsed as sequences of punctuation tokens within
        'inline' nodes, making them difficult to query reliably. We focus on
        link reference definitions which are properly structured.

        Returns:
            Tree-sitter query string for finding link reference definitions
        """
        return """
            (link_reference_definition) @link_ref_def
        """

    def extract_function_name(self, node: "TSNode | None", source: str) -> str:
        """Extract name from a Markdown code block.

        For code blocks, attempts to extract language name or use fallback.

        Args:
            node: Tree-sitter code block node
            source: Source markdown text

        Returns:
            Code block identifier or fallback name
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "code_block")

        # For fenced code blocks, try to get the language
        if node.type == "fenced_code_block":
            # Look for info_string (language specifier)
            info_node = self.find_child_by_type(node, "info_string")
            if info_node:
                lang = self.get_node_text(info_node, source).strip()
                if lang:
                    return f"code_block_{lang}"

        # For any code block, use line number for identification
        return self.get_fallback_name(node, "code_block")

    def extract_class_name(self, node: "TSNode | None", source: str) -> str:
        """Extract title from a Markdown heading.

        Args:
            node: Tree-sitter heading node
            source: Source markdown text

        Returns:
            Heading text or fallback name
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return self.get_fallback_name(node, "heading")

        # Extract heading text content
        heading_text = ""

        if node.type == "atx_heading":
            # Get all child text nodes, skipping the markers
            for child in self.walk_tree(node):
                if child.type in ("text", "code_span", "emphasis", "strong"):
                    text_content = self.get_node_text(child, source).strip()
                    if text_content and not text_content.startswith("#"):
                        heading_text += text_content + " "

        elif node.type == "setext_heading":
            # For setext headings, get the first line (before the underline)
            text_node = self.find_child_by_type(node, "paragraph")
            if text_node:
                heading_text = self.get_node_text(text_node, source).strip()

        # Clean up the heading text
        heading_text = heading_text.strip()
        if heading_text:
            # Convert to a valid identifier-like name
            cleaned = "".join(c if c.isalnum() else "_" for c in heading_text)
            cleaned = "_".join(word for word in cleaned.split("_") if word)
            return cleaned[:50]  # Limit length

        return self.get_fallback_name(node, "heading")

    def extract_heading_level(self, node: "TSNode | None", source: str) -> int:
        """Extract heading level from a Markdown heading node.

        Args:
            node: Tree-sitter heading node
            source: Source markdown text

        Returns:
            Heading level (1-6), defaults to 1 if not determinable
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return 1

        if node.type == "atx_heading":
            # Count the number of # symbols
            marker_node = None
            for i in range(node.child_count):
                child = node.child(i)
                if (
                    child
                    and child.type.startswith("atx_h")
                    and child.type.endswith("_marker")
                ):
                    marker_node = child
                    break

            if marker_node:
                marker_text = self.get_node_text(marker_node, source)
                return marker_text.count("#")

        elif node.type == "setext_heading":
            # Check the underline character to determine level
            for child in self.walk_tree(node):
                if child.type == "setext_h1_underline":
                    return 1
                elif child.type == "setext_h2_underline":
                    return 2

        return 1

    def extract_code_language(self, node: "TSNode | None", source: str) -> str:
        """Extract language from a fenced code block.

        Args:
            node: Tree-sitter fenced code block node
            source: Source markdown text

        Returns:
            Language identifier or empty string
        """
        if (
            not TREE_SITTER_AVAILABLE
            or node is None
            or node.type != "fenced_code_block"
        ):
            return ""

        # Look for info_string (language specifier)
        info_node = self.find_child_by_type(node, "info_string")
        if info_node:
            lang = self.get_node_text(info_node, source).strip()
            # Return just the language part (before any spaces)
            return lang.split()[0] if lang else ""

        return ""

    def extract_list_type(self, node: "TSNode | None", source: str) -> str:
        """Extract list type (ordered/unordered) from a list node.

        Args:
            node: Tree-sitter list node
            source: Source markdown text

        Returns:
            "ordered" or "unordered"
        """
        if not TREE_SITTER_AVAILABLE or node is None or node.type != "list":
            return "unordered"

        # Check the first list item for its marker
        first_item = self.find_child_by_type(node, "list_item")
        if first_item:
            # Look for ordered list marker (numbers followed by . or ))
            for child in self.walk_tree(first_item):
                if (
                    child.type == "list_marker_dot"
                    or child.type == "list_marker_parenthesis"
                ):
                    return "ordered"
                elif child.type in (
                    "list_marker_minus",
                    "list_marker_plus",
                    "list_marker_star",
                ):
                    return "unordered"

        return "unordered"

    def extract_link_url(self, node: "TSNode | None", source: str) -> str:
        """Extract URL from a link reference definition node.

        Args:
            node: Tree-sitter link_reference_definition node
            source: Source markdown text

        Returns:
            URL or empty string
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return ""

        # Look for link destination (only available in link_reference_definition)
        dest_node = self.find_child_by_type(node, "link_destination")
        if dest_node:
            url = self.get_node_text(dest_node, source).strip()
            # Remove angle brackets if present
            if url.startswith("<") and url.endswith(">"):
                url = url[1:-1]
            return url

        return ""

    def extract_link_text(self, node: "TSNode | None", source: str) -> str:
        """Extract label text from a link reference definition.

        Args:
            node: Tree-sitter link_reference_definition node
            source: Source markdown text

        Returns:
            Link label text or empty string
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return ""

        # Look for link label (the [text] part of [text]: url)
        label_node = self.find_child_by_type(node, "link_label")
        if label_node:
            label_text = self.get_node_text(label_node, source).strip()
            # Remove brackets if present
            if label_text.startswith("[") and label_text.endswith("]"):
                label_text = label_text[1:-1]
            return label_text

        return ""

    def should_include_node(self, node: "TSNode | None", source: str) -> bool:
        """Determine if a Markdown node should be included as a chunk.

        Filters out very small content and some structural elements.

        Args:
            node: Tree-sitter node
            source: Source markdown text

        Returns:
            True if node should be included, False otherwise
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return False

        # Get the node text to check size
        text = self.get_node_text(node, source).strip()

        # Skip very small nodes (less than 10 characters for markdown)
        if len(text) < 10:
            return False

        # Skip empty paragraphs
        if node.type == "paragraph" and not text:
            return False

        # Skip list markers and other structural elements
        if node.type in (
            "list_marker_minus",
            "list_marker_plus",
            "list_marker_star",
            "list_marker_dot",
            "list_marker_parenthesis",
            "atx_h1_marker",
            "atx_h2_marker",
            "atx_h3_marker",
            "atx_h4_marker",
            "atx_h5_marker",
            "atx_h6_marker",
        ):
            return False

        return True

    def create_heading_chunk(
        self, node: "TSNode | None", source: str, file_path, name: str
    ) -> dict[str, Any]:
        """Create a chunk dictionary for a Markdown heading.

        Args:
            node: Tree-sitter heading node
            source: Source markdown text
            file_path: Path to markdown file
            name: Heading name

        Returns:
            Dictionary representing the heading chunk
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        level = self.extract_heading_level(node, source)
        heading_text = self.get_node_text(node, source).strip()

        return self.create_chunk_dict(
            node=node,
            source=source,
            file_path=file_path,
            chunk_type=ChunkType.CLASS,  # Treat headings as high-level structures
            name=name,
            display_name=f"# {heading_text}",
            level=level,
            heading_type=node.type,
        )

    def create_code_block_chunk(
        self, node: "TSNode | None", source: str, file_path, name: str
    ) -> dict[str, Any]:
        """Create a chunk dictionary for a Markdown code block.

        Args:
            node: Tree-sitter code block node
            source: Source markdown text
            file_path: Path to markdown file
            name: Code block name

        Returns:
            Dictionary representing the code block chunk
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        language = (
            self.extract_code_language(node, source)
            if node.type == "fenced_code_block"
            else ""
        )

        return self.create_chunk_dict(
            node=node,
            source=source,
            file_path=file_path,
            chunk_type=ChunkType.FUNCTION,  # Treat code blocks as function-like
            name=name,
            display_name=f"Code Block ({language})" if language else "Code Block",
            code_language=language,
            block_type=node.type,
        )

    def create_list_chunk(
        self, node: "TSNode | None", source: str, file_path, name: str
    ) -> dict[str, Any]:
        """Create a chunk dictionary for a Markdown list.

        Args:
            node: Tree-sitter list node
            source: Source markdown text
            file_path: Path to markdown file
            name: List name

        Returns:
            Dictionary representing the list chunk
        """
        if not TREE_SITTER_AVAILABLE or node is None:
            return {}

        list_type = self.extract_list_type(node, source)

        return self.create_chunk_dict(
            node=node,
            source=source,
            file_path=file_path,
            chunk_type=ChunkType.STRUCT,  # Treat lists as structured content
            name=name,
            display_name=f"{list_type.title()} List",
            list_type=list_type,
        )

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in Markdown.

        Since Markdown uses tree-sitter, we map concepts to existing queries.
        """
        if concept == UniversalConcept.DEFINITION:
            # Combine heading and code block queries - these are the main "definitions" in markdown
            heading_query = self.get_heading_query()
            function_query = self.get_function_query()

            queries = []
            if heading_query.strip():
                queries.append(heading_query.strip())
            if function_query.strip():
                queries.append(function_query.strip())

            if queries:
                return "\n".join(queries)
            else:
                return None

        elif concept == UniversalConcept.BLOCK:
            # Use paragraph, list, and blockquote queries for block content
            paragraph_query = self.get_paragraph_query()
            list_query = self.get_list_query()
            blockquote_query = self.get_blockquote_query()

            queries = []
            if paragraph_query.strip():
                queries.append(paragraph_query.strip())
            if list_query.strip():
                queries.append(list_query.strip())
            if blockquote_query.strip():
                queries.append(blockquote_query.strip())

            if queries:
                return "\n".join(queries)
            else:
                return None

        elif concept == UniversalConcept.COMMENT:
            # Use comment query
            comment_query = self.get_comment_query()
            if comment_query.strip():
                return comment_query.strip()
            else:
                return None

        elif concept == UniversalConcept.IMPORT:
            # Use link query for imports/references
            link_query = self.get_link_query()
            if link_query.strip():
                return link_query.strip()
            else:
                return None

        elif concept == UniversalConcept.STRUCTURE:
            # Use table query for structured content
            table_query = self.get_table_query()
            if table_query.strip():
                return table_query.strip()
            else:
                return None

        else:
            return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for markdown processing
        source = content.decode("utf-8", errors="replace")

        if concept == UniversalConcept.DEFINITION:
            # Try to find the main definition node (heading or code block)
            def_node = self._find_definition_node(captures)

            if def_node:
                if def_node.type in ("atx_heading", "setext_heading"):
                    return self.extract_class_name(def_node, source)
                elif def_node.type in ("fenced_code_block", "indented_code_block"):
                    return self.extract_function_name(def_node, source)
                else:
                    # Default to heading extraction
                    return self.extract_class_name(def_node, source)
            else:
                return "unnamed_definition"

        elif concept == UniversalConcept.BLOCK:
            # Find block content node
            block_node = self._find_definition_node(captures)
            if block_node:
                if block_node.type == "paragraph":
                    # Extract first few words for paragraph name
                    text = self.get_node_text(block_node, source).strip()
                    words = text.split()[:5]
                    if words:
                        name = "_".join(w for w in words if w.isalnum())
                        return f"paragraph_{name[:30]}"
                    return "paragraph"
                elif block_node.type == "list":
                    list_type = self.extract_list_type(block_node, source)
                    return f"{list_type}_list"
                elif block_node.type == "list_item":
                    text = self.get_node_text(block_node, source).strip()
                    # Remove list markers
                    text = text.lstrip("*-+ ").lstrip("0123456789. ")
                    words = text.split()[:3]
                    if words:
                        name = "_".join(w for w in words if w.isalnum())
                        return f"item_{name[:20]}"
                    return "list_item"
                elif block_node.type == "block_quote":
                    return "blockquote"
                else:
                    return f"{block_node.type}_block"
            else:
                return "unnamed_block"

        elif concept == UniversalConcept.COMMENT:
            # Comments in markdown are HTML comments
            comment_node = self._find_definition_node(captures)
            if comment_node:
                line = comment_node.start_point[0] + 1
                return f"comment_line_{line}"
            else:
                return "unnamed_comment"

        elif concept == UniversalConcept.IMPORT:
            # Link reference definitions only (inline links/images aren't reliably parseable)
            link_node = self._find_definition_node(captures)
            if link_node:
                if link_node.type == "link_reference_definition":
                    label_text = self.extract_link_text(link_node, source)
                    if label_text:
                        name = "".join(c if c.isalnum() else "_" for c in label_text)
                        name = "_".join(word for word in name.split("_") if word)
                        return f"link_ref_{name[:20]}"
                    return "link_ref_def"
                else:
                    return f"{link_node.type}_import"
            else:
                return "unnamed_import"

        elif concept == UniversalConcept.STRUCTURE:
            # Tables and structured content
            struct_node = self._find_definition_node(captures)
            if struct_node and struct_node.type == "pipe_table":
                return "table"
            else:
                return "structure"

        else:
            return f"unnamed_{concept.value}"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> str:
        """Extract content from captures for this concept.

        This is the critical method - it must preserve the original markdown text
        so that regex searches can find the exact content.
        """

        # Find the main definition node and extract its text
        def_node = self._find_definition_node(captures)
        if def_node:
            source = content.decode("utf-8", errors="replace")
            return self.get_node_text(def_node, source)
        return ""

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, "TSNode"], content: bytes
    ) -> dict[str, Any]:
        """Extract markdown-specific metadata."""

        def_node = self._find_definition_node(captures)
        source = content.decode("utf-8", errors="replace")

        metadata = {
            "concept": concept.value,
            "language": self.language.value,
            "capture_names": list(captures.keys()),
        }

        if def_node:
            metadata["node_type"] = def_node.type

            # Add concept-specific metadata
            if concept == UniversalConcept.DEFINITION:
                if def_node.type in ("atx_heading", "setext_heading"):
                    metadata["heading_level"] = self.extract_heading_level(
                        def_node, source
                    )
                elif def_node.type == "fenced_code_block":
                    metadata["code_language"] = self.extract_code_language(
                        def_node, source
                    )

            elif concept == UniversalConcept.BLOCK:
                if def_node.type == "list":
                    metadata["list_type"] = self.extract_list_type(def_node, source)

            elif concept == UniversalConcept.IMPORT:
                if def_node.type == "link_reference_definition":
                    metadata["url"] = self.extract_link_url(def_node, source)
                    metadata["label"] = self.extract_link_text(def_node, source)

        return metadata

    def _find_definition_node(
        self, captures: dict[str, "TSNode"]
    ) -> Optional["TSNode"]:
        """Find the main definition node from captures.

        Args:
            captures: Dictionary of captured nodes

        Returns:
            Main definition node or None
        """
        # Priority order - prefer specific nodes over generic
        priority_keys = [
            "atx_heading",
            "setext_heading",
            "fenced_code_block",
            "code_block",
            "paragraph",
            "list_item",
            "blockquote",
            "table",
        ]

        for key in priority_keys:
            if key in captures:
                return captures[key]

        # Return first capture if no priority match
        if captures:
            return list(captures.values())[0]

        return None

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path
    ) -> Path | None:
        """Data formats don't have imports."""
        return None
