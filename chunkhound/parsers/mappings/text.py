"""Text language mapping for unified parser architecture.

This module provides plain text parsing and extraction logic
for the universal concept system. It handles paragraphs, sections,
and line-based chunking using simple text analysis.
"""

import re
from pathlib import Path
from typing import Any

from tree_sitter import Node

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.base import BaseMapping
from chunkhound.parsers.universal_engine import UniversalConcept


class TextMapping(BaseMapping):
    """Text-specific mapping for universal concepts using line-based analysis."""

    def __init__(self) -> None:
        """Initialize Text mapping."""
        super().__init__(Language.TEXT)

    # BaseMapping required methods
    def get_function_query(self) -> str:
        """Get tree-sitter query pattern for function definitions (not applicable to text)."""
        return ""

    def get_class_query(self) -> str:
        """Get tree-sitter query pattern for class definitions (not applicable to text)."""
        return ""

    def get_comment_query(self) -> str:
        """Get tree-sitter query pattern for comments (text files don't have formal comments)."""
        return ""

    def extract_function_name(self, node: Node | None, source: str) -> str:
        """Extract function name from a function definition node (not applicable to text)."""
        return ""

    def extract_class_name(self, node: Node | None, source: str) -> str:
        """Extract class name from a class definition node (not applicable to text)."""
        return ""

    # LanguageMapping protocol methods
    def get_query_for_concept(self, concept: UniversalConcept) -> str | None:
        """Get tree-sitter query for universal concept in text.

        Since plain text doesn't use tree-sitter, we return None for all concepts
        and handle parsing through custom text analysis.
        """
        # Plain text doesn't use tree-sitter queries - we'll parse using text analysis
        return None

    def extract_name(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract name from captures for this concept."""

        # Convert bytes to string for text processing
        source = content.decode("utf-8")
        lines = source.split("\n")

        if concept == UniversalConcept.DEFINITION:
            # Look for headings or prominent lines
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                # Check for heading patterns
                if line.startswith("#"):
                    # Markdown-style heading
                    heading = line.lstrip("#").strip()
                    if heading:
                        return f"heading_{heading[:30]}"
                elif len(line) > 5 and (line.isupper() or line.endswith(":")):
                    # All caps or ending with colon might be a section header
                    header = line.rstrip(":").strip()
                    return f"section_{header[:30]}"
                elif i < len(lines) - 1 and lines[i + 1].strip() in [
                    "=" * len(line),
                    "-" * len(line),
                ]:
                    # Underlined heading (reStructuredText style)
                    return f"title_{line[:30]}"
                elif line.startswith(("*", "-", "+")):
                    # List item
                    item_text = line.lstrip("*-+ ").strip()
                    if item_text:
                        return f"list_item_{item_text[:20]}"
                elif re.match(r"^\d+\.", line):
                    # Numbered list
                    item_text = re.sub(r"^\d+\.\s*", "", line).strip()
                    if item_text:
                        return f"numbered_item_{item_text[:20]}"

                # If it's the first non-empty line, use it as a definition
                if i == 0 or (i > 0 and not any(lines[j].strip() for j in range(i))):
                    return f"first_line_{line[:30]}"

            return "text_definition"

        elif concept == UniversalConcept.BLOCK:
            # Identify text blocks (paragraphs, sections)
            paragraphs = self._split_into_paragraphs(source)
            if len(paragraphs) > 1:
                return f"text_block_{len(paragraphs)}_paragraphs"
            else:
                return "single_paragraph"

        elif concept == UniversalConcept.COMMENT:
            # Look for lines that might be comments or annotations
            for line in lines:
                line = line.strip()
                if line.startswith(("TODO:", "FIXME:", "NOTE:", "WARNING:", "HACK:")):
                    annotation = line.split(":")[0].lower()
                    return f"annotation_{annotation}"
                elif line.startswith(("# ", "// ", "<!-- ", "/* ")):
                    # Comment-like patterns
                    return "comment_line"
                elif line.startswith("---") or line.startswith("==="):
                    # Divider lines
                    return "divider_line"

            return "text_comment"

        elif concept == UniversalConcept.IMPORT:
            # Look for references to other files or external content
            for line in lines:
                line = line.strip().lower()
                if any(
                    keyword in line
                    for keyword in [
                        "include",
                        "import",
                        "see also",
                        "reference",
                        "link to",
                    ]
                ):
                    return "text_reference"
                elif line.startswith(("http://", "https://", "file://", "ftp://")):
                    return "url_reference"
                elif re.search(
                    r"\b[\w.-]+\.(txt|md|doc|pdf|html)\b", line, re.IGNORECASE
                ):
                    return "file_reference"

            return "text_import"

        elif concept == UniversalConcept.STRUCTURE:
            return "text_document"

        return "unnamed"

    def extract_content(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> str:
        """Extract content from captures for this concept."""

        # For text, we return different content based on the concept
        source = content.decode("utf-8")

        if concept == UniversalConcept.DEFINITION:
            # Extract the first meaningful section or paragraph
            paragraphs = self._split_into_paragraphs(source)
            if paragraphs:
                return paragraphs[0]
            else:
                lines = source.split("\n")
                # Return first few non-empty lines
                meaningful_lines = [line for line in lines[:5] if line.strip()]
                return "\n".join(meaningful_lines) if meaningful_lines else source[:200]

        elif concept == UniversalConcept.BLOCK:
            # Return paragraph blocks
            paragraphs = self._split_into_paragraphs(source)
            if len(paragraphs) > 1:
                # Return a representative block or summary
                return f"Document with {len(paragraphs)} paragraphs:\n\n{paragraphs[0][:200]}..."
            else:
                return source

        else:
            # Return the entire content for other concepts
            return source

    def extract_metadata(
        self, concept: UniversalConcept, captures: dict[str, Node], content: bytes
    ) -> dict[str, Any]:
        """Extract text-specific metadata."""

        source = content.decode("utf-8")
        metadata = {}

        # Basic text statistics
        lines = source.split("\n")
        words = source.split()

        metadata["line_count"] = len(lines)
        metadata["word_count"] = len(words)
        metadata["char_count"] = len(source)
        metadata["non_empty_lines"] = len([line for line in lines if line.strip()])

        if concept == UniversalConcept.DEFINITION:
            # Analyze potential definitions or headings
            headings = self._extract_headings(source)
            if headings:
                metadata["headings"] = headings[:5]  # Limit to first 5
                metadata["has_structure"] = True

            lists = self._extract_lists(source)
            if lists:
                metadata["list_count"] = len(lists)
                metadata["has_lists"] = True

            # Check for common document patterns
            if any(
                pattern in source.lower()
                for pattern in ["table of contents", "toc", "index"]
            ):
                metadata["document_type"] = "structured"
            elif len(headings) > 2:
                metadata["document_type"] = "hierarchical"
            elif lists:
                metadata["document_type"] = "list_based"
            else:
                metadata["document_type"] = "plain"

        elif concept == UniversalConcept.BLOCK:
            # Analyze paragraph structure
            paragraphs = self._split_into_paragraphs(source)
            metadata["paragraph_count"] = len(paragraphs)

            if paragraphs:
                avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(
                    paragraphs
                )
                metadata["avg_paragraph_words"] = int(avg_paragraph_length)

                # Analyze paragraph types
                short_paragraphs = sum(1 for p in paragraphs if len(p.split()) < 10)
                long_paragraphs = sum(1 for p in paragraphs if len(p.split()) > 50)

                if short_paragraphs > len(paragraphs) * 0.7:
                    metadata["paragraph_style"] = "short_form"
                elif long_paragraphs > len(paragraphs) * 0.3:
                    metadata["paragraph_style"] = "long_form"
                else:
                    metadata["paragraph_style"] = "mixed"

        elif concept == UniversalConcept.COMMENT:
            # Look for annotation patterns
            annotations = self._extract_annotations(source)
            if annotations:
                metadata["annotation_count"] = len(annotations)
                metadata["annotation_types"] = list(set(annotations))

            # Count potential comment lines
            comment_lines = sum(
                1
                for line in lines
                if line.strip().startswith(("#", "//", "/*", "<!--"))
            )
            if comment_lines > 0:
                metadata["comment_lines"] = comment_lines

        elif concept == UniversalConcept.IMPORT:
            # Extract references and links
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', source)
            if urls:
                metadata["url_count"] = len(urls)
                metadata["has_external_links"] = True

            # Look for file references
            file_refs = re.findall(
                r"\b[\w.-]+\.(txt|md|doc|pdf|html|json|xml)\b", source, re.IGNORECASE
            )
            if file_refs:
                metadata["file_references"] = len(set(file_refs))
                metadata["reference_types"] = list(
                    set(ext.split(".")[-1].lower() for ext in file_refs)
                )

        elif concept == UniversalConcept.STRUCTURE:
            # Overall document analysis
            metadata["text_type"] = "plain_text"

            # Check for specific formats within the text
            if re.search(r"```|~~~", source):
                metadata["has_code_blocks"] = True
            if re.search(r"\*\*.*\*\*|__.*__|_.*_|\*.*\*", source):
                metadata["has_formatting"] = True
            if re.search(r"^\s*[-*+]\s", source, re.MULTILINE):
                metadata["has_bullet_lists"] = True
            if re.search(r"^\s*\d+\.\s", source, re.MULTILINE):
                metadata["has_numbered_lists"] = True

            # Estimate reading complexity
            if len(words) > 0:
                avg_word_length = sum(len(word) for word in words) / len(words)
                metadata["avg_word_length"] = round(avg_word_length, 2)

                # Simple readability estimate
                if avg_word_length > 6:
                    metadata["complexity"] = "high"
                elif avg_word_length > 4.5:
                    metadata["complexity"] = "medium"
                else:
                    metadata["complexity"] = "low"

        return metadata

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs using blank line separation."""
        paragraphs = []
        current_paragraph = []

        for line in text.split("\n"):
            if line.strip():
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []

        # Add the last paragraph if it exists
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        return paragraphs

    def _extract_headings(self, text: str) -> list[str]:
        """Extract potential headings from text."""
        headings = []
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Markdown-style headings
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                if heading:
                    headings.append(heading)

            # Underlined headings
            elif i < len(lines) - 1 and lines[i + 1].strip() in [
                "=" * len(line),
                "-" * len(line),
            ]:
                headings.append(line)

            # All caps lines (potential headings)
            elif len(line) > 3 and line.isupper() and not line.endswith("."):
                headings.append(line)

            # Lines ending with colon (section headers)
            elif line.endswith(":") and len(line.split()) < 8:
                headings.append(line[:-1])

        return headings

    def _extract_lists(self, text: str) -> list[str]:
        """Extract list items from text."""
        lists = []
        lines = text.split("\n")

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("*", "-", "+")):
                item = stripped[1:].strip()
                if item:
                    lists.append(item)
            elif re.match(r"^\d+\.", stripped):
                item = re.sub(r"^\d+\.\s*", "", stripped)
                if item:
                    lists.append(item)

        return lists

    def _extract_annotations(self, text: str) -> list[str]:
        """Extract annotation types from text."""
        annotations = []

        for match in re.finditer(
            r"\b(TODO|FIXME|NOTE|WARNING|HACK|BUG|XXX):", text, re.IGNORECASE
        ):
            annotations.append(match.group(1).upper())

        return annotations

    def resolve_import_path(
        self,
        import_text: str,
        base_dir: Path,
        source_file: Path
    ) -> Path | None:
        """Data formats don't have imports."""
        return None
