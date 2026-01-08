"""Vue language mapping for unified parser architecture.

This module provides Vue SFC-specific parsing that handles the multi-section
structure of Vue Single File Components (.vue files).

## Approach
- Extract <script>, <template>, <style> sections via regex
- Parse script content with TypeScript parser (inherited)
- Create text chunks for template and style sections
- Add Vue-specific metadata for compiler macros and composables

## Supported Features
- <script setup lang="ts"> parsing
- defineProps, defineEmits, defineExpose detection
- Composable function detection (use* pattern)
- Regular <script lang="ts"> support
- Template as searchable text block
- Style as optional text block

## Limitations (Phase 1)
- Template directives not parsed (no v-if/v-for structure)
- No cross-section reference tracking
- No component usage graph
- Basic section extraction (regex-based)
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.typescript import TypeScriptMapping

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

try:
    from tree_sitter import Node as TSNode

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TSNode = Any  # type: ignore


class VueMapping(TypeScriptMapping):
    """Vue SFC language mapping extending TypeScript mapping.

    Handles Vue Single File Component structure with multiple sections.
    Script sections are parsed as TypeScript, template and style as text.
    """

    def __init__(self) -> None:
        """Initialize Vue mapping (delegates to TypeScript for script parsing)."""
        super().__init__()
        self.language = Language.VUE  # Override to VUE

    # Section extraction patterns
    SCRIPT_PATTERN = re.compile(
        r"<script\s*([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE
    )

    TEMPLATE_PATTERN = re.compile(
        r"<template\s*([^>]*)>(.*?)</template>", re.DOTALL | re.IGNORECASE
    )

    STYLE_PATTERN = re.compile(
        r"<style\s*([^>]*)>(.*?)</style>", re.DOTALL | re.IGNORECASE
    )

    # Vue compiler macro patterns
    VUE_MACROS = [
        "defineProps",
        "defineEmits",
        "defineExpose",
        "withDefaults",
        "defineOptions",
        "defineSlots",
        "defineModel",
    ]

    def extract_sections(self, content: str) -> dict[str, list[tuple[str, str, int]]]:
        """Extract script, template, and style sections from Vue SFC.

        Args:
            content: Full Vue SFC content

        Returns:
            Dictionary with 'script', 'template', 'style' keys, each containing
            list of (attributes, section_content, start_line) tuples
        """
        sections: dict[str, list[tuple[str, str, int]]] = {
            "script": [],
            "template": [],
            "style": [],
        }

        # Extract script sections
        for match in self.SCRIPT_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            script_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["script"].append((attrs, script_content, start_line))

        # Extract template sections
        for match in self.TEMPLATE_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            template_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["template"].append((attrs, template_content, start_line))

        # Extract style sections
        for match in self.STYLE_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            style_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["style"].append((attrs, style_content, start_line))

        return sections

    def is_script_setup(self, attrs: str) -> bool:
        """Check if script tag has 'setup' attribute."""
        return "setup" in attrs.lower()

    def get_script_lang(self, attrs: str) -> str:
        """Extract lang attribute from script tag (ts, js, tsx, jsx)."""
        lang_match = re.search(r'lang\s*=\s*["\']?(\w+)["\']?', attrs, re.IGNORECASE)
        if lang_match:
            return lang_match.group(1).lower()
        return "js"  # Default to JavaScript

    def detect_vue_macros(self, script_content: str) -> list[str]:
        """Detect which Vue compiler macros are used in script."""
        found_macros = []
        for macro in self.VUE_MACROS:
            if macro in script_content:
                found_macros.append(macro)
        return found_macros

    def detect_composables(self, script_content: str) -> list[str]:
        """Detect composable usage (functions starting with 'use')."""
        # Pattern: const/let { ... } = useXxx(...) or const x = useXxx(...)
        composable_pattern = re.compile(r"\b(use[A-Z]\w*)\s*\(")
        matches = composable_pattern.findall(script_content)
        return list(set(matches))  # Remove duplicates

    def _extract_attributes(
        self, start_tag_node: Optional["TSNode"], source: str
    ) -> str:
        """Extract attributes from a start_tag node.

        Args:
            start_tag_node: Tree-sitter start_tag node
            source: Full source code

        Returns:
            Attributes as a string (e.g., "setup lang=\"ts\"")
        """
        if not start_tag_node:
            return ""

        attrs = []
        for child in start_tag_node.children:
            if child.type == "attribute":
                # Get the attribute text (e.g., "lang=\"ts\"" or "setup")
                attr_text = source[child.start_byte : child.end_byte]
                attrs.append(attr_text)

        return " ".join(attrs)

    def extract_sections_ts(
        self, content: str
    ) -> dict[str, list[tuple[str, str, int]]]:
        """Extract sections using tree-sitter (Phase 2).

        Args:
            content: Full Vue SFC content

        Returns:
            Dictionary with 'script', 'template', 'style' keys, each containing
            list of (attributes, section_content, start_line) tuples
        """
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            # Fall back to regex if tree-sitter-language-pack not available
            return self.extract_sections(content)

        try:
            parser = get_parser("vue")
            tree = parser.parse(content.encode())
        except Exception:
            # Fall back to regex on any parsing error
            return self.extract_sections(content)

        sections: dict[str, list[tuple[str, str, int]]] = {
            "script": [],
            "template": [],
            "style": [],
        }

        for child in tree.root_node.children:
            if child.type == "script_element":
                # Extract attributes from start_tag
                start_tag = None
                for c in child.children:
                    if c.type == "start_tag":
                        start_tag = c
                        break

                attrs = (
                    self._extract_attributes(start_tag, content) if start_tag else ""
                )

                # Find raw_text node
                raw_text_node = None
                for c in child.children:
                    if c.type == "raw_text":
                        raw_text_node = c
                        break

                if raw_text_node:
                    script_content = content[
                        raw_text_node.start_byte : raw_text_node.end_byte
                    ]
                    start_line = raw_text_node.start_point[0] + 1
                    sections["script"].append((attrs, script_content, start_line))

            elif child.type == "template_element":
                # Extract attributes from start_tag
                start_tag = None
                end_tag = None
                for c in child.children:
                    if c.type == "start_tag":
                        start_tag = c
                    elif c.type == "end_tag":
                        end_tag = c

                attrs = (
                    self._extract_attributes(start_tag, content) if start_tag else ""
                )

                # Template content is between start_tag and end_tag
                if start_tag and end_tag:
                    # Content starts after start_tag ends and goes until end_tag starts
                    content_start = start_tag.end_byte
                    content_end = end_tag.start_byte
                    template_content = content[content_start:content_end]
                    start_line = start_tag.end_point[0] + 1
                    sections["template"].append((attrs, template_content, start_line))

            elif child.type == "style_element":
                # Extract attributes from start_tag
                start_tag = None
                for c in child.children:
                    if c.type == "start_tag":
                        start_tag = c
                        break

                attrs = (
                    self._extract_attributes(start_tag, content) if start_tag else ""
                )

                # Find raw_text node
                raw_text_node = None
                for c in child.children:
                    if c.type == "raw_text":
                        raw_text_node = c
                        break

                if raw_text_node:
                    style_content = content[
                        raw_text_node.start_byte : raw_text_node.end_byte
                    ]
                    start_line = raw_text_node.start_point[0] + 1
                    sections["style"].append((attrs, style_content, start_line))

        return sections

    def resolve_import_path(self, import_text: str, base_dir: Path, source_file: Path) -> Path | None:
        """Resolve relative import path to absolute file path.

        Args:
            import_text: Import statement text
            base_dir: Base directory for resolution
            source_file: Source file containing the import

        Returns:
            Resolved absolute path or None if not resolvable
        """
        match = re.search(r'''(?:from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))''', import_text)
        if not match:
            return None
        import_path = match.group(1) or match.group(2)
        if not import_path or not import_path.startswith('.'):
            return None
        source_dir = self._resolve_source_dir(source_file, base_dir)
        resolved = (source_dir / import_path).resolve()
        if resolved.exists() and resolved.is_file():
            return resolved
        for ext in ['.vue', '.ts', '.js', '.tsx', '.jsx']:
            with_ext = resolved.with_suffix(ext)
            if with_ext.exists():
                return with_ext
        for index in ['index.vue', 'index.ts', 'index.js']:
            index_path = resolved / index
            if index_path.exists():
                return index_path
        return None
