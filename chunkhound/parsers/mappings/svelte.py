"""Svelte language mapping for unified parser architecture.

This module provides Svelte component-specific parsing that handles the multi-section
structure of Svelte Single File Components (.svelte files).

## Approach
- Extract <script>, <template>, <style> sections via regex
- Parse script content with TypeScript parser (inherited)
- Create text chunks for template and style sections

## Supported Features
- <script lang="ts"> parsing
- Regular <script> support
- Template as searchable text block
- Style as optional text block

## Limitations (Phase 1)
- Template directives not parsed (no #if/#each structure)
- No cross-section reference tracking
- No component usage graph
- Basic section extraction (regex-based)
"""

import re

from chunkhound.core.types.common import Language
from chunkhound.parsers.mappings.typescript import TypeScriptMapping


class SvelteMapping(TypeScriptMapping):
    """Svelte component language mapping extending TypeScript mapping.

    Handles Svelte Single File Component structure with multiple sections.
    Script sections are parsed as TypeScript, template and style as text.
    """

    def __init__(self) -> None:
        """Initialize Svelte mapping (delegates to TypeScript for script parsing)."""
        super().__init__()
        self.language = Language.SVELTE  # Override to SVELTE

    # Section extraction patterns
    SCRIPT_PATTERN = re.compile(
        r"<script\s*([^>]*)>(.*?)</script>", re.DOTALL | re.IGNORECASE
    )

    # Svelte doesn't use <template> tags - the template is implicit
    # Everything outside <script> and <style> is template
    STYLE_PATTERN = re.compile(
        r"<style\s*([^>]*)>(.*?)</style>", re.DOTALL | re.IGNORECASE
    )

    def extract_sections(self, content: str) -> dict[str, list[tuple[str, str, int]]]:
        """Extract script, template, and style sections from Svelte component.

        Args:
            content: Full Svelte component content

        Returns:
            Dictionary with 'script', 'template', 'style' keys, each containing
            list of (attributes, section_content, start_line) tuples
        """
        sections: dict[str, list[tuple[str, str, int]]] = {
            "script": [],
            "template": [],
            "style": [],
        }

        # Collect all ranges to exclude from template
        # (position-based, not string replacement)
        excluded_ranges: list[tuple[int, int]] = []

        # Extract script sections
        for match in self.SCRIPT_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            script_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["script"].append((attrs, script_content, start_line))
            excluded_ranges.append((match.start(), match.end()))

        # Extract style sections
        for match in self.STYLE_PATTERN.finditer(content):
            attrs = match.group(1).strip()
            style_content = match.group(2)
            start_line = content[: match.start()].count("\n") + 1
            sections["style"].append((attrs, style_content, start_line))
            excluded_ranges.append((match.start(), match.end()))

        # Extract template (everything outside script and style tags)
        # Use position-based extraction to avoid string replacement bugs
        # Emit one template chunk per span to preserve accurate line numbers
        if excluded_ranges:
            # Sort ranges by start position
            excluded_ranges.sort()

            last_end = 0

            for start, end in excluded_ranges:
                if start > last_end:
                    # Content between last_end and start is template
                    template_part = content[last_end:start]
                    if template_part.strip():
                        # Calculate line number for this template part
                        template_start_line = content[:last_end].count("\n") + 1
                        # Preserve raw content for accurate line offsets
                        sections["template"].append(
                            ("", template_part, template_start_line)
                        )
                last_end = end

            # Add remaining content after last excluded range
            if last_end < len(content):
                template_part = content[last_end:]
                if template_part.strip():
                    template_start_line = content[:last_end].count("\n") + 1
                    sections["template"].append(
                        ("", template_part, template_start_line)
                    )
        else:
            # No script or style sections, entire content is template
            if content.strip():
                sections["template"].append(("", content, 1))

        return sections
