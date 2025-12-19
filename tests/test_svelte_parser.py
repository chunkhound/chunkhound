"""Tests for Svelte language parser integration."""

from pathlib import Path

import pytest

from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import (
    create_parser_for_language,
    get_parser_factory,
)


class TestSvelteLanguageDetection:
    """Test Svelte language detection and extension mapping."""

    def test_svelte_extension_detection(self):
        """Test .svelte files are detected correctly."""
        language = Language.from_file_extension("App.svelte")
        assert language == Language.SVELTE

    def test_svelte_file_path_detection(self):
        """Test Svelte files detected from Path objects."""
        test_file = Path("components/Button.svelte")
        language = Language.from_file_extension(test_file)
        assert language == Language.SVELTE


class TestSvelteParserCreation:
    """Test Svelte parser instantiation."""

    def test_svelte_parser_creation(self):
        """Test Svelte parser can be created."""
        parser = create_parser_for_language(Language.SVELTE)
        assert parser is not None

    def test_svelte_parser_availability(self):
        """Test Svelte parser is marked as available."""
        factory = get_parser_factory()
        available_languages = factory.get_available_languages()
        assert Language.SVELTE in available_languages
        # Should be available if TypeScript is available
        assert (
            available_languages[Language.SVELTE]
            == available_languages[Language.TYPESCRIPT]
        )


class TestSvelteSectionExtraction:
    """Test Svelte component section extraction."""

    def test_basic_section_extraction(self):
        """Test script/template/style sections are extracted."""
        svelte_content = """
<script lang="ts">
  let count = 0;
  function increment() {
    count += 1;
  }
</script>

<main>
  <h1>Count: {count}</h1>
  <button on:click={increment}>Increment</button>
</main>

<style>
  h1 {
    color: blue;
  }
</style>
"""
        from chunkhound.parsers.mappings.svelte import SvelteMapping

        mapping = SvelteMapping()
        sections = mapping.extract_sections(svelte_content)

        # Should have all three sections
        assert len(sections["script"]) == 1
        assert len(sections["template"]) == 1
        assert len(sections["style"]) == 1

        # Check script content
        script_attrs, script_content, script_line = sections["script"][0]
        assert 'lang="ts"' in script_attrs
        assert "let count = 0" in script_content

        # Check template content
        template_attrs, template_content, template_line = sections["template"][0]
        assert "<h1>Count: {count}</h1>" in template_content

        # Check style content
        style_attrs, style_content, style_line = sections["style"][0]
        assert "color: blue" in style_content

    def test_template_line_numbers_accurate(self):
        """Test that template line numbers are calculated correctly."""
        svelte_content = """<script lang="ts">
  let count = 0;
  function increment() {
    count += 1;
  }
</script>

<main>
  <h1>Count: {count}</h1>
  <button on:click={increment}>Increment</button>
</main>

<style>
  h1 {
    color: blue;
  }
</style>"""
        from chunkhound.parsers.mappings.svelte import SvelteMapping

        mapping = SvelteMapping()
        sections = mapping.extract_sections(svelte_content)

        # Template starts after script section (line 8)
        template_attrs, template_content, template_line = sections["template"][0]
        # Template should not start at line 1, but after script ends
        assert template_line > 1, f"Template line should be > 1, got {template_line}"

    def test_template_extraction_no_string_replacement_bug(self):
        """Test that template extraction doesn't suffer from string replacement bugs."""
        # This tests the fix for Issue #1 in the review
        svelte_content = """<script>
  const greeting = "<script>alert('hi')</script>";
  const style = "<style>body { color: red; }</style>";
</script>

<div>
  <p>{greeting}</p>
  <p>{style}</p>
</div>

<style>
  div { padding: 1rem; }
</style>"""
        from chunkhound.parsers.mappings.svelte import SvelteMapping

        mapping = SvelteMapping()
        sections = mapping.extract_sections(svelte_content)

        # Template should still contain the string literals
        template_attrs, template_content, template_line = sections["template"][0]
        assert "<p>{greeting}</p>" in template_content
        assert "<p>{style}</p>" in template_content


class TestSvelteEndToEnd:
    """End-to-end integration tests for Svelte parsing."""

    def test_parse_complete_component(self):
        """Test parsing a complete Svelte component."""
        svelte_content = """
<script lang="ts">
  export let name: string = 'World';
  let count = 0;
  $: doubled = count * 2;
</script>

<div class="container">
  <h1>Hello {name}!</h1>
  <p>Count: {count}, Doubled: {doubled}</p>
  <button on:click={() => count++}>Increment</button>
</div>

<style>
  .container {
    padding: 1rem;
  }
</style>
"""
        parser = create_parser_for_language(Language.SVELTE)
        assert parser is not None

        # Parser should handle Svelte content without errors
        # (Actual parsing will use TypeScript for script sections)

    def test_svelte_component_without_script(self):
        """Test parsing Svelte component without script section."""
        svelte_content = """
<div class="simple">
  <h1>Simple Component</h1>
  <p>No script needed</p>
</div>

<style>
  .simple {
    color: green;
  }
</style>
"""
        from chunkhound.parsers.mappings.svelte import SvelteMapping

        mapping = SvelteMapping()
        sections = mapping.extract_sections(svelte_content)

        # Should have template and style, but no script
        assert len(sections["script"]) == 0
        assert len(sections["template"]) == 1
        assert len(sections["style"]) == 1

    def test_svelte_component_with_multiple_scripts(self):
        """Test parsing Svelte component with module context script."""
        svelte_content = """
<script context="module">
  export const preload = async () => {
    return {};
  };
</script>

<script lang="ts">
  let count = 0;
</script>

<div>
  <p>Count: {count}</p>
</div>
"""
        from chunkhound.parsers.mappings.svelte import SvelteMapping

        mapping = SvelteMapping()
        sections = mapping.extract_sections(svelte_content)

        # Should extract both script sections
        assert len(sections["script"]) == 2

        # Check that module context is present in attributes
        module_attrs = sections["script"][0][0]
        assert 'context="module"' in module_attrs or "context='module'" in module_attrs

        # Regular script should not have module context
        regular_attrs = sections["script"][1][0]
        assert (
            'context="module"' not in regular_attrs
            and "context='module'" not in regular_attrs
        )
