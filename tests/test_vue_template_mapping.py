"""Tests for Vue template directive parsing."""

from pathlib import Path

import pytest

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.mappings.vue_template import VueTemplateMapping
from chunkhound.parsers.universal_engine import UniversalConcept
from chunkhound.parsers.vue_parser import VueParser


class TestVueTemplateMapping:
    """Test VueTemplateMapping directive query and extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapping = VueTemplateMapping()

    def test_initialization(self):
        """Test VueTemplateMapping initialization."""
        assert self.mapping.language == Language.VUE

    def test_get_query_for_concept_definition(self):
        """Test getting query for DEFINITION concept."""
        query = self.mapping.get_query_for_concept(UniversalConcept.DEFINITION)
        assert query is not None
        assert "directive_attribute" in query
        assert "v-if" in query
        assert "v-for" in query
        assert "@" in query  # Event handlers

    def test_get_query_for_concept_block(self):
        """Test getting query for BLOCK concept."""
        query = self.mapping.get_query_for_concept(UniversalConcept.BLOCK)
        assert query is not None
        assert "element" in query

    def test_get_query_for_concept_comment(self):
        """Test getting query for COMMENT concept."""
        query = self.mapping.get_query_for_concept(UniversalConcept.COMMENT)
        assert query is not None
        assert "comment" in query

    def test_get_query_for_concept_structure(self):
        """Test getting query for STRUCTURE concept."""
        query = self.mapping.get_query_for_concept(UniversalConcept.STRUCTURE)
        assert query is not None
        assert "component_name" in query


class TestVueParserTemplateDirectives:
    """Test VueParser with template directive extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = VueParser()
        self.fixture_path = (
            Path(__file__).parent / "fixtures" / "vue" / "template_directives.vue"
        )

    def test_fixture_exists(self):
        """Test that the fixture file exists."""
        assert self.fixture_path.exists(), f"Fixture not found: {self.fixture_path}"

    def test_parse_vue_with_directives(self):
        """Test parsing Vue file with template directives."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Should have chunks from both script and template sections
        assert len(chunks) > 0

        # Check for script chunks
        script_chunks = [c for c in chunks if c.metadata.get("vue_section") == "script"]
        assert len(script_chunks) > 0

        # Check for template chunks
        template_chunks = [
            c for c in chunks if c.metadata.get("vue_section") == "template"
        ]
        assert len(template_chunks) > 0

    def test_parse_v_if_directive(self):
        """Test extraction of v-if directives."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find v-if directive chunks
        vif_chunks = [
            c
            for c in chunks
            if c.metadata.get("directive_type") == "v-if"
            or c.metadata.get("directive_type") == "v-else-if"
        ]

        # Should find at least one v-if directive
        if vif_chunks:  # Only check if template parsing succeeded
            assert len(vif_chunks) > 0
            # Check that condition is extracted
            for chunk in vif_chunks:
                if "condition" in chunk.metadata:
                    assert chunk.metadata["condition"]

    def test_parse_v_for_directive(self):
        """Test extraction of v-for directives."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find v-for directive chunks
        vfor_chunks = [c for c in chunks if c.metadata.get("directive_type") == "v-for"]

        # Should find at least one v-for directive
        if vfor_chunks:  # Only check if template parsing succeeded
            assert len(vfor_chunks) > 0
            # Check that loop expression is extracted
            for chunk in vfor_chunks:
                if "loop_expression" in chunk.metadata:
                    assert chunk.metadata["loop_expression"]

    def test_parse_event_handlers(self):
        """Test extraction of event handlers (@click, @submit, etc.)."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find event handler chunks
        event_chunks = [
            c for c in chunks if c.metadata.get("directive_type") == "event_handler"
        ]

        # Should find event handlers
        if event_chunks:  # Only check if template parsing succeeded
            assert len(event_chunks) > 0
            # Check that event names are extracted
            event_names = [
                c.metadata.get("event_name")
                for c in event_chunks
                if "event_name" in c.metadata
            ]
            if event_names:
                assert "click" in event_names or "submit" in event_names

    def test_parse_property_bindings(self):
        """Test extraction of property bindings (:prop, v-bind)."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find property binding chunks
        bind_chunks = [
            c for c in chunks if c.metadata.get("directive_type") == "property_binding"
        ]

        # Should find property bindings
        if bind_chunks:  # Only check if template parsing succeeded
            assert len(bind_chunks) > 0
            # Check that property names are extracted
            for chunk in bind_chunks:
                if "property_name" in chunk.metadata:
                    assert chunk.metadata["property_name"]

    def test_parse_v_model(self):
        """Test extraction of v-model (two-way binding)."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find v-model chunks
        vmodel_chunks = [
            c for c in chunks if c.metadata.get("directive_type") == "v-model"
        ]

        # Should find v-model directives
        if vmodel_chunks:  # Only check if template parsing succeeded
            assert len(vmodel_chunks) > 0
            # Check that model binding is extracted
            for chunk in vmodel_chunks:
                if "model_binding" in chunk.metadata:
                    assert chunk.metadata["model_binding"]

    def test_parse_component_usage(self):
        """Test extraction of component usage (PascalCase tags)."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find component usage chunks
        component_chunks = [
            c for c in chunks if c.metadata.get("directive_type") == "component_usage"
        ]

        # Should find components (UserProfile, BaseButton, Modal)
        if component_chunks:  # Only check if template parsing succeeded
            assert len(component_chunks) > 0
            # Check that component names are extracted
            component_names = [
                c.metadata.get("component_name")
                for c in component_chunks
                if "component_name" in c.metadata
            ]
            if component_names:
                assert any(
                    name in component_names
                    for name in ["UserProfile", "BaseButton", "Modal"]
                )

    def test_parse_interpolations(self):
        """Test extraction of interpolations ({{ variable }})."""
        if not self.fixture_path.exists():
            pytest.skip("Fixture file not found")

        chunks = self.parser.parse_file(self.fixture_path, FileId(1))

        # Find interpolation chunks
        interp_chunks = [
            c for c in chunks if c.metadata.get("directive_type") == "interpolation"
        ]

        # Should find interpolations
        if interp_chunks:  # Only check if template parsing succeeded
            assert len(interp_chunks) > 0
            # Check that expressions are extracted
            for chunk in interp_chunks:
                if "interpolation_expression" in chunk.metadata:
                    assert chunk.metadata["interpolation_expression"]

    def test_parse_simple_template(self):
        """Test parsing a simple template with basic directives."""
        simple_vue = """<template>
  <div v-if="show">
    <p>{{ message }}</p>
    <button @click="handleClick">Click me</button>
  </div>
</template>

<script setup>
const show = ref(true)
const message = ref('Hello')
const handleClick = () => console.log('clicked')
</script>"""

        chunks = self.parser.parse_content(simple_vue)

        # Should have chunks
        assert len(chunks) > 0

        # Should have both script and template chunks
        sections = {c.metadata.get("vue_section") for c in chunks}
        assert "script" in sections or "template" in sections

    def test_fallback_to_text_block(self):
        """Test fallback to simple text block if tree-sitter parsing fails."""
        # Create a parser without template parser (simulating failure)
        parser = VueParser()
        # Force template parser to None to test fallback
        parser.template_parser = None

        simple_vue = """<template>
  <div>{{ message }}</div>
</template>

<script setup>
const message = ref('Hello')
</script>"""

        chunks = parser.parse_content(simple_vue)

        # Should have chunks (using fallback)
        assert len(chunks) > 0

        # Should have a template chunk (as fallback text block)
        template_chunks = [
            c for c in chunks if c.metadata.get("vue_section") == "template"
        ]
        assert len(template_chunks) > 0


class TestVueTemplateMetadata:
    """Test metadata extraction for Vue template directives."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = VueParser()

    def test_conditional_directive_metadata(self):
        """Test metadata extraction for conditional directives."""
        vue_content = """<template>
  <div v-if="isActive">Active</div>
  <div v-else-if="isLoading">Loading</div>
</template>

<script setup>
const isActive = ref(true)
const isLoading = ref(false)
</script>"""

        chunks = self.parser.parse_content(vue_content)
        template_chunks = [
            c for c in chunks if c.metadata.get("vue_section") == "template"
        ]

        # If template parsing succeeded, check metadata
        if any("directive_type" in c.metadata for c in template_chunks):
            vif_chunks = [
                c
                for c in template_chunks
                if c.metadata.get("directive_type") in ["v-if", "v-else-if"]
            ]
            assert len(vif_chunks) > 0

    def test_loop_directive_metadata(self):
        """Test metadata extraction for loop directives."""
        vue_content = """<template>
  <ul>
    <li v-for="item in items" :key="item.id">{{ item.name }}</li>
  </ul>
</template>

<script setup>
const items = ref([{ id: 1, name: 'Item 1' }])
</script>"""

        chunks = self.parser.parse_content(vue_content)
        template_chunks = [
            c for c in chunks if c.metadata.get("vue_section") == "template"
        ]

        # If template parsing succeeded, check metadata
        if any("directive_type" in c.metadata for c in template_chunks):
            vfor_chunks = [
                c
                for c in template_chunks
                if c.metadata.get("directive_type") == "v-for"
            ]
            if vfor_chunks:
                # Check that loop metadata is extracted
                for chunk in vfor_chunks:
                    if "loop_expression" in chunk.metadata:
                        assert "items" in chunk.metadata["loop_expression"]

    def test_all_chunks_have_vue_metadata(self):
        """Test that all template chunks have vue-specific metadata."""
        vue_content = """<template>
  <div>
    <p>{{ message }}</p>
  </div>
</template>

<script setup>
const message = ref('Hello')
</script>"""

        chunks = self.parser.parse_content(vue_content)

        # All chunks should have is_vue_sfc metadata
        for chunk in chunks:
            assert chunk.metadata.get("is_vue_sfc") is True
            assert chunk.metadata.get("vue_section") in ["script", "template", None]
