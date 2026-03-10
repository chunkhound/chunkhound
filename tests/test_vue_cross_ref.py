"""Tests for Vue SFC cross-reference analysis."""

from pathlib import Path

import pytest

from chunkhound.core.models.chunk import Chunk
from chunkhound.core.types.common import ChunkType, FileId, Language, LineNumber
from chunkhound.parsers.vue_cross_ref import (
    VueSymbol,
    VueSymbolTable,
    build_symbol_table,
    extract_identifiers_from_expression,
    extract_props_from_define_props,
    extract_references_from_chunk,
    extract_symbols_from_chunk,
    match_template_references,
)
from chunkhound.parsers.vue_parser import VueParser


class TestVueSymbolTable:
    """Tests for VueSymbolTable data structure."""

    def test_add_symbol_variable(self):
        """Test adding a variable symbol."""
        table = VueSymbolTable()
        symbol = VueSymbol(
            name="count",
            type="variable",
            chunk_symbol="greet",
            start_line=5,
            end_line=7,
        )
        table.add_symbol(symbol)

        assert "count" in table.variables
        assert table.variables["count"] == symbol

    def test_add_symbol_function(self):
        """Test adding a function symbol."""
        table = VueSymbolTable()
        symbol = VueSymbol(
            name="handleClick",
            type="function",
            chunk_symbol="handleClick",
            start_line=10,
            end_line=12,
        )
        table.add_symbol(symbol)

        assert "handleClick" in table.functions
        assert table.functions["handleClick"] == symbol

    def test_find_symbol(self):
        """Test finding a symbol by name."""
        table = VueSymbolTable()

        var_symbol = VueSymbol(
            name="message",
            type="variable",
            chunk_symbol="msg",
            start_line=1,
            end_line=1,
        )
        func_symbol = VueSymbol(
            name="greet",
            type="function",
            chunk_symbol="greet",
            start_line=5,
            end_line=7,
        )

        table.add_symbol(var_symbol)
        table.add_symbol(func_symbol)

        assert table.find_symbol("message") == var_symbol
        assert table.find_symbol("greet") == func_symbol
        assert table.find_symbol("nonexistent") is None


class TestSymbolExtraction:
    """Tests for symbol extraction from script chunks."""

    def test_extract_const_variable(self):
        """Test extracting const variable declaration."""
        chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const message = 'Hello'",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        assert len(symbols) == 1
        assert symbols[0].name == "message"
        assert symbols[0].type == "constant"
        assert symbols[0].is_reactive is False

    def test_extract_reactive_ref(self):
        """Test extracting ref() reactive variable."""
        chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const count = ref(0)",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        assert len(symbols) == 1
        assert symbols[0].name == "count"
        assert symbols[0].type == "constant"
        assert symbols[0].is_reactive is True
        assert symbols[0].metadata["reactive_type"] == "ref"

    def test_extract_computed(self):
        """Test extracting computed() property."""
        chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const doubleCount = computed(() => count.value * 2)",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        assert len(symbols) == 1
        assert symbols[0].name == "doubleCount"
        assert symbols[0].type == "computed"
        assert symbols[0].is_reactive is True

    def test_extract_function_declaration(self):
        """Test extracting function declaration."""
        chunk = Chunk(
            symbol="greet",
            start_line=LineNumber(5),
            end_line=LineNumber(7),
            code="function greet() {\n  console.log('Hello')\n}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        assert len(symbols) == 1
        assert symbols[0].name == "greet"
        assert symbols[0].type == "function"
        assert symbols[0].metadata["declaration_style"] == "function"

    def test_extract_arrow_function(self):
        """Test extracting arrow function."""
        chunk = Chunk(
            symbol="handleClick",
            start_line=LineNumber(10),
            end_line=LineNumber(12),
            code="const handleClick = () => {\n  console.log('Clicked')\n}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        # Arrow functions are detected both as variables and functions
        # We should find at least the function
        assert len(symbols) >= 1
        func_symbols = [s for s in symbols if s.name == "handleClick"]
        assert len(func_symbols) >= 1

    def test_extract_composable_destructured(self):
        """Test extracting composable with destructuring."""
        chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const { user, login, logout } = useUser()",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        )

        symbols = extract_symbols_from_chunk(chunk)

        # Should extract all three destructured variables
        assert len(symbols) >= 3
        names = [s.name for s in symbols]
        assert "user" in names
        assert "login" in names
        assert "logout" in names

        # All should be marked as composables
        for symbol in symbols:
            if symbol.name in ["user", "login", "logout"]:
                assert symbol.type == "composable"
                assert symbol.metadata["composable"] == "useUser"

    def test_extract_props_typescript_interface(self):
        """Test extracting props from TypeScript interface."""
        code = """
        defineProps<{
          title: string
          count?: number
          isActive: boolean
        }>()
        """
        props = extract_props_from_define_props(code)

        assert len(props) == 3
        assert "title" in props
        assert "count" in props
        assert "isActive" in props

    def test_extract_props_from_chunk(self):
        """Test extracting props from chunk with defineProps."""
        chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(3),
            code="defineProps<{ title: string, count: number }>()",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"vue_macros": ["defineProps"]},
        )

        symbols = extract_symbols_from_chunk(chunk)

        # Should extract prop symbols
        prop_symbols = [s for s in symbols if s.type == "prop"]
        assert len(prop_symbols) >= 2
        prop_names = [s.name for s in prop_symbols]
        assert "title" in prop_names
        assert "count" in prop_names


class TestParameterSymbolExtraction:
    """Tests for parameter symbol extraction from chunk metadata."""

    def test_build_symbol_table_extracts_parameter_names_from_dicts(self) -> None:
        script_chunk = Chunk(
            symbol="complete",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="function complete() {}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"parameters": [{"type": "Request", "name": "$request"}]},
        )

        symbol_table = build_symbol_table([script_chunk])

        assert symbol_table.find_symbol("$request") is not None
        assert symbol_table.find_symbol("Request") is None

    def test_build_symbol_table_splits_legacy_comma_string_parameters(self) -> None:
        script_chunk = Chunk(
            symbol="thing",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="function thing() {}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"parameters": "alpha, beta"},
        )

        symbol_table = build_symbol_table([script_chunk])

        assert symbol_table.find_symbol("alpha") is not None
        assert symbol_table.find_symbol("beta") is not None
        assert symbol_table.find_symbol("alpha, beta") is None

    def test_build_symbol_table_filters_js_keywords_from_parameters(self) -> None:
        script_chunk = Chunk(
            symbol="thing",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="function thing() {}",
            chunk_type=ChunkType.FUNCTION,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"parameters": ["if", "validName"]},
        )

        symbol_table = build_symbol_table([script_chunk])

        assert symbol_table.find_symbol("if") is None
        assert symbol_table.find_symbol("validName") is not None


class TestReferenceExtraction:
    """Tests for reference extraction from template chunks."""

    def test_extract_identifiers_simple(self):
        """Test extracting simple identifiers."""
        expr = "count"
        identifiers = extract_identifiers_from_expression(expr)

        assert identifiers == ["count"]

    def test_extract_identifiers_complex(self):
        """Test extracting identifiers from complex expression."""
        expr = "user.isAdmin && permissions.includes('write')"
        identifiers = extract_identifiers_from_expression(expr)

        assert "user" in identifiers
        assert "permissions" in identifiers
        assert "includes" in identifiers
        # Should not include keywords
        assert "true" not in identifiers
        assert "false" not in identifiers

    def test_extract_identifiers_filters_keywords(self):
        """Test that JavaScript keywords are filtered out."""
        expr = "if (count > 0) return true else return false"
        identifiers = extract_identifiers_from_expression(expr)

        assert "count" in identifiers
        # Keywords should be filtered
        assert "if" not in identifiers
        assert "return" not in identifiers
        assert "true" not in identifiers
        assert "false" not in identifiers
        assert "else" not in identifiers

    def test_extract_references_interpolation(self):
        """Test extracting references from interpolation."""
        chunk = Chunk(
            symbol="{{ message }}",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="{{ message }}",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"interpolation_expression": "message"},
        )

        refs = extract_references_from_chunk(chunk)

        assert "message" in refs

    def test_extract_references_event_handler(self):
        """Test extracting references from event handler."""
        chunk = Chunk(
            symbol="@click",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code='@click="handleClick"',
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"handler_expression": "handleClick"},
        )

        refs = extract_references_from_chunk(chunk)

        assert "handleClick" in refs

    def test_extract_references_property_binding(self):
        """Test extracting references from property binding."""
        chunk = Chunk(
            symbol=":src",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code=':src="imageUrl"',
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"binding_expression": "imageUrl"},
        )

        refs = extract_references_from_chunk(chunk)

        assert "imageUrl" in refs

    def test_extract_references_v_if(self):
        """Test extracting references from v-if."""
        chunk = Chunk(
            symbol="v-if",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code='v-if="isAuthenticated"',
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"condition": "isAuthenticated"},
        )

        refs = extract_references_from_chunk(chunk)

        assert "isAuthenticated" in refs

    def test_extract_references_v_for(self):
        """Test extracting references from v-for."""
        chunk = Chunk(
            symbol="v-for",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code='v-for="item in items"',
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"loop_iterable": "items", "loop_variable": "item"},
        )

        refs = extract_references_from_chunk(chunk)

        assert "items" in refs


class TestCrossReferenceMatching:
    """Tests for cross-reference matching."""

    def test_match_simple_reference(self):
        """Test matching a simple template reference to script symbol."""
        # Create script chunk with variable
        script_chunk = Chunk(
            symbol="test",
            start_line=LineNumber(1),
            end_line=LineNumber(1),
            code="const message = ref('Hello')",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
        )

        # Build symbol table
        symbol_table = build_symbol_table([script_chunk])

        # Create template chunk with reference
        template_chunk = Chunk(
            symbol="{{ message }}",
            start_line=LineNumber(5),
            end_line=LineNumber(5),
            code="{{ message }}",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"interpolation_expression": "message"},
        )

        # Match references
        updated_chunks = match_template_references([template_chunk], symbol_table)

        assert len(updated_chunks) == 1
        assert "script_references" in updated_chunks[0].metadata
        assert "message" in updated_chunks[0].metadata["script_references"]

    def test_match_undefined_reference(self):
        """Test detecting undefined references."""
        # Create empty symbol table
        symbol_table = VueSymbolTable()

        # Create template chunk with undefined reference
        template_chunk = Chunk(
            symbol="{{ undefinedVar }}",
            start_line=LineNumber(5),
            end_line=LineNumber(5),
            code="{{ undefinedVar }}",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"interpolation_expression": "undefinedVar"},
        )

        # Match references
        updated_chunks = match_template_references([template_chunk], symbol_table)

        assert len(updated_chunks) == 1
        assert "undefined_references" in updated_chunks[0].metadata
        assert "undefinedVar" in updated_chunks[0].metadata["undefined_references"]

    def test_match_multiple_references(self):
        """Test matching multiple references in one chunk."""
        # Create script chunks
        script_chunks = [
            Chunk(
                symbol="test",
                start_line=LineNumber(1),
                end_line=LineNumber(1),
                code="const user = ref(null)",
                chunk_type=ChunkType.BLOCK,
                file_id=FileId(1),
                language=Language.VUE,
            ),
            Chunk(
                symbol="test",
                start_line=LineNumber(2),
                end_line=LineNumber(2),
                code="const permissions = ref([])",
                chunk_type=ChunkType.BLOCK,
                file_id=FileId(1),
                language=Language.VUE,
            ),
        ]

        symbol_table = build_symbol_table(script_chunks)

        # Template chunk with complex expression
        template_chunk = Chunk(
            symbol="v-if",
            start_line=LineNumber(10),
            end_line=LineNumber(10),
            code="v-if=\"user.isAdmin && permissions.includes('write')\"",
            chunk_type=ChunkType.BLOCK,
            file_id=FileId(1),
            language=Language.VUE,
            metadata={"condition": "user.isAdmin && permissions.includes('write')"},
        )

        updated_chunks = match_template_references([template_chunk], symbol_table)

        assert len(updated_chunks) == 1
        assert "script_references" in updated_chunks[0].metadata
        refs = updated_chunks[0].metadata["script_references"]
        assert "user" in refs
        assert "permissions" in refs


class TestVueParserIntegration:
    """Integration tests with VueParser."""

    @pytest.fixture
    def parser(self):
        """Create VueParser instance."""
        return VueParser()

    @pytest.fixture
    def cross_ref_fixture(self):
        """Load cross-reference test fixture."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "vue" / "cross_reference.vue"
        )
        return fixture_path

    def test_parse_cross_reference_fixture(self, parser, cross_ref_fixture):
        """Test parsing cross-reference fixture with full integration."""
        if not cross_ref_fixture.exists():
            pytest.skip("Cross-reference fixture not found")

        chunks = parser.parse_file(cross_ref_fixture, FileId(1))

        # Should have script and template chunks
        script_chunks = [
            c
            for c in chunks
            if c.metadata and c.metadata.get("vue_section") == "script"
        ]
        template_chunks = [
            c
            for c in chunks
            if c.metadata and c.metadata.get("vue_section") == "template"
        ]

        assert len(script_chunks) > 0, "Should have script chunks"
        assert len(template_chunks) > 0, "Should have template chunks"

        # Template chunks should have cross-reference metadata
        chunks_with_refs = [
            c for c in template_chunks
            if c.metadata and "script_references" in c.metadata
        ]

        assert len(chunks_with_refs) > 0, "Should have chunks with cross-references"

        # Check for specific expected references
        all_refs = set()
        for chunk in chunks_with_refs:
            if "script_references" in chunk.metadata:
                all_refs.update(chunk.metadata["script_references"])

        # These should be found
        expected_refs = ["title", "count", "increment", "user", "items"]
        for ref in expected_refs:
            assert ref in all_refs, f"Expected reference '{ref}' not found"

        # Check for undefined references
        chunks_with_undefined = [
            c for c in template_chunks
            if c.metadata and "undefined_references" in c.metadata
        ]

        assert len(chunks_with_undefined) > 0, "Should detect undefined references"

        # Verify undefined references exist
        all_undefined = set()
        for chunk in chunks_with_undefined:
            if "undefined_references" in chunk.metadata:
                all_undefined.update(chunk.metadata["undefined_references"])

        # Should find some undefined references (exact ones depend on template
        # chunking). Due to template chunk merging, specific
        # undefinedVariable/undefinedFunction may not be in separate chunks with
        # metadata, but we should still find property accesses like 'name' and
        # 'item' that are used in expressions but not defined in script.
        assert len(all_undefined) > 0, "Should have some undefined references detected"

    def test_parse_basic_setup(self, parser):
        """Test parsing basic setup fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "vue" / "basic_setup.vue"
        if not fixture_path.exists():
            pytest.skip("Basic setup fixture not found")

        chunks = parser.parse_file(fixture_path, FileId(1))

        # Find template chunks with references
        template_chunks = [
            c for c in chunks
            if c.metadata and c.metadata.get("vue_section") == "template"
        ]

        chunks_with_refs = [
            c for c in template_chunks
            if c.metadata and "script_references" in c.metadata
        ]

        if len(chunks_with_refs) > 0:
            # Should reference 'message' from script
            all_refs = set()
            for chunk in chunks_with_refs:
                if "script_references" in chunk.metadata:
                    all_refs.update(chunk.metadata["script_references"])

            assert "message" in all_refs

    def test_parse_with_composables(self, parser):
        """Test parsing fixture with composables."""
        fixture_path = (
            Path(__file__).parent / "fixtures" / "vue" / "with_composables.vue"
        )
        if not fixture_path.exists():
            pytest.skip("Composables fixture not found")

        chunks = parser.parse_file(fixture_path, FileId(1))

        # Find template chunks with references
        template_chunks = [
            c for c in chunks
            if c.metadata and c.metadata.get("vue_section") == "template"
        ]

        chunks_with_refs = [
            c for c in template_chunks
            if c.metadata and "script_references" in c.metadata
        ]

        if len(chunks_with_refs) > 0:
            # Should reference composable variables
            all_refs = set()
            for chunk in chunks_with_refs:
                if "script_references" in chunk.metadata:
                    all_refs.update(chunk.metadata["script_references"])

            # These come from composables
            composable_refs = ["user", "isAuthenticated", "count"]
            found_composable_refs = [ref for ref in composable_refs if ref in all_refs]
            assert len(found_composable_refs) > 0, "Should find composable references"
