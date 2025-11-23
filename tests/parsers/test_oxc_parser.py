"""Tests for OxcParser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import Language, FileId, ChunkType

# Import OXC_AVAILABLE flag to conditionally skip tests
from chunkhound.parsers.parser_factory import OXC_AVAILABLE


# Marker for tests requiring OXC
requires_oxc = pytest.mark.skipif(
    not OXC_AVAILABLE,
    reason="oxc_python not available or incompatible API"
)


@pytest.fixture
def js_parser():
    """Create a JavaScript OxcParser."""
    if not OXC_AVAILABLE:
        pytest.skip("oxc_python not available")
    from chunkhound.parsers.oxc_parser import OxcParser
    return OxcParser(Language.JAVASCRIPT)


@pytest.fixture
def ts_parser():
    """Create a TypeScript OxcParser."""
    if not OXC_AVAILABLE:
        pytest.skip("oxc_python not available")
    from chunkhound.parsers.oxc_parser import OxcParser
    return OxcParser(Language.TYPESCRIPT)


@pytest.fixture
def jsx_parser():
    """Create a JSX OxcParser."""
    if not OXC_AVAILABLE:
        pytest.skip("oxc_python not available")
    from chunkhound.parsers.oxc_parser import OxcParser
    return OxcParser(Language.JSX)


@pytest.fixture
def tsx_parser():
    """Create a TSX OxcParser."""
    if not OXC_AVAILABLE:
        pytest.skip("oxc_python not available")
    from chunkhound.parsers.oxc_parser import OxcParser
    return OxcParser(Language.TSX)


@requires_oxc
class TestOxcParserBasics:
    """Basic parsing tests."""

    def test_parse_simple_function(self, js_parser):
        chunks = js_parser.parse_content("function foo() { return 42; }")

        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].symbol == "foo"

    def test_parse_class(self, js_parser):
        code = """
        class MyClass {
            constructor() {}
            myMethod() { return 1; }
        }
        """
        chunks = js_parser.parse_content(code)

        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].symbol == "MyClass"

    def test_parse_typescript(self, ts_parser):
        code = """
        interface User {
            name: string;
            age: number;
        }

        type Status = 'active' | 'inactive';

        function greet(user: User): string {
            return `Hello, ${user.name}`;
        }
        """
        chunks = ts_parser.parse_content(code)

        interface_chunks = [c for c in chunks if c.chunk_type == ChunkType.INTERFACE]
        assert len(interface_chunks) == 1

        type_chunks = [c for c in chunks if c.chunk_type == ChunkType.TYPE_ALIAS]
        assert len(type_chunks) == 1

    def test_parse_jsx(self, jsx_parser):
        code = """
        function App() {
            return <div>Hello World</div>;
        }
        """
        chunks = jsx_parser.parse_content(code)

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].symbol == "App"

    def test_parse_tsx(self, tsx_parser):
        code = """
        interface Props {
            name: string;
        }

        const Greeting: React.FC<Props> = ({ name }) => {
            return <h1>Hello, {name}!</h1>;
        };
        """
        chunks = tsx_parser.parse_content(code)

        assert len(chunks) >= 2  # Interface + variable/arrow function

    def test_parse_empty_content(self, js_parser):
        chunks = js_parser.parse_content("")
        assert chunks == []

    def test_parse_whitespace_only(self, js_parser):
        chunks = js_parser.parse_content("   \n\n   ")
        assert chunks == []


@requires_oxc
class TestOxcParserNodeTypes:
    """Test specific node type extraction."""

    def test_arrow_function(self, js_parser):
        code = "const add = (a, b) => a + b;"
        chunks = js_parser.parse_content(code)

        # Should extract the variable declaration
        var_chunks = [c for c in chunks if c.chunk_type == ChunkType.VARIABLE]
        assert len(var_chunks) >= 1

    def test_method_definition(self, js_parser):
        code = """
        class Calculator {
            add(a, b) { return a + b; }
            subtract(a, b) { return a - b; }
        }
        """
        chunks = js_parser.parse_content(code)

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(method_chunks) == 2

    def test_ts_enum(self, ts_parser):
        code = """
        enum Color {
            Red,
            Green,
            Blue
        }
        """
        chunks = ts_parser.parse_content(code)

        enum_chunks = [c for c in chunks if c.chunk_type == ChunkType.ENUM]
        assert len(enum_chunks) == 1

    def test_import_declaration(self, js_parser):
        code = "import { foo, bar } from './utils';"
        chunks = js_parser.parse_content(code)

        # ImportDeclaration maps to UNKNOWN
        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.UNKNOWN]
        assert len(import_chunks) >= 1

    def test_export_declaration(self, js_parser):
        code = "export function myExport() { return 1; }"
        chunks = js_parser.parse_content(code)

        # Should have the export with function
        assert len(chunks) >= 1


@requires_oxc
class TestOxcParserMetadata:
    """Test metadata extraction."""

    def test_async_function_metadata(self, js_parser):
        code = "async function fetchData() { return await fetch('/api'); }"
        chunks = js_parser.parse_content(code)

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata.get("is_async") is True

    def test_generator_function_metadata(self, js_parser):
        code = "function* generator() { yield 1; yield 2; }"
        chunks = js_parser.parse_content(code)

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata.get("is_generator") is True

    def test_line_numbers(self, js_parser):
        code = """line1
        function foo() {
            return 42;
        }
        line5"""
        chunks = js_parser.parse_content(code)

        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        # Function starts on line 2
        assert func_chunks[0].start_line == 2


@requires_oxc
class TestOxcParserFactoryIntegration:
    """Integration tests with ParserFactory."""

    def test_factory_returns_correct_parser_type(self):
        """Test that factory returns OxcParser when available, else UniversalParser."""
        from chunkhound.parsers.parser_factory import ParserFactory, OXC_AVAILABLE

        factory = ParserFactory()

        for lang in [Language.JAVASCRIPT, Language.TYPESCRIPT, Language.JSX, Language.TSX]:
            parser = factory.create_parser(lang)
            if OXC_AVAILABLE:
                from chunkhound.parsers.oxc_parser import OxcParser
                assert isinstance(parser, OxcParser), f"Expected OxcParser for {lang}, got {type(parser)}"
            else:
                # Falls back to tree-sitter UniversalParser
                from chunkhound.parsers.universal_parser import UniversalParser
                assert isinstance(parser, UniversalParser), f"Expected UniversalParser fallback for {lang}"


@requires_oxc
class TestOxcParserEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_syntax_error(self, js_parser):
        """Parser should handle syntax errors gracefully."""
        # Malformed JS - missing closing brace
        code = "function broken() { return 1; "
        # Should not raise, but may return partial results or empty
        chunks = js_parser.parse_content(code)
        # Just verify it doesn't crash
        assert isinstance(chunks, list)

    def test_parse_unicode(self, js_parser):
        """Parser should handle unicode content."""
        code = 'function greet() { return "Hello, "; }'
        chunks = js_parser.parse_content(code)
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1

    def test_language_property(self, js_parser, ts_parser):
        """Test language property returns correct language."""
        assert js_parser.language == Language.JAVASCRIPT
        assert ts_parser.language == Language.TYPESCRIPT

    def test_supported_extensions(self, js_parser, ts_parser, jsx_parser, tsx_parser):
        """Test supported_extensions returns correct extensions."""
        assert ".js" in js_parser.supported_extensions
        assert ".ts" in ts_parser.supported_extensions
        assert ".jsx" in jsx_parser.supported_extensions
        assert ".tsx" in tsx_parser.supported_extensions

    def test_is_initialized(self, js_parser):
        """Test is_initialized property."""
        assert js_parser.is_initialized is True

    def test_can_parse_file(self, js_parser, tmp_path):
        """Test can_parse_file method."""
        js_file = tmp_path / "test.js"
        js_file.write_text("const x = 1;")
        assert js_parser.can_parse_file(js_file) is True

        py_file = tmp_path / "test.py"
        py_file.write_text("x = 1")
        assert js_parser.can_parse_file(py_file) is False

    def test_validate_syntax(self, js_parser):
        """Test syntax validation."""
        # Valid code
        errors = js_parser.validate_syntax("function foo() { return 1; }")
        assert errors == []

        # Invalid code
        errors = js_parser.validate_syntax("function foo() { return ")
        assert len(errors) > 0


# Test that runs regardless of OXC availability
class TestOxcAvailabilityFlag:
    """Test that OXC_AVAILABLE flag works correctly."""

    def test_oxc_available_is_boolean(self):
        """Verify OXC_AVAILABLE is a boolean."""
        assert isinstance(OXC_AVAILABLE, bool)

    def test_factory_handles_oxc_unavailable(self):
        """Factory should work regardless of OXC availability."""
        from chunkhound.parsers.parser_factory import ParserFactory

        factory = ParserFactory()
        # Should not raise - either returns OxcParser or falls back to UniversalParser
        parser = factory.create_parser(Language.JAVASCRIPT)
        assert parser is not None
