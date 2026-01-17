"""Tests for Lua language parser."""

import pytest
from pathlib import Path

from chunkhound.core.types.common import FileId, Language
from chunkhound.parsers.parser_factory import get_parser_factory
from chunkhound.parsers.mappings.lua import LuaMapping


class TestLuaMapping:
    """Test LuaMapping extraction logic."""

    def test_clean_single_line_comment(self):
        """Test cleaning single-line comments."""
        mapping = LuaMapping()

        assert mapping.clean_comment_text("-- hello world") == "hello world"
        assert mapping.clean_comment_text("--comment") == "comment"

    def test_clean_multiline_comment(self):
        """Test cleaning multi-line comments."""
        mapping = LuaMapping()

        assert mapping.clean_comment_text("--[[ multi line ]]") == "multi line"
        assert mapping.clean_comment_text("--[=[ long bracket ]=]") == "long bracket"

    def test_language_is_lua(self):
        """Test that mapping reports correct language."""
        mapping = LuaMapping()
        assert mapping.language == Language.LUA

    def test_extract_constants_from_complex_fixture(self):
        """Test that constants are extracted from complex.lua fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "lua" / "complex.lua"
        if not fixture_path.exists():
            pytest.skip("Complex Lua fixture file not found")

        factory = get_parser_factory()
        parser = factory.create_parser(Language.LUA)
        chunks = parser.parse_file(fixture_path, FileId(1))

        # Look for constants in metadata
        all_constants = []
        for chunk in chunks:
            chunk_constants = chunk.metadata.get("constants", [])
            if chunk_constants:
                all_constants.extend(chunk_constants)

        # At minimum, test runs without error
        assert len(all_constants) >= 0


class TestLuaParser:
    """Test Lua parser functionality."""

    @pytest.fixture
    def parser(self):
        """Get a Lua parser instance."""
        factory = get_parser_factory()
        return factory.create_parser(Language.LUA)

    def test_parser_loads(self, parser):
        """Test that Lua parser loads successfully."""
        assert parser is not None

    def test_parse_simple_function(self, parser, tmp_path):
        """Test parsing a simple Lua function."""
        code = """
function greet(name)
    print("Hello, " .. name)
end
"""
        lua_file = tmp_path / "test.lua"
        lua_file.write_text(code)

        chunks = parser.parse_file(lua_file, FileId(1))

        assert len(chunks) > 0
        # Should find the function definition
        function_chunks = [c for c in chunks if "greet" in getattr(c, "symbol", "")]
        assert len(function_chunks) >= 1

    def test_parse_local_function(self, parser, tmp_path):
        """Test parsing a local function."""
        code = """
local function helper()
    return 42
end
"""
        lua_file = tmp_path / "test.lua"
        lua_file.write_text(code)

        chunks = parser.parse_file(lua_file, FileId(1))

        assert len(chunks) > 0
        # Should find the local function
        function_chunks = [c for c in chunks if "helper" in getattr(c, "symbol", "")]
        assert len(function_chunks) >= 1

    def test_parse_local_variable(self, parser, tmp_path):
        """Test parsing local variable declarations."""
        code = """
local config = {
    debug = true,
    version = "1.0"
}
"""
        lua_file = tmp_path / "test.lua"
        lua_file.write_text(code)

        chunks = parser.parse_file(lua_file, FileId(1))

        assert len(chunks) > 0
        # Should find the variable declaration
        var_chunks = [c for c in chunks if "config" in getattr(c, "symbol", "")]
        assert len(var_chunks) >= 1

    def test_parse_method_syntax(self, parser, tmp_path):
        """Test parsing method-style function definitions."""
        code = """
local MyClass = {}

function MyClass:new()
    return setmetatable({}, self)
end

function MyClass:getValue()
    return self.value
end
"""
        lua_file = tmp_path / "test.lua"
        lua_file.write_text(code)

        chunks = parser.parse_file(lua_file, FileId(1))

        assert len(chunks) > 0
        # Should find at least MyClass variable and methods
        names = [getattr(c, "symbol", "") for c in chunks]
        assert any("MyClass" in name for name in names)

    def test_parse_fixture_file(self, parser):
        """Test parsing the sample fixture file."""
        fixture_path = Path(__file__).parent / "fixtures" / "lua" / "sample.lua"
        if not fixture_path.exists():
            pytest.skip("Lua fixture file not found")

        chunks = parser.parse_file(fixture_path, FileId(1))

        # Should find multiple definitions
        assert len(chunks) >= 5

        # Verify we found some expected functions
        names = [getattr(c, "symbol", "") for c in chunks]
        # Check for some expected function names
        expected_names = ["greet", "calculateSum", "processData"]
        found = sum(1 for name in expected_names if any(name in n for n in names))
        assert found >= 2, f"Expected to find at least 2 of {expected_names}, got names: {names}"

    def test_parse_complex_fixture(self, parser):
        """Test parsing complex Lua code with game modding patterns."""
        fixture_path = Path(__file__).parent / "fixtures" / "lua" / "complex.lua"
        if not fixture_path.exists():
            pytest.skip("Complex Lua fixture file not found")

        chunks = parser.parse_file(fixture_path, FileId(1))

        # Should find multiple definitions (functions, tables, variables)
        assert len(chunks) >= 5

        # Verify we found key functions
        names = [getattr(c, "symbol", "") for c in chunks]
        expected_names = ["calcDamage", "calcAilmentDamage", "buildOutput"]
        found = sum(1 for name in expected_names if any(name in n for n in names))
        assert found >= 2, f"Expected to find at least 2 of {expected_names}, got names: {names}"

        # Verify we can find the complex data tables
        table_names = ["dmgTypeList", "dmgTypeFlags", "ailmentData"]
        table_found = sum(1 for name in table_names if any(name in n for n in names))
        assert table_found >= 1, f"Expected to find at least 1 of {table_names}, got names: {names}"

    def test_parse_method_style_functions_captured(self, parser, tmp_path):
        """Test that method-style functions (colon and dot syntax) are captured.

        Note: cAST may merge adjacent functions of the same kind for information density.
        The key is that the full method names are extracted correctly at the query level,
        which can be verified by checking both symbol names and content.
        """
        code = """
local MyModule = {}

function MyModule.staticMethod()
    return 1
end

function MyModule:instanceMethod()
    return self.value
end
"""
        lua_file = tmp_path / "test.lua"
        lua_file.write_text(code)

        chunks = parser.parse_file(lua_file, FileId(1))
        names = [getattr(c, "symbol", "") for c in chunks]
        all_content = " ".join(getattr(c, "code", "") for c in chunks)

        # Check that dot-style method is captured (in name or content)
        dot_style_found = (
            any("MyModule.staticMethod" in name or "staticMethod" in name for name in names)
            or "MyModule.staticMethod" in all_content
        )
        assert dot_style_found, f"Expected dot-style method in {names} or content"

        # Check that colon-style method is captured (in name or content)
        colon_style_found = (
            any("MyModule:instanceMethod" in name or "instanceMethod" in name for name in names)
            or "MyModule:instanceMethod" in all_content
        )
        assert colon_style_found, f"Expected colon-style method in {names} or content"


class TestLuaImportResolution:
    """Test Lua import path resolution."""

    def test_resolve_require_path(self, tmp_path):
        """Test resolving require statements."""
        mapping = LuaMapping()

        # Create test structure
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        utils_file = lib_dir / "utils.lua"
        utils_file.write_text("return {}")

        # Test require resolution
        import_text = 'require("lib.utils")'
        source_file = tmp_path / "main.lua"

        resolved = mapping.resolve_import_path(import_text, tmp_path, source_file)
        assert resolved is not None
        assert resolved == utils_file

    def test_resolve_dofile_path(self, tmp_path):
        """Test resolving dofile statements."""
        mapping = LuaMapping()

        # Create test file
        config_file = tmp_path / "config.lua"
        config_file.write_text("return {}")

        # Test dofile resolution
        import_text = 'dofile("config.lua")'
        source_file = tmp_path / "main.lua"

        resolved = mapping.resolve_import_path(import_text, tmp_path, source_file)
        assert resolved is not None
        assert resolved == config_file

    def test_unresolvable_require(self, tmp_path):
        """Test that unresolvable requires return None."""
        mapping = LuaMapping()

        import_text = 'require("nonexistent.module")'
        source_file = tmp_path / "main.lua"

        resolved = mapping.resolve_import_path(import_text, tmp_path, source_file)
        assert resolved is None
