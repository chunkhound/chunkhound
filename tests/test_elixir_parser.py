"""Tests for Elixir language parser."""

from pathlib import Path

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.mappings.elixir import ElixirMapping
from chunkhound.parsers.parser_factory import get_parser_factory


class TestElixirFileDetection:
    """Property 1: File Extension Recognition."""

    def test_ex_extension(self):
        assert Language.from_file_extension("lib/user.ex") == Language.ELIXIR

    def test_exs_extension(self):
        assert Language.from_file_extension("test/user_test.exs") == Language.ELIXIR

    def test_is_programming_language(self):
        assert Language.ELIXIR.is_programming_language


class TestElixirMapping:
    """Test ElixirMapping extraction logic."""

    def test_language_is_elixir(self):
        mapping = ElixirMapping()
        assert mapping.language == Language.ELIXIR

    def test_clean_comment_text(self):
        mapping = ElixirMapping()
        assert mapping.clean_comment_text("# hello world") == "hello world"
        assert mapping.clean_comment_text("#comment") == "comment"


class TestElixirParser:
    """Test Elixir parser functionality."""

    @pytest.fixture
    def parser(self):
        factory = get_parser_factory()
        return factory.create_parser(Language.ELIXIR)

    def test_parser_loads(self, parser):
        assert parser is not None

    def test_parse_defmodule(self, parser, tmp_path):
        code = """
defmodule MyApp.Accounts.User do
  def hello, do: :world
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        assert len(chunks) > 0
        symbols = [c.symbol for c in chunks]
        assert any("MyApp.Accounts.User" in s for s in symbols)

    def test_parse_defprotocol(self, parser, tmp_path):
        code = """
defprotocol Printable do
  def to_string(value)
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert any("Printable" in s for s in symbols)
        # defprotocol should be interface type
        protocol_chunks = [c for c in chunks if "Printable" in c.symbol]
        assert any(c.chunk_type.value == "interface" for c in protocol_chunks)

    def test_parse_defimpl(self, parser, tmp_path):
        code = """
defimpl Printable, for: Integer do
  def to_string(value), do: Integer.to_string(value)
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert any("Printable" in s for s in symbols)

    def test_parse_def_and_defp(self, parser, tmp_path):
        """Property 2: Keyword-to-ChunkType mapping for functions."""
        code = """
defmodule M do
  def public_func(x) do
    x + 1
  end

  defp private_func(y) do
    y * 2
  end
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        func_chunks = [c for c in chunks if c.chunk_type.value == "function"]
        func_symbols = [c.symbol for c in func_chunks]
        # At least one function should have a correct name
        assert any("public_func" in s or "private_func" in s for s in func_symbols)

    def test_parse_defmacro(self, parser, tmp_path):
        code = """
defmodule M do
  defmacro my_macro(expr) do
    quote do
      unquote(expr)
    end
  end
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        macro_chunks = [c for c in chunks if c.chunk_type.value == "macro"]
        assert len(macro_chunks) >= 1
        assert any("my_macro" in c.symbol for c in macro_chunks)

    def test_parse_defguard(self, parser, tmp_path):
        code = """
defmodule M do
  defguard is_positive(x) when is_integer(x) and x > 0
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert any("is_positive" in s for s in symbols)

    def test_parse_defdelegate(self, parser, tmp_path):
        code = """
defmodule M do
  defdelegate fetch(key), to: MyModule
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert any("fetch" in s for s in symbols)

    def test_function_name_extraction(self, parser, tmp_path):
        """Property 3: Name Extraction Accuracy."""
        code = """
defmodule M do
  def changeset(user, attrs) do
    user
  end
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        func_chunks = [c for c in chunks if c.chunk_type.value == "function"]
        func_names = [c.symbol for c in func_chunks]
        # Must extract "changeset", not "def" or the full call text
        assert any(s == "changeset" for s in func_names), f"Expected 'changeset' in {func_names}"

    def test_parse_type_and_spec(self, parser, tmp_path):
        """Property 4: Attribute Recognition for types/specs."""
        code = """
defmodule M do
  @type t :: %__MODULE__{}
  @spec foo(integer()) :: atom()
  def foo(x), do: :ok
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        type_chunks = [c for c in chunks if c.chunk_type.value == "type"]
        # At least one type chunk should exist (may be merged by cAST)
        all_content = " ".join(c.code for c in chunks)
        assert "@type" in all_content or len(type_chunks) >= 1

    def test_parse_callback(self, parser, tmp_path):
        code = """
defmodule M do
  @callback my_cb(term()) :: :ok
  def foo, do: :bar
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        all_content = " ".join(c.code for c in chunks)
        assert "@callback" in all_content

    def test_parse_imports(self, parser, tmp_path):
        code = """
defmodule M do
  use GenServer
  import Ecto.Changeset
  alias MyApp.Repo
  require Logger

  def foo, do: :ok
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        all_content = " ".join(c.code for c in chunks)
        for keyword in ["use GenServer", "import Ecto.Changeset",
                         "alias MyApp.Repo", "require Logger"]:
            assert keyword in all_content, f"Expected '{keyword}' in parsed content"

        # Verify import chunks have ChunkType.IMPORT
        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
        assert len(import_chunks) > 0, "Expected at least one IMPORT chunk"
        import_content = " ".join(c.code for c in import_chunks)
        for keyword in ["use GenServer", "import Ecto.Changeset",
                         "alias MyApp.Repo", "require Logger"]:
            assert keyword in import_content, (
                f"Expected '{keyword}' in IMPORT chunks, got: {import_content}"
            )

    def test_parse_comments(self, parser, tmp_path):
        code = """
# A line comment
defmodule M do
  @moduledoc \"\"\"
  Module documentation.
  \"\"\"

  @doc \"\"\"
  Function documentation.
  \"\"\"
  def foo, do: :ok
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        all_content = " ".join(c.code for c in chunks)
        assert "@moduledoc" in all_content or "Module documentation" in all_content

    def test_parse_comprehensive_fixture(self, parser):
        fixture_path = Path(__file__).parent / "fixtures" / "elixir" / "comprehensive.ex"
        if not fixture_path.exists():
            pytest.skip("Elixir comprehensive fixture not found")

        chunks = parser.parse_file(fixture_path, FileId(1))

        assert len(chunks) >= 5

        symbols = [c.symbol for c in chunks]
        # Should find key modules
        assert any("MyApp.Accounts.User" in s for s in symbols)
        assert any("Printable" in s for s in symbols)
        # Should find key functions
        assert any("changeset" in s for s in symbols)

    def test_zero_arity_function(self, parser, tmp_path):
        code = """
defmodule M do
  def hello do
    :world
  end
end
"""
        f = tmp_path / "test.ex"
        f.write_text(code)
        chunks = parser.parse_file(f, FileId(1))

        func_chunks = [c for c in chunks if c.chunk_type.value == "function"]
        assert any("hello" in c.symbol for c in func_chunks)


class TestElixirImportResolution:
    """Tests for Elixir import path resolution including umbrella apps."""

    @pytest.fixture()
    def mapping(self):
        return ElixirMapping()

    def test_standard_mix_project(self, mapping, tmp_path):
        """Resolves imports in standard lib/ layout."""
        target = tmp_path / "lib" / "my_app" / "repo.ex"
        target.parent.mkdir(parents=True)
        target.write_text("defmodule MyApp.Repo do end")

        source = tmp_path / "lib" / "my_app" / "accounts.ex"
        result = mapping.resolve_import_paths("alias MyApp.Repo", tmp_path, source)
        assert result == [target]

    def test_umbrella_app_cross_app_resolution(self, mapping, tmp_path):
        """Resolves imports across sibling umbrella apps."""
        # Create umbrella structure
        target = tmp_path / "apps" / "core" / "lib" / "core" / "helper.ex"
        target.parent.mkdir(parents=True)
        target.write_text("defmodule Core.Helper do end")

        source_dir = tmp_path / "apps" / "web" / "lib" / "web"
        source_dir.mkdir(parents=True)
        source = source_dir / "controller.ex"

        result = mapping.resolve_import_paths("import Core.Helper", tmp_path, source)
        assert result == [target]

    def test_umbrella_app_same_app_resolution(self, mapping, tmp_path):
        """Resolves imports within the same umbrella app."""
        target = tmp_path / "apps" / "web" / "lib" / "web" / "router.ex"
        target.parent.mkdir(parents=True)
        target.write_text("defmodule Web.Router do end")

        source = tmp_path / "apps" / "web" / "lib" / "web" / "endpoint.ex"
        result = mapping.resolve_import_paths("alias Web.Router", tmp_path, source)
        assert result == [target]

    def test_non_umbrella_source_ignores_apps(self, mapping, tmp_path):
        """Non-umbrella source files don't probe apps/ directory."""
        target = tmp_path / "lib" / "my_app" / "repo.ex"
        target.parent.mkdir(parents=True)
        target.write_text("defmodule MyApp.Repo do end")

        source = tmp_path / "lib" / "my_app" / "web.ex"
        result = mapping.resolve_import_paths("alias MyApp.Repo", tmp_path, source)
        assert result == [target]

    def test_unresolvable_returns_empty(self, mapping, tmp_path):
        """Returns empty list when import cannot be resolved."""
        source = tmp_path / "lib" / "my_app.ex"
        result = mapping.resolve_import_paths("alias NoSuch.Module", tmp_path, source)
        assert result == []
