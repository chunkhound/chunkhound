"""Tests for Ruby language parser."""

import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.mappings.ruby import RubyMapping
from chunkhound.parsers.parser_factory import get_parser_factory


class TestRubyFileDetection:
    """Property 1: File Extension Recognition."""

    def test_rb_extension(self):
        assert Language.from_file_extension("app/models/user.rb") == Language.RUBY

    def test_rake_extension(self):
        assert Language.from_file_extension("lib/tasks/db.rake") == Language.RUBY

    def test_gemspec_extension(self):
        assert Language.from_file_extension("foo.gemspec") == Language.RUBY

    def test_gemfile_by_name(self):
        assert Language.from_file_extension("Gemfile") == Language.RUBY

    def test_rakefile_by_name(self):
        assert Language.from_file_extension("Rakefile") == Language.RUBY

    def test_is_programming_language(self):
        assert Language.RUBY.is_programming_language

    def test_supports_classes(self):
        assert Language.RUBY.supports_classes


class TestRubyMapping:
    """Test RubyMapping extraction logic."""

    def test_language_is_ruby(self):
        assert RubyMapping().language == Language.RUBY

    def test_clean_comment_text(self):
        mapping = RubyMapping()
        assert mapping.clean_comment_text("# hello world") == "hello world"
        assert mapping.clean_comment_text("#comment") == "comment"


class TestRubyParser:
    """Test Ruby parser functionality against user-visible chunk contracts."""

    @pytest.fixture
    def parser(self):
        return get_parser_factory().create_parser(Language.RUBY)

    def test_parser_loads(self, parser):
        assert parser is not None

    def test_parse_module(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("module AssessmentFunctions\n  def register; end\nend\n")
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert any(s == "AssessmentFunctions" for s in symbols)
        # Ruby modules are namespaces, not functions
        module_chunks = [c for c in chunks if c.symbol == "AssessmentFunctions"]
        assert any(c.chunk_type == ChunkType.NAMESPACE for c in module_chunks)

    def test_parse_class_with_inheritance(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("class BookingCandidate < ApplicationRecord\nend\n")
        chunks = parser.parse_file(f, FileId(1))

        class_chunks = [c for c in chunks if c.symbol == "BookingCandidate"]
        assert class_chunks, [c.symbol for c in chunks]
        assert any(c.chunk_type == ChunkType.CLASS for c in class_chunks)

    def test_parse_instance_method(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("def register_assessments(x)\n  x + 1\nend\n")
        chunks = parser.parse_file(f, FileId(1))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert any(c.symbol == "register_assessments" for c in method_chunks)

    def test_parse_singleton_method(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("class C\n  def self.build(args)\n    new(args)\n  end\nend\n")
        chunks = parser.parse_file(f, FileId(1))

        symbols = [c.symbol for c in chunks]
        assert "build" in symbols

    def test_parse_constant(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text("MAX_RETRIES = 3\n")
        chunks = parser.parse_file(f, FileId(1))

        const_chunks = [c for c in chunks if c.symbol == "MAX_RETRIES"]
        assert const_chunks
        assert any(c.chunk_type == ChunkType.VARIABLE for c in const_chunks)

    def test_parse_requires(self, parser, tmp_path):
        f = tmp_path / "test.rb"
        f.write_text('require "json"\nrequire_relative "../lib/foo"\n')
        chunks = parser.parse_file(f, FileId(1))

        import_chunks = [c for c in chunks if c.chunk_type == ChunkType.IMPORT]
        import_content = " ".join(c.code for c in import_chunks)
        assert "json" in import_content
        assert "foo" in import_content

    def test_method_name_extraction_is_clean(self, parser, tmp_path):
        """Property 3: extract the name, not 'def' or the full signature."""
        f = tmp_path / "test.rb"
        f.write_text('def full_name\n  "#{first} #{last}"\nend\n')
        chunks = parser.parse_file(f, FileId(1))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert any(c.symbol == "full_name" for c in method_chunks), (
            f"Expected 'full_name' in {[c.symbol for c in method_chunks]}"
        )

    @pytest.mark.parametrize(
        "source,expected_symbol",
        [
            ("def name=(value)\n  @name = value\nend\n", "name="),
            ("def [](key)\n  @data[key]\nend\n", "[]"),
            ("def <=>(other)\n  0\nend\n", "<=>"),
            ("def +(other)\n  dup\nend\n", "+"),
        ],
    )
    def test_non_identifier_method_names_extracted(
        self, parser, tmp_path, source, expected_symbol
    ):
        """Setter and operator methods must become searchable METHOD chunks.

        Ruby method names are not always ``identifier`` nodes: setters
        (``name=``) are ``setter`` nodes and operators (``[]``, ``<=>``,
        ``+``) are ``operator`` nodes. The DEFINITION query must match every
        shape (upstream tree-sitter-ruby tags.scm uses ``name: (_)``); the
        narrower ``name: (identifier)`` drops these methods from the index
        entirely. One method per file keeps the cAST greedy-merge pass from
        absorbing the chunk under a neighbour's name.
        """
        f = tmp_path / "test.rb"
        f.write_text(source)
        chunks = parser.parse_file(f, FileId(1))

        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert any(c.symbol == expected_symbol for c in method_chunks), (
            f"Expected {expected_symbol!r} in {[c.symbol for c in method_chunks]}"
        )


class TestRubyImportResolution:
    """Tests for Ruby require/require_relative path resolution."""

    @pytest.fixture
    def mapping(self):
        return RubyMapping()

    def test_require_relative_resolves(self, mapping, tmp_path):
        target = tmp_path / "lib" / "foo.rb"
        target.parent.mkdir(parents=True)
        target.write_text("module Foo; end")

        source = tmp_path / "lib" / "bar.rb"
        result = mapping.resolve_import_paths(
            'require_relative "foo"', tmp_path, source
        )
        assert result == [target]

    def test_require_resolves_against_lib(self, mapping, tmp_path):
        target = tmp_path / "lib" / "helpers.rb"
        target.parent.mkdir(parents=True)
        target.write_text("module Helpers; end")

        source = tmp_path / "app" / "thing.rb"
        result = mapping.resolve_import_paths('require "helpers"', tmp_path, source)
        assert result == [target]

    def test_unresolvable_returns_empty(self, mapping, tmp_path):
        source = tmp_path / "app" / "thing.rb"
        result = mapping.resolve_import_paths('require "nope"', tmp_path, source)
        assert result == []
