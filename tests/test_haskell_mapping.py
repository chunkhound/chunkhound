import pytest

from chunkhound.core.types.common import FileId, Language, ChunkType
from chunkhound.parsers.parser_factory import get_parser_factory


def parse_haskell(content: str):
    factory = get_parser_factory()
    parser = factory.create_parser(Language.HASKELL)
    return parser.parse_content(content, "test.hs", FileId(1))


def test_haskell_captures_pattern_synonyms_signature_and_equation():
    # Based on tree-sitter-haskell corpus patterns
    content = (
        "pattern A :: A -> A -> (A, A)\n"
        "pattern A, A :: A\n"
        "pattern A {a, a} = (a, a)\n"
        "pattern (:->) :: A\n"
        "pattern a :-> b <- a\n"
    )

    chunks = parse_haskell(content)

    # Ensure we got some function-definition-like chunks for patterns
    names = {c.symbol for c in chunks if c.chunk_type == ChunkType.FUNCTION}

    assert any("A" == n or n.startswith("A") for n in names), (
        f"Expected pattern synonym name 'A' in FUNCTION chunks, got: {sorted(names)}"
    )


def test_haskell_captures_top_level_bind_and_function():
    content = (
        "const = 42\n"
        "add x y = x + y\n"
    )

    chunks = parse_haskell(content)
    function_names = {c.symbol for c in chunks if c.chunk_type == ChunkType.FUNCTION}
    variable_names = {c.symbol for c in chunks if c.chunk_type == ChunkType.VARIABLE}

    assert "const" in variable_names, (
        f"Expected 'const' bind to be VARIABLE, got: {sorted(variable_names)}"
    )
    assert "add" in function_names, (
        f"Expected 'add' function to be FUNCTION, got: {sorted(function_names)}"
    )


def test_haskell_captures_class_and_instance_methods():
    content = (
        "class C a where\n"
        "  m :: a -> a\n"
        "  m x = x\n"
        "\n"
        "instance C Int where\n"
        "  m x = x\n"
    )

    chunks = parse_haskell(content)

    # Methods are emitted as BLOCKs by the universal pipeline
    method_names = {c.symbol for c in chunks if c.chunk_type == ChunkType.BLOCK}
    assert "m" in method_names, (
        f"Expected method 'm' in BLOCK chunks, got: {sorted(method_names)}"
    )
