import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.JAVASCRIPT),
    reason="JavaScript parser not available",
)
def test_js_scalar_declaration_maps_to_variable():
    factory = ParserFactory()
    parser = factory.create_parser(Language.JAVASCRIPT)

    content = "const x = 1;\n"
    chunks = parser.parse_content(content, "test.js", FileId(1))
    assert chunks

    x = next((c for c in chunks if c.symbol == "x"), None)
    assert x is not None
    assert x.metadata.get("kind") == "variable"
    assert x.chunk_type == ChunkType.VARIABLE


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.JAVASCRIPT),
    reason="JavaScript parser not available",
)
def test_js_object_initializer_still_maps_to_object():
    factory = ParserFactory()
    parser = factory.create_parser(Language.JAVASCRIPT)

    content = "const cfg = { a: 1 };\n"
    chunks = parser.parse_content(content, "test.js", FileId(1))
    assert chunks

    cfg = next((c for c in chunks if c.symbol == "cfg"), None)
    assert cfg is not None
    assert cfg.metadata.get("chunk_type_hint") == "object"
    assert cfg.chunk_type == ChunkType.OBJECT


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.JAVASCRIPT),
    reason="JavaScript parser not available",
)
def test_js_class_declaration_maps_to_class():
    factory = ParserFactory()
    parser = factory.create_parser(Language.JAVASCRIPT)

    content = "class Foo {}\n"
    chunks = parser.parse_content(content, "test.js", FileId(1))
    assert chunks

    foo = next((c for c in chunks if c.symbol == "Foo"), None)
    assert foo is not None
    assert foo.metadata.get("node_type") == "class_declaration"
    assert foo.chunk_type == ChunkType.CLASS

