import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.PYTHON),
    reason="Python tree-sitter parser not available",
)
def test_python_assignment_maps_to_variable_chunk_type():
    factory = ParserFactory()
    parser = factory.create_parser(Language.PYTHON)

    content = "FOO = 1\n"
    chunks = parser.parse_content(content, "test.py", FileId(1))
    assert chunks

    foo = next((c for c in chunks if c.symbol == "FOO"), None)
    assert foo is not None
    assert foo.metadata.get("node_type") == "assignment"
    assert foo.metadata.get("kind") == "variable"
    assert foo.chunk_type == ChunkType.VARIABLE

