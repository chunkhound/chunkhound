import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.TOML),
    reason="TOML parser not available",
)
def test_toml_emits_key_value_and_table_chunk_types():
    factory = ParserFactory()
    parser = factory.create_parser(Language.TOML)

    content = "a = 1\n\n[tool]\nname = \"demo\"\n"
    chunks = parser.parse_content(content, "test.toml", FileId(1))
    assert chunks

    assert any(
        c.chunk_type == ChunkType.KEY_VALUE and c.metadata.get("chunk_type_hint") == "key_value"
        for c in chunks
    )
    assert any(
        c.chunk_type == ChunkType.TABLE and c.metadata.get("chunk_type_hint") == "table"
        for c in chunks
    )

