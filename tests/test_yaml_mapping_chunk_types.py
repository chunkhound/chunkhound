import pytest

from chunkhound.core.types.common import ChunkType, FileId, Language
from chunkhound.parsers.parser_factory import ParserFactory


@pytest.mark.skipif(
    not ParserFactory().is_language_available(Language.YAML),
    reason="YAML parser not available",
)
def test_yaml_tree_sitter_emits_key_value_and_array(monkeypatch):
    # Force tree-sitter YAML (RapidYAML wrapper will delegate to fallback when disabled).
    monkeypatch.setenv("CHUNKHOUND_YAML_ENGINE", "tree")

    factory = ParserFactory()
    parser = factory.create_parser(Language.YAML)

    content = (
        "root:\n"
        "  name: example\n"
        "  items:\n"
        "    - a\n"
        "    - b\n"
    )
    chunks = parser.parse_content(content, "test.yaml", FileId(1))
    assert chunks

    key_values = [c for c in chunks if c.chunk_type == ChunkType.KEY_VALUE]
    arrays = [c for c in chunks if c.chunk_type == ChunkType.ARRAY]

    assert any(c.metadata.get("chunk_type_hint") == "key_value" for c in key_values)
    assert any(c.metadata.get("chunk_type_hint") == "array" for c in arrays)
