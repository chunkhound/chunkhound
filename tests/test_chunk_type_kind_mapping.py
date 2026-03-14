import pytest

from chunkhound.core.types.common import ChunkType, Language
from chunkhound.parsers.parser_factory import ParserFactory
from chunkhound.parsers.universal_engine import UniversalConcept


@pytest.fixture()
def universal_parser():
    # TEXT parser does not require tree-sitter and still provides UniversalParser.
    return ParserFactory().create_parser(Language.TEXT)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("variable", ChunkType.VARIABLE),
        ("loop_variable", ChunkType.VARIABLE),
        ("constant", ChunkType.VARIABLE),
        ("const", ChunkType.VARIABLE),
        ("define", ChunkType.VARIABLE),
        ("property", ChunkType.PROPERTY),
        ("field", ChunkType.FIELD),
        ("constructor", ChunkType.CONSTRUCTOR),
        ("initializer", ChunkType.CONSTRUCTOR),
        ("type_alias", ChunkType.TYPE_ALIAS),
        ("typedef", ChunkType.TYPE_ALIAS),
        ("type", ChunkType.TYPE),
        ("macro", ChunkType.MACRO),
        ("mapping_pair", ChunkType.KEY_VALUE),
        ("sequence_item", ChunkType.ARRAY),
    ],
)
def test_kind_maps_definition_to_expected_chunk_type(universal_parser, kind, expected):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.DEFINITION,
        {"kind": kind, "node_type": ""},
    )
    assert chunk_type == expected


def test_hint_still_overrides_kind(universal_parser):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.DEFINITION,
        {"chunk_type_hint": "array", "kind": "variable", "node_type": ""},
    )
    assert chunk_type == ChunkType.ARRAY


def test_import_concept_maps_to_import_chunk_type(universal_parser):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.IMPORT,
        {},
    )
    assert chunk_type == ChunkType.IMPORT


def test_import_hint_overrides_definition_concept(universal_parser):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.DEFINITION,
        {"chunk_type_hint": "import", "kind": "variable", "node_type": ""},
    )
    assert chunk_type == ChunkType.IMPORT


def test_comment_concept_maps_to_comment_chunk_type(universal_parser):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.COMMENT,
        {},
    )
    assert chunk_type == ChunkType.COMMENT


def test_block_concept_maps_to_block_chunk_type(universal_parser):
    chunk_type = universal_parser._map_concept_to_chunk_type(  # type: ignore[attr-defined]
        UniversalConcept.BLOCK,
        {},
    )
    assert chunk_type == ChunkType.BLOCK

