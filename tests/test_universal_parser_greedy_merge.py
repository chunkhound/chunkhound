from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.parsers.universal_engine import UniversalChunk, UniversalConcept


def test_greedy_merge_does_not_duplicate_partially_overlapping_chunks():
    parser = create_parser_for_language(Language.TSX)
    outer = UniversalChunk(
        concept=UniversalConcept.DEFINITION,
        name="submitFeedback_part1",
        content=(
            "async function submitFeedback() {\n"
            "  const payload = {\n"
            "    route_query_keys: ["
            "...new URLSearchParams(window.location.search).keys()"
            "].sort(),\n"
        ),
        start_line=150,
        end_line=172,
        metadata={},
        language_node_type="function_declaration",
    )
    nested = UniversalChunk(
        concept=UniversalConcept.DEFINITION,
        name="payload",
        content=(
            "const payload = {\n"
            "    route_query_keys: ["
            "...new URLSearchParams(window.location.search).keys()"
            "].sort(),\n"
            "    viewport: { width: window.innerWidth },\n"
            "  };"
        ),
        start_line=166,
        end_line=185,
        metadata={},
        language_node_type="lexical_declaration",
    )

    chunks = parser._greedy_merge_pass([outer, nested])

    assert len(chunks) == 2
    assert all(chunk.content.count("const payload = {") == 1 for chunk in chunks)
