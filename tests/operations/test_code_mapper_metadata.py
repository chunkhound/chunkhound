from chunkhound.code_mapper.pipeline import (
    _is_empty_research_result,
    _merge_sources_metadata,
)


def test_merge_sources_metadata_deduplicates_files_and_chunks() -> None:
    """_merge_sources_metadata should deduplicate files and chunk ranges."""
    results = [
        {
            "metadata": {
                "sources": {
                    "files": ["scope/a.py", "scope/b.py"],
                    "chunks": [
                        {"file_path": "scope/a.py", "start_line": 1, "end_line": 10},
                        # Duplicate of the same chunk; should be collapsed.
                        {"file_path": "scope/a.py", "start_line": 1, "end_line": 10},
                    ],
                },
                "aggregation_stats": {"files_total": 10, "chunks_total": 20},
            },
        },
        {
            "metadata": {
                "sources": {
                    "files": ["scope/b.py", "scope/c.py"],
                    "chunks": [
                        {"file_path": "scope/b.py", "start_line": 5, "end_line": 15},
                        {"file_path": "scope/c.py", "start_line": 7, "end_line": 9},
                    ],
                },
                # Later aggregation stats should be able to override earlier zeros.
                "aggregation_stats": {"files_total": 10, "chunks_total": 20},
            },
        },
    ]

    unified_files, unified_chunks, total_files, total_chunks = _merge_sources_metadata(
        results
    )

    # Files are deduplicated and normalized as keys in the mapping.
    assert set(unified_files.keys()) == {"scope/a.py", "scope/b.py", "scope/c.py"}

    # Chunks are deduplicated by (file_path, start_line, end_line).
    assert len(unified_chunks) == 3
    assert {
        "file_path": "scope/a.py",
        "start_line": 1,
        "end_line": 10,
    } in unified_chunks
    assert {
        "file_path": "scope/b.py",
        "start_line": 5,
        "end_line": 15,
    } in unified_chunks
    assert {
        "file_path": "scope/c.py",
        "start_line": 7,
        "end_line": 9,
    } in unified_chunks

    # Aggregation stats are propagated when present.
    assert total_files == 10
    assert total_chunks == 20


def test_is_empty_research_result_detects_skipped_synthesis() -> None:
    """_is_empty_research_result should detect DeepResearch 'no context' responses."""
    result = {
        "answer": (
            "No relevant code context found for: 'foo'.\n\n"
            "Try a more code-specific question."
        ),
        "metadata": {
            "skipped_synthesis": True,
            "depth_reached": 0,
            "nodes_explored": 1,
            "chunks_analyzed": 0,
            "files_analyzed": 0,
        },
    }

    assert _is_empty_research_result(result)

    non_empty = {
        "answer": "# Heading\n\nSome real content.",
        "metadata": {},
    }
    assert not _is_empty_research_result(non_empty)
