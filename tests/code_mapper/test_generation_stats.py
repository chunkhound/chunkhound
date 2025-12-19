from chunkhound.code_mapper.metadata import build_generation_stats_with_coverage


def test_build_generation_stats_with_coverage_scoped_totals() -> None:
    stats, coverage = build_generation_stats_with_coverage(
        generator_mode="code_research",
        total_research_calls=2,
        unified_source_files={"scope/a.py": "", "scope/b.py": ""},
        unified_chunks_dedup=[
            {"file_path": "scope/a.py", "start_line": 1, "end_line": 2},
            {"file_path": "scope/b.py", "start_line": 3, "end_line": 4},
        ],
        scope_label="scope",
        scope_total_files=3,
        scope_total_chunks=4,
        total_files_global=None,
        total_chunks_global=None,
    )

    assert stats["files"]["total_indexed"] == 3
    assert stats["files"]["basis"] == "scope"
    assert stats["files"]["coverage"] == "66.67%"
    assert stats["files"]["unreferenced_in_scope"] == 1
    assert stats["chunks"]["total_indexed"] == 4
    assert stats["chunks"]["coverage"] == "50.00%"

    assert coverage.referenced_files == 2
    assert coverage.files_denominator == 3
    assert coverage.unreferenced_files_in_scope == 1


def test_build_generation_stats_with_coverage_uses_global_fallback() -> None:
    stats, coverage = build_generation_stats_with_coverage(
        generator_mode="code_research",
        total_research_calls=1,
        unified_source_files={"scope/a.py": ""},
        unified_chunks_dedup=[
            {"file_path": "scope/a.py", "start_line": 1, "end_line": 2}
        ],
        scope_label="scope",
        scope_total_files=0,
        scope_total_chunks=0,
        total_files_global=10,
        total_chunks_global=20,
    )

    assert stats["files"]["basis"] == "database"
    assert stats["files"]["total_indexed"] == 10
    assert stats["chunks"]["total_indexed"] == 20
    assert coverage.files_denominator == 10
    assert coverage.chunks_denominator == 20
