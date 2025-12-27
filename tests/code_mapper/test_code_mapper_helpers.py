from chunkhound.code_mapper import pipeline as code_mapper_pipeline


def test_extract_points_of_interest_parses_numbered_and_bullets() -> None:
    text = "\n".join(
        [
            "1. First point",
            "2) Second point",
            "- Third point",
            "* Fourth point",
            "",
        ]
    )

    points = code_mapper_pipeline._extract_points_of_interest(text, max_points=10)

    assert points == ["First point", "Second point", "Third point", "Fourth point"]


def test_extract_points_of_interest_dedupes_and_respects_limit() -> None:
    text = "\n".join(
        [
            "1. Repeat",
            "2. Repeat",
            "- Repeat",
            "- Unique",
        ]
    )

    points = code_mapper_pipeline._extract_points_of_interest(text, max_points=2)

    assert points == ["Repeat", "Unique"]


def test_derive_heading_from_point_strips_formatting() -> None:
    heading = code_mapper_pipeline._derive_heading_from_point("**Heading**: details here")

    assert heading == "Heading"


def test_derive_heading_from_point_truncates_long_text() -> None:
    long_text = "A" * 120
    heading = code_mapper_pipeline._derive_heading_from_point(long_text)

    assert heading.endswith("...")
    assert len(heading) <= 80


def test_slugify_heading_normalizes_text() -> None:
    slug = code_mapper_pipeline._slugify_heading("Heading: v2.1 (beta)!")

    assert slug == "heading-v2-1-beta"


def test_slugify_heading_caps_length() -> None:
    slug = code_mapper_pipeline._slugify_heading("A" * 200)

    assert len(slug) <= 60


def test_coverage_summary_lines_with_denominators() -> None:
    lines = code_mapper_pipeline._coverage_summary_lines(
        referenced_files=5,
        referenced_chunks=20,
        files_denominator=10,
        chunks_denominator=40,
        scope_total_files=10,
        scope_total_chunks=40,
    )

    assert lines[0] == "## Coverage Summary"
    assert "Referenced files: 5 / 10" in lines[2]
    assert "50.00%" in lines[2]
    assert "Referenced chunks: 20 / 40" in lines[3]
    assert "50.00%" in lines[3]


def test_coverage_summary_lines_without_denominators() -> None:
    lines = code_mapper_pipeline._coverage_summary_lines(
        referenced_files=5,
        referenced_chunks=20,
        files_denominator=None,
        chunks_denominator=None,
        scope_total_files=0,
        scope_total_chunks=0,
    )

    assert "database totals unavailable" in lines[2]
    assert "database totals unavailable" in lines[3]
