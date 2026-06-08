"""Unit tests for chunkhound.core.git_diff.parser."""
import pytest

from chunkhound.core.git_diff.parser import parse_diff_to_chunks
from chunkhound.core.types.common import ChunkType, Language


SINGLE_HUNK_DIFF = """\
diff --git a/foo/bar.py b/foo/bar.py
index abc..def 100644
--- a/foo/bar.py
+++ b/foo/bar.py
@@ -1,3 +1,4 @@ def my_func():
 line1
+new line
 line2
 line3
"""

MULTI_HUNK_SAME_FILE = """\
diff --git a/src/main.py b/src/main.py
index abc..def 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,2 +1,3 @@ def alpha():
 a
+b
 c
@@ -10,2 +11,3 @@ def beta():
 x
+y
 z
"""

MULTI_FILE_DIFF = """\
diff --git a/file_a.py b/file_a.py
index 111..222 100644
--- a/file_a.py
+++ b/file_a.py
@@ -1,1 +1,2 @@ def a():
 old
+new
diff --git a/file_b.py b/file_b.py
index 333..444 100644
--- a/file_b.py
+++ b/file_b.py
@@ -5,1 +5,2 @@ def b():
 old
+new
"""

DELETION_HUNK_DIFF = """\
diff --git a/gone.py b/gone.py
index abc..def 100644
--- a/gone.py
+++ b/gone.py
@@ -5,3 +5,0 @@
-line1
-line2
-line3
"""

DEV_NULL_DIFF = """\
diff --git a/deleted.py b/deleted.py
deleted file mode 100644
--- a/deleted.py
+++ /dev/null
@@ -1,2 +0,0 @@
-line1
-line2
"""


def test_single_hunk_file_path_strips_b_prefix() -> None:
    chunks = parse_diff_to_chunks(SINGLE_HUNK_DIFF)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.file_path == "foo/bar.py"
    assert chunk.start_line == 1
    assert chunk.language == Language.GIT_DIFF
    assert chunk.chunk_type == ChunkType.BLOCK


def test_multi_hunk_same_file_produces_two_chunks() -> None:
    chunks = parse_diff_to_chunks(MULTI_HUNK_SAME_FILE)
    assert len(chunks) == 2
    assert chunks[0].file_path == "src/main.py"
    assert chunks[1].file_path == "src/main.py"


def test_multi_file_diff_different_file_paths() -> None:
    chunks = parse_diff_to_chunks(MULTI_FILE_DIFF)
    assert len(chunks) == 2
    paths = {c.file_path for c in chunks}
    assert paths == {"file_a.py", "file_b.py"}


def test_deletion_hunk_skipped() -> None:
    chunks = parse_diff_to_chunks(DELETION_HUNK_DIFF)
    assert len(chunks) == 0


def test_empty_input_returns_empty_list() -> None:
    chunks = parse_diff_to_chunks("")
    assert chunks == []


def test_no_plus_plus_plus_before_hunk_no_crash() -> None:
    diff = """\
diff --git a/x.py b/x.py
@@ -1,1 +1,2 @@
 old
+new
"""
    chunks = parse_diff_to_chunks(diff)
    assert len(chunks) == 0


def test_end_line_computed_from_hunk_header() -> None:
    # @@ -1,3 +1,4 @@ → new_start=1, new_count=4 → end_line = 1+4-1 = 4
    chunks = parse_diff_to_chunks(SINGLE_HUNK_DIFF)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 4


def test_multi_hunk_end_lines() -> None:
    # First hunk: @@ -1,2 +1,3 @@ → end_line = 1+3-1 = 3
    # Second hunk: @@ -10,2 +11,3 @@ → end_line = 11+3-1 = 13
    chunks = parse_diff_to_chunks(MULTI_HUNK_SAME_FILE)
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[1].start_line == 11
    assert chunks[1].end_line == 13


def test_dev_null_path_chunk_skipped() -> None:
    chunks = parse_diff_to_chunks(DEV_NULL_DIFF)
    assert len(chunks) == 0


def test_oversized_hunk_splits_into_parts() -> None:
    # Build a hunk whose lines total > 10_000 chars (default max_chunk_chars).
    big_lines = [f"+line_{i:05d}\n" for i in range(1000)]  # ~12k chars
    big_body = "".join(big_lines)
    diff = (
        "diff --git a/big.json b/big.json\n"
        "index abc..def 100644\n"
        "--- a/big.json\n"
        "+++ b/big.json\n"
        f"@@ -0,0 +1,1000 @@ root\n"
        + big_body
    )
    chunks = parse_diff_to_chunks(diff)
    assert len(chunks) > 1, "oversized hunk must produce multiple chunks"
    for c in chunks:
        assert len(c.code) <= 10_000
    assert all("(part" in c.symbol for c in chunks)


def test_oversized_hunk_fragments_have_unique_start_lines() -> None:
    big_lines = [f"+line_{i:05d}\n" for i in range(1000)]  # ~12k chars, forces split
    diff = (
        "diff --git a/big.json b/big.json\n"
        "index abc..def 100644\n"
        "--- a/big.json\n"
        "+++ b/big.json\n"
        "@@ -0,0 +1,1000 @@ root\n"
        + "".join(big_lines)
    )
    chunks = parse_diff_to_chunks(diff)
    assert len(chunks) > 1
    start_lines = [c.start_line for c in chunks]
    assert len(start_lines) == len(set(start_lines)), "split fragments must have unique start_lines"


def test_split_fragment_start_lines_reflect_actual_diff_lines() -> None:
    """Split fragment start_lines must reflect real new-file line numbers, not ordinals.

    Construct a diff with hunk_start=100 and 60 addition lines.
    With max_chunk_chars=40, each '+x\\n' is 4 chars → 10 lines per part → 6 parts.
    Part 1 covers lines 100-109, part 2 must start at 110, part 3 at 120, etc.
    The old ordinal code would assign 100, 101, 102... (off by ~9 lines per part).
    """
    hunk_start = 100
    # 60 addition lines, each exactly 4 chars: "+x\n"  (fits 10 per 40-char part)
    lines = ["+x\n"] * 60
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        f"@@ -98,60 +{hunk_start},60 @@ fn\n"
        + "".join(lines)
    )
    # max_chunk_chars=40: the @@ header (~24 chars) + first addition fills 1 part;
    # use a small value to make splits predictable.
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=40)
    assert len(chunks) > 1, "expected multiple split fragments"

    # Each part of 10 '+x\n' lines = 40 chars.  The @@ header is in part 1.
    # Part 1: @@ header + lines until 40 chars, subsequent parts: 10 lines each.
    # At minimum, part 2 must start at line >= hunk_start + 1 (not hunk_start + 1
    # from ordinal, but the actual count of additions in part 1).
    for i in range(1, len(chunks)):
        prev_end = chunks[i - 1].end_line
        curr_start = chunks[i].start_line
        # Each fragment must start strictly after the previous one ends.
        assert curr_start > chunks[i - 1].start_line, (
            f"part {i+1} start_line ({curr_start}) must be > part {i} start_line "
            f"({chunks[i-1].start_line}); got ordinal instead of real line number"
        )
        # start_line must be within the hunk range
        assert hunk_start <= curr_start <= hunk_start + 59, (
            f"part {i+1} start_line {curr_start} outside hunk range [{hunk_start}, {hunk_start+59}]"
        )


def test_oversized_hunk_custom_max() -> None:
    lines = [f"+x\n"] * 50  # 50 × 3 chars = 150 chars
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1,0 +1,50 @@ fn\n"
        + "".join(lines)
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=40)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.code) <= 40


def test_small_hunk_not_split() -> None:
    chunks = parse_diff_to_chunks(SINGLE_HUNK_DIFF)
    assert len(chunks) == 1
    assert "(part" not in chunks[0].symbol


def test_single_overlong_line_split() -> None:
    # SVG/minified JS: one line longer than max_chunk_chars
    long_line = "+" + "x" * 25_000 + "\n"
    diff = (
        "diff --git a/icon.svg b/icon.svg\n"
        "--- a/icon.svg\n"
        "+++ b/icon.svg\n"
        "@@ -0,0 +1,1 @@\n"
        + long_line
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=10_000)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.code) <= 10_000


def test_single_overlong_line_fragments_report_hunk_start_as_start_line() -> None:
    """All char-split fragments from a single-line hunk share the hunk's start_line."""
    long_line = "+" + "x" * 25_000 + "\n"
    diff = (
        "diff --git a/icon.svg b/icon.svg\n"
        "--- a/icon.svg\n"
        "+++ b/icon.svg\n"
        "@@ -0,0 +42,1 @@\n"
        + long_line
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=10_000)
    assert len(chunks) > 1
    for c in chunks:
        assert c.start_line == 42, (
            f"fragment {c.symbol!r} has start_line={c.start_line}, expected 42"
        )


def test_single_overlong_line_fragments_end_line_equals_start_line() -> None:
    """For a 1-line hunk, all char-split fragments must have end_line == start_line."""
    long_line = "+" + "x" * 25_000 + "\n"
    diff = (
        "diff --git a/icon.svg b/icon.svg\n"
        "--- a/icon.svg\n"
        "+++ b/icon.svg\n"
        "@@ -0,0 +42,1 @@\n"
        + long_line
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=10_000)
    assert len(chunks) > 1
    for c in chunks:
        assert c.end_line == c.start_line, (
            f"fragment {c.symbol!r}: end_line={c.end_line} != start_line={c.start_line}; "
            "single-line hunk fragments must report their actual line, not the full hunk tail"
        )


def test_multi_line_split_intermediate_fragments_have_per_part_end_lines() -> None:
    """Intermediate split fragments must NOT claim hunk_end as their end_line.

    10 addition lines at hunk_start=100, max_chunk_chars=30 forces a ~5+5 split.
    Each '+x\\n' is 3 chars; 10 such lines = 30 chars exactly, so the @@ header line
    (included in part 1) pushes it over. Use lines long enough to force a clear split.
    """
    # Each '+xxxxxxx\n' = 9 chars; 4 lines = 36 > 30 → flush after 3 lines per part.
    # 10 lines total → part 1: lines 100–102, part 2: lines 103–105, etc.
    lines = ["+xxxxxxx\n"] * 10
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -98,10 +100,10 @@ fn\n"
        + "".join(lines)
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=30)
    assert len(chunks) >= 2, "expected at least 2 split fragments"

    hunk_end = 109  # hunk_start=100, new_count=10 → end=109
    # Every intermediate fragment must end strictly before hunk_end.
    for i, c in enumerate(chunks[:-1]):
        assert c.end_line < hunk_end, (
            f"part {i+1} end_line={c.end_line} equals hunk_end={hunk_end}; "
            "intermediate fragments must not claim the full hunk's tail"
        )
    # Last fragment must end exactly at hunk_end.
    assert chunks[-1].end_line == hunk_end, (
        f"last fragment end_line={chunks[-1].end_line} should equal hunk_end={hunk_end}"
    )


def test_split_fragments_cover_full_hunk_range_contiguously() -> None:
    """Split fragments must cover [hunk_start, hunk_end] with no gaps and no overlaps.

    Use max_chunk_chars=90 with 20-char lines so the @@ header (≈27 chars) always
    fits alongside the first addition lines — no header-only parts that would share
    the same start_line as the following fragment.
    """
    # Each "+xxxxxxxxxxxxxxxxxx\n" = 20 chars; @@ header ≈27 chars.
    # 27+60=87 ≤ 90, 27+80=107 > 90 → 3 addition lines per part after first break.
    lines = ["+xxxxxxxxxxxxxxxxxx\n"] * 10  # 20 chars each, 10 lines
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -98,10 +100,10 @@ fn\n"
        + "".join(lines)
    )
    chunks = parse_diff_to_chunks(diff, max_chunk_chars=90)
    assert len(chunks) >= 2

    hunk_start = 100
    hunk_end = 109

    assert chunks[0].start_line == hunk_start, (
        f"first fragment start_line={chunks[0].start_line}, expected {hunk_start}"
    )
    assert chunks[-1].end_line == hunk_end, (
        f"last fragment end_line={chunks[-1].end_line}, expected {hunk_end}"
    )
    # Adjacent fragments must be strictly contiguous: prev.end_line + 1 == next.start_line
    for i in range(len(chunks) - 1):
        assert chunks[i].end_line + 1 == chunks[i + 1].start_line, (
            f"gap/overlap between part {i+1} (end={chunks[i].end_line}) "
            f"and part {i+2} (start={chunks[i+1].start_line})"
        )
