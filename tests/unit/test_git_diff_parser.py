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
