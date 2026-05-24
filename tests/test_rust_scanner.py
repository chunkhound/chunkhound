import pytest
from pathlib import Path

pytest.importorskip("chunkhound_native", reason="Rust extension not built")
from chunkhound_native import scan_files

FIXTURES = Path(__file__).parent / "fixtures"


def test_scan_files_matches_python_walker():
    from chunkhound.utils.file_patterns import walk_directory_tree
    rust_files = set(scan_files(str(FIXTURES), ["py", "md"]))
    python_files, _ = walk_directory_tree(
        FIXTURES, FIXTURES, ["**/*.py", "**/*.md"], [], {}
    )
    assert rust_files == {str(p) for p in python_files}


def test_skip_dirs_prunes_excluded():
    pruned = set(scan_files(str(FIXTURES), ["py"], skip_dirs=["__pycache__"]))
    assert all("__pycache__" not in p for p in pruned)


def test_returns_empty_for_unknown_extension():
    result = scan_files(str(FIXTURES), ["xyz_never_exists"])
    assert result == []
