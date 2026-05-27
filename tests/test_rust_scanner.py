import pytest
from pathlib import Path

try:
    from chunkhound_native import scan_files
except (ImportError, AttributeError):
    pytest.skip("Rust extension not built", allow_module_level=True)


def test_scan_files_matches_python_walker(tmp_path):
    from chunkhound.utils.file_patterns import walk_directory_tree
    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.txt").write_text("skip")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("y = 2")

    rust_files = set(scan_files(str(tmp_path), ["py", "md"]))
    python_files, _ = walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*.md"], [], {}
    )
    assert rust_files == {str(p) for p in python_files}


def test_skip_dirs_prunes_excluded(tmp_path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"")
    (tmp_path / "real.py").write_text("x = 1")

    pruned = set(scan_files(str(tmp_path), ["py", "pyc"], skip_dirs=["__pycache__"]))
    assert all("__pycache__" not in p for p in pruned)


def test_returns_empty_for_unknown_extension(tmp_path):
    (tmp_path / "file.py").write_text("x = 1")
    result = scan_files(str(tmp_path), ["xyz_never_exists"])
    assert result == []


def test_walk_directory_tree_uses_rust_path(tmp_path):
    """Integration: env-var gate + _fnmatch_to_gitignore + scan_files all wired together."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.txt").write_text("skip")

    original = fp._USE_RUST
    fp._USE_RUST = True
    try:
        files, _ = fp.walk_directory_tree(
            tmp_path, tmp_path, ["**/*.py", "**/*.md"], [], {}
        )
    finally:
        fp._USE_RUST = original

    assert {p.name for p in files} == {"a.py", "b.md"}
