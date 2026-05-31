import pytest
from pathlib import Path

try:
    from chunkhound_native import scan_files as _scan_files
    _RUST_AVAILABLE = True
except (ImportError, AttributeError):
    _scan_files = None
    _RUST_AVAILABLE = False

requires_rust = pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust extension not built")


@requires_rust
def test_scan_files_matches_python_walker(tmp_path):
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.txt").write_text("skip")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("y = 2")

    rust_files = set(_scan_files(str(tmp_path), ["py", "md"]))

    original = fp._USE_RUST
    fp._USE_RUST = False
    try:
        python_files, _ = fp.walk_directory_tree(
            tmp_path, tmp_path, ["**/*.py", "**/*.md"], [], {}
        )
    finally:
        fp._USE_RUST = original

    assert rust_files == {str(p) for p in python_files}


@requires_rust
def test_skip_dirs_prunes_excluded(tmp_path):
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.cpython-312.pyc").write_bytes(b"")
    (tmp_path / "real.py").write_text("x = 1")

    pruned = set(_scan_files(str(tmp_path), ["py", "pyc"], skip_dirs=["__pycache__"]))
    assert all("__pycache__" not in p for p in pruned)


@requires_rust
def test_returns_empty_for_unknown_extension(tmp_path):
    (tmp_path / "file.py").write_text("x = 1")
    result = _scan_files(str(tmp_path), ["xyz_never_exists"])
    assert result == []


def test_python_fallback_when_rust_disabled(tmp_path):
    """Python path is taken when _USE_RUST=False even if the extension is present."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.rs").write_text("fn main() {}")

    original = fp._USE_RUST
    fp._USE_RUST = False
    try:
        files, _ = fp.walk_directory_tree(
            tmp_path, tmp_path, ["**/*.py"], [], {}
        )
    finally:
        fp._USE_RUST = original

    assert {p.name for p in files} == {"a.py"}


@requires_rust
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


@requires_rust
def test_exclude_patterns_parity(tmp_path):
    """Both paths must produce identical results when exclude_patterns are supplied."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "keep.py").write_text("x = 1")
    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "dep.py").write_text("y = 2")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "mod.pyc").write_bytes(b"")

    exclude = ["**/node_modules/**", "**/__pycache__/**"]

    rust_files = set(
        _scan_files(
            str(tmp_path),
            ["py", "pyc"],
            skip_dirs=[],
            exclude_patterns=[fp._fnmatch_to_gitignore(p) for p in exclude],
        )
    )

    original = fp._USE_RUST
    fp._USE_RUST = False
    try:
        python_files, _ = fp.walk_directory_tree(
            tmp_path, tmp_path, ["**/*.py", "**/*.pyc"], exclude, {}
        )
    finally:
        fp._USE_RUST = original

    assert rust_files == {str(p) for p in python_files}
