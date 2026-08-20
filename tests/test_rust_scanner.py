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
def test_scan_files_matches_python_walker(tmp_path, monkeypatch):
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.txt").write_text("skip")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.py").write_text("y = 2")

    rust_files = set(_scan_files(str(tmp_path), ["py", "md"]))

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    python_files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*.md"], [], {}
    )

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


@requires_rust
def test_include_all_matches_unknown_extensions(tmp_path):
    (tmp_path / "known.py").write_text("x = 1")
    (tmp_path / "unknown.xyz").write_text("?")

    result = set(_scan_files(str(tmp_path), ["py"], include_all=True))

    assert result == {str(tmp_path / "known.py"), str(tmp_path / "unknown.xyz")}


def test_python_fallback_when_rust_disabled(tmp_path, monkeypatch):
    """Python path is taken when CHUNKHOUND_USE_RUST=0 even if the extension is present."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.rs").write_text("fn main() {}")

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py"], [], {}
    )

    assert {p.name for p in files} == {"a.py"}


@requires_rust
def test_walk_directory_tree_uses_rust_path(tmp_path, monkeypatch):
    """Integration: env-var gate + _fnmatch_to_gitignore + scan_files all wired together."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "a.py").write_text("x = 1")
    (tmp_path / "b.md").write_text("# hi")
    (tmp_path / "c.txt").write_text("skip")

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")
    files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*.md"], [], {}
    )

    assert {p.name for p in files} == {"a.py", "b.md"}


@requires_rust
def test_index_unknown_files_sentinel_uses_rust_fast_path(tmp_path, monkeypatch):
    """`"**/*"` (the index_unknown_files sentinel) must still take the Rust
    fast path instead of silently falling back to the slow Python os.walk.
    """
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "known.py").write_text("x = 1")
    (tmp_path / "unknown.xyz").write_text("?")

    calls = []
    real_scan_files = fp._rust_scan_files

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_scan_files(*args, **kwargs)

    monkeypatch.setattr(fp, "_rust_scan_files", spy)
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*"], [], {}
    )

    assert calls, (
        "the Rust fast path should have been invoked, not the slow os.walk fallback"
    )
    assert calls[0]["include_all"] is True
    assert {p.name for p in files} == {"known.py", "unknown.xyz"}


@requires_rust
def test_index_unknown_files_sentinel_parity(tmp_path, monkeypatch):
    """Rust and Python paths must agree on results for the "**/*" sentinel."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "known.py").write_text("x = 1")
    (tmp_path / "unknown.xyz").write_text("?")
    (tmp_path / "noext").write_text("?")

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")
    rust_files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*"], [], {}
    )

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    python_files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*"], [], {}
    )

    assert {str(p) for p in rust_files} == {str(p) for p in python_files}
    assert {p.name for p in rust_files} == {"known.py", "unknown.xyz", "noext"}


@requires_rust
def test_heavy_dir_anchor_conflict_declines_rust_fast_path(tmp_path, monkeypatch):
    """An include pattern explicitly anchored into a HEAVY_DIRS-named
    directory (e.g. node_modules) alongside the unrestricted "**/*" sentinel
    must decline the Rust fast path, not silently drop that subtree via
    Rust's anchor-unaware skip_dirs=HEAVY_DIRS.

    This pins down parity with the Python fallback, not full correctness:
    the fallback itself has a known pre-existing gap for this exact pattern
    combination (see the comment in walk_directory_tree) where an unrelated
    sibling directory not mentioned by any anchor is also incorrectly
    pruned. Both paths must still agree with each other.
    """
    import chunkhound.utils.file_patterns as fp

    nm = tmp_path / "node_modules"
    nm.mkdir()
    (nm / "dep.ts").write_text("x")
    (tmp_path / "top.py").write_text("x")

    patterns = ["node_modules/**/*.ts", "**/*"]

    calls = []
    monkeypatch.setattr(
        fp, "_rust_scan_files", lambda *a, **kw: calls.append(kw) or []
    )
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")
    files_with_rust_enabled, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, patterns, [], {}
    )

    assert not calls, (
        "the anchor conflict should decline the fast path entirely (both "
        "_include_all and the plain ext/name gate are False here, since "
        "both patterns are complex), falling back to the slow os.walk path"
    )
    # _rust_scan_files was never called, so this ran the real slow os.walk path
    # (not the spy's stub return value) — confirms the guard's decline actually
    # routes through to a working fallback, not a silently empty result.
    assert {p.name for p in files_with_rust_enabled} == {"dep.ts", "top.py"}

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    python_files, _ = fp.walk_directory_tree(tmp_path, tmp_path, patterns, [], {})

    assert {p.name for p in python_files} == {"dep.ts", "top.py"}


@requires_rust
def test_exclude_patterns_parity(tmp_path, monkeypatch):
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

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    python_files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py", "**/*.pyc"], exclude, {}
    )

    assert rust_files == {str(p) for p in python_files}


@requires_rust
def test_exact_names_parity(tmp_path, monkeypatch):
    """Parity: exact filename patterns like Makefile are found by both paths."""
    import chunkhound.utils.file_patterns as fp

    (tmp_path / "Makefile").write_text("all:")
    (tmp_path / "skip.py").write_text("x = 1")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "Makefile").write_text("build:")

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")
    rust_files, _ = fp.walk_directory_tree(tmp_path, tmp_path, ["Makefile"], [], {})

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "0")
    python_files, _ = fp.walk_directory_tree(tmp_path, tmp_path, ["Makefile"], [], {})

    assert {str(p) for p in rust_files} == {str(p) for p in python_files}
    assert len(rust_files) == 2


def test_max_files_forces_python_path(tmp_path, monkeypatch):
    """Rust path is bypassed when max_files is set; Python path enforces the cap."""
    import chunkhound.utils.file_patterns as fp

    for i in range(5):
        (tmp_path / f"f{i}.py").write_text("x = 1")

    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    files, _ = fp.walk_directory_tree(
        tmp_path, tmp_path, ["**/*.py"], [], {}, max_files=2
    )
    assert len(files) == 2


@requires_rust
def test_native_extra_has_exact_names():
    """Installed chunkhound-native wheel must expose the exact_names parameter."""
    import inspect
    sig = inspect.signature(_scan_files)
    assert "exact_names" in sig.parameters, (
        "chunkhound-native wheel is too old — missing exact_names. "
        "Install chunkhound-native>=0.2.0"
    )


@requires_rust
def test_native_extra_has_include_all():
    """Installed chunkhound-native wheel must expose the include_all parameter."""
    import inspect
    sig = inspect.signature(_scan_files)
    assert "include_all" in sig.parameters, (
        "chunkhound-native wheel is too old — missing include_all. "
        "Install a newer chunkhound-native."
    )
