"""Integration test for default include patterns discovering root-level files.

This test uses the real IndexingCoordinator with an in-memory DuckDB provider
and the default IndexingConfig include patterns. It verifies that a file placed
at the project root is discovered without any custom include/exclude rules.
"""

from types import SimpleNamespace

import pytest

from chunkhound.core.config.indexing_config import IndexingConfig
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator
from chunkhound.utils.file_patterns import normalize_include_patterns
from tests.integration.conftest import seed_py_and_png


@pytest.mark.asyncio
async def test_root_file_discovered_with_default_patterns(tmp_path):
    # Arrange: in-memory DB, Python parser, default include patterns
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    # Create a root-level file
    root_file = tmp_path / "root.py"
    root_file.write_text("print('ok')\n")

    # Use default include/exclude from IndexingConfig
    cfg = IndexingConfig()
    include_patterns = list(cfg.include)
    exclude_patterns = []

    # Act: discover files
    files = await coordinator._discover_files(
        tmp_path,
        patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        parallel_discovery=False,
    )

    # Assert: root file is present
    assert root_file in files, (
        f"Root-level file not discovered. Files: {[p.name for p in files]}"
    )


@pytest.mark.asyncio
async def test_custom_directory_wildcard_include_still_filters_unsupported_extensions(
    tmp_path,
):
    """A custom include list (blanket directory wildcard) must not bypass
    extension-support filtering. Regression test for a real customer config
    (`"Q/**/*"`-style patterns) that let every file through regardless of
    extension.
    """
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    py_file, png_file = seed_py_and_png(tmp_path)

    files = await coordinator._discover_files(
        tmp_path,
        patterns=normalize_include_patterns(["src/**/*"]),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert py_file in files
    assert png_file not in files, (
        f"Unsupported-extension file leaked through custom include. Files: "
        f"{[p.name for p in files]}"
    )


@pytest.mark.asyncio
async def test_index_unknown_files_disables_extension_filter_for_custom_include(
    tmp_path,
):
    """With index_unknown_files=True, a custom include list should behave as
    it did before this fix — every matched file comes through regardless of
    extension.
    """
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    cfg = IndexingConfig(include=["src/**/*"], index_unknown_files=True)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        SimpleNamespace(indexing=cfg),
    )

    py_file, png_file = seed_py_and_png(tmp_path)

    files = await coordinator._discover_files(
        tmp_path,
        patterns=normalize_include_patterns(list(cfg.include)),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert py_file in files
    assert png_file in files, (
        f"index_unknown_files=True should let unsupported extensions through. "
        f"Files: {[p.name for p in files]}"
    )


@pytest.mark.asyncio
async def test_explicit_unsupported_extension_pattern_is_still_discovered(tmp_path):
    """An include pattern that explicitly names an unsupported extension
    (e.g. `**/*.xyzunk`) must still be discovered even without
    index_unknown_files — that's a deliberate, specific request, distinct
    from a directory wildcard that sweeps up every extension incidentally.
    Parity with the existing "Unknown file type" skip-recording path
    (batch_processor.py), exercised by tests/integration/test_lancedb_skip_parity.py.
    """
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    py_file = tmp_path / "main.py"
    py_file.write_text("print('ok')\n")
    unk_file = tmp_path / "data.xyzunk"
    unk_file.write_text("binary\n")

    files = await coordinator._discover_files(
        tmp_path,
        patterns=normalize_include_patterns(["*.py", "*.xyzunk"]),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert py_file in files
    assert unk_file in files, (
        f"Explicitly-included unsupported extension should still be "
        f"discovered. Files: {[p.name for p in files]}"
    )


@pytest.mark.asyncio
async def test_explicit_unsupported_extension_pattern_with_directory_anchor_is_still_discovered(
    tmp_path,
):
    """A directory-anchored include pattern that explicitly names an
    unsupported extension (e.g. `src/**/*.xyzunk`) must still be discovered,
    the same as its non-anchored counterpart (`*.xyzunk`). The directory
    anchor is incidental to the request being specific — only a bare
    wildcard tail (e.g. `src/**/*`) makes it a blanket sweep. Regression
    test for PR #380 review finding #5.
    """
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / "main.py"
    py_file.write_text("print('ok')\n")
    unk_file = src_dir / "data.xyzunk"
    unk_file.write_text("binary\n")

    files = await coordinator._discover_files(
        tmp_path,
        patterns=normalize_include_patterns(["src/**/*.py", "src/**/*.xyzunk"]),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert py_file in files
    assert unk_file in files, (
        f"Directory-anchored explicit include of an unsupported extension "
        f"should still be discovered. Files: {[p.name for p in files]}"
    )


@pytest.mark.asyncio
async def test_explicit_pattern_carveout_is_case_insensitive(tmp_path, monkeypatch):
    """The explicit-pattern carve-out must match case-insensitively.

    The Rust fast walker (`scan_files` in src/lib.rs) lowercases extensions
    before comparing, so a pattern written as `*.xyzunk` already discovers a
    file with an upper-case extension via the Rust path (default: on). The
    carve-out in `_filter_unsupported_extensions` must not then drop it for
    being a case-sensitive mismatch against the pattern as literally written.
    """
    monkeypatch.setenv("CHUNKHOUND_USE_RUST", "1")

    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()

    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(
        db,
        tmp_path,
        None,
        {Language.PYTHON: parser},
        None,
        None,
    )

    py_file = tmp_path / "main.py"
    py_file.write_text("print('ok')\n")
    unk_file = tmp_path / "DATA.XYZUNK"
    unk_file.write_text("binary\n")

    files = await coordinator._discover_files(
        tmp_path,
        patterns=normalize_include_patterns(["*.py", "*.xyzunk"]),
        exclude_patterns=[],
        parallel_discovery=False,
    )

    assert py_file in files
    assert unk_file in files, (
        f"Case-mismatched but explicitly-included extension should still be "
        f"discovered. Files: {[p.name for p in files]}"
    )
