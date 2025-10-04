"""Integration tests for parallel directory discovery.

Verifies that parallel and sequential file discovery produce identical results,
especially with .gitignore patterns and root-level files.
"""

import pytest
from pathlib import Path
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


@pytest.fixture
async def coordinator(tmp_path):
    """Create test coordinator with config."""
    from chunkhound.core.config.config import Config
    import json

    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    parser = create_parser_for_language(Language.PYTHON)

    # Create .chunkhound.json for Config initialization
    config_file = tmp_path / ".chunkhound.json"
    config_file.write_text(json.dumps({"version": "1.0"}))

    # Create minimal config for testing
    config = Config(target_dir=tmp_path)

    coordinator = IndexingCoordinator(
        db, tmp_path, None, {Language.PYTHON: parser}, None, config
    )
    return coordinator


@pytest.fixture
def test_repo(tmp_path):
    """Create a test repository with .gitignore and nested structure."""
    # Create root .gitignore
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.log\n*.tmp\ndebug/\n")

    # Create root-level files
    (tmp_path / "root_file.py").write_text("# root file")
    (tmp_path / "debug.log").write_text("should be ignored")
    (tmp_path / "temp.tmp").write_text("should be ignored")

    # Create nested directories
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("# main file")
    (src_dir / "test.log").write_text("should be ignored")

    # Create subdirectory with its own gitignore
    sub_dir = src_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / ".gitignore").write_text("*.cache\n")
    (sub_dir / "code.py").write_text("# code file")
    (sub_dir / "data.cache").write_text("should be ignored")

    # Create debug directory (should be excluded by root .gitignore)
    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    (debug_dir / "debug.py").write_text("# should be ignored")

    return tmp_path


@pytest.mark.asyncio
async def test_parallel_matches_sequential(coordinator, test_repo):
    """Verify parallel and sequential discovery produce identical results."""
    patterns = ["**/*.py"]
    exclude_patterns = []

    # Run sequential discovery (explicitly disable parallel)
    sequential_files = await coordinator._discover_files(
        test_repo, patterns, exclude_patterns, parallel_discovery=False
    )

    # Run parallel discovery (explicitly enable parallel)
    parallel_files = await coordinator._discover_files(
        test_repo, patterns, exclude_patterns, parallel_discovery=True
    )

    # Convert to sets for comparison
    sequential_set = set(sequential_files)
    parallel_set = set(parallel_files)

    # Should have identical results
    assert sequential_set == parallel_set, (
        f"Mismatch between parallel and sequential:\n"
        f"Only in sequential: {sequential_set - parallel_set}\n"
        f"Only in parallel: {parallel_set - sequential_set}"
    )

    # Should find exactly 3 .py files (root_file.py, main.py, code.py)
    # Should NOT find debug.py (excluded by .gitignore directory pattern)
    assert len(parallel_files) == 3, f"Expected 3 files, got {len(parallel_files)}"

    # Verify expected files are present
    filenames = {f.name for f in parallel_files}
    assert "root_file.py" in filenames, "Should find root-level .py file"
    assert "main.py" in filenames, "Should find nested .py file"
    assert "code.py" in filenames, "Should find deeply nested .py file"
    assert "debug.py" not in filenames, "Should exclude files in gitignored directory"


@pytest.mark.asyncio
async def test_gitignore_excludes_root_files(coordinator, test_repo):
    """Verify .gitignore patterns exclude root-level files in parallel mode."""
    patterns = ["**/*"]
    exclude_patterns = []

    # Run parallel discovery (explicitly enabled)
    parallel_files = await coordinator._discover_files(
        test_repo, patterns, exclude_patterns, parallel_discovery=True
    )

    filenames = {f.name for f in parallel_files}

    # Root .gitignore should exclude these
    assert "debug.log" not in filenames, "Should exclude .log files at root"
    assert "temp.tmp" not in filenames, "Should exclude .tmp files at root"
    assert "test.log" not in filenames, "Should exclude .log files in subdirs"
    assert "data.cache" not in filenames, "Should exclude .cache files from subdir gitignore"


@pytest.mark.asyncio
async def test_parallel_discovery_with_few_dirs(coordinator, tmp_path):
    """Verify fallback to sequential when few top-level directories."""
    # Create small structure (< min_dirs_for_parallel threshold)
    (tmp_path / "file1.py").write_text("# file 1")
    (tmp_path / "file2.py").write_text("# file 2")
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "file3.py").write_text("# file 3")

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Should fall back to sequential (< min_dirs_for_parallel threshold)
    files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )

    assert len(files) == 3, "Should find all 3 .py files"


@pytest.mark.asyncio
async def test_parallel_discovery_with_many_dirs(coordinator, tmp_path):
    """Verify parallel execution with many top-level directories."""
    # Create many top-level directories to trigger parallel mode
    for i in range(10):
        dir_path = tmp_path / f"dir{i}"
        dir_path.mkdir()
        (dir_path / f"file{i}.py").write_text(f"# file {i}")

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Compare sequential vs parallel (>= min_dirs_for_parallel threshold)
    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )

    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )

    assert set(sequential_files) == set(parallel_files)
    assert len(parallel_files) == 10, "Should find all 10 .py files"


@pytest.mark.asyncio
async def test_nested_gitignore_inheritance(coordinator, tmp_path):
    """Verify nested .gitignore files are properly inherited in parallel mode."""
    # Root .gitignore
    (tmp_path / ".gitignore").write_text("*.log\n")

    # Create nested structure
    level1 = tmp_path / "level1"
    level1.mkdir()
    (level1 / "file1.py").write_text("# level 1")
    (level1 / "debug.log").write_text("should be excluded")

    # Nested .gitignore adding more patterns
    (level1 / ".gitignore").write_text("*.tmp\n")

    level2 = level1 / "level2"
    level2.mkdir()
    (level2 / "file2.py").write_text("# level 2")
    (level2 / "test.log").write_text("should be excluded by root gitignore")
    (level2 / "data.tmp").write_text("should be excluded by level1 gitignore")

    patterns = ["**/*"]
    exclude_patterns = []

    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )

    filenames = {f.name for f in parallel_files}

    # Should find .py files
    assert "file1.py" in filenames
    assert "file2.py" in filenames

    # Should exclude based on inherited patterns
    assert "debug.log" not in filenames, "Level 1 .log excluded by root gitignore"
    assert "test.log" not in filenames, "Level 2 .log excluded by root gitignore"
    assert "data.tmp" not in filenames, "Level 2 .tmp excluded by level1 gitignore"


@pytest.mark.asyncio
async def test_permission_error_fallback(coordinator, tmp_path):
    """Verify graceful fallback when encountering permission errors."""
    import os
    import stat

    # Create test structure
    accessible_dir = tmp_path / "accessible"
    accessible_dir.mkdir()
    (accessible_dir / "file.py").write_text("# accessible")

    restricted_dir = tmp_path / "restricted"
    restricted_dir.mkdir()
    (restricted_dir / "hidden.py").write_text("# should not be found")

    # Remove read permissions from restricted directory
    # Note: This test may not work on all platforms (Windows, root user)
    try:
        os.chmod(restricted_dir, stat.S_IWUSR)
    except (OSError, PermissionError):
        pytest.skip("Cannot change permissions on this platform")

    patterns = ["**/*.py"]
    exclude_patterns = []

    try:
        # Should complete without crashing despite permission error
        files = await coordinator._discover_files(
            tmp_path, patterns, exclude_patterns, parallel_discovery=True
        )

        # Should find accessible file, skip restricted
        filenames = {f.name for f in files}
        assert "file.py" in filenames
        assert "hidden.py" not in filenames
    finally:
        # Restore permissions for cleanup
        try:
            os.chmod(restricted_dir, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        except (OSError, PermissionError):
            pass


@pytest.mark.asyncio
async def test_symbolic_link_handling(coordinator, tmp_path):
    """Verify symbolic links are NOT followed (prevents infinite loops).

    SECURITY: Symlinks are intentionally not followed to avoid:
    - Infinite loops from circular symlinks
    - Indexing content multiple times
    - Traversing outside intended directory tree
    """
    # Create actual directory with file
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    (real_dir / "code.py").write_text("# real file")

    # Create symbolic link to directory
    link_dir = tmp_path / "link"
    try:
        link_dir.symlink_to(real_dir)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links not supported on this platform")

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Parallel discovery
    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )

    # Sequential discovery
    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )

    # Both should handle symlinks consistently (not following them)
    assert len(parallel_files) == len(sequential_files)

    # Should find the file through real path only (not through symlink)
    assert any("code.py" in str(f) for f in parallel_files)
    # File should be found exactly once (no duplication through symlink)
    code_files = [f for f in parallel_files if "code.py" in str(f)]
    assert len(code_files) == 1


@pytest.mark.asyncio
async def test_symlink_loop_protection(coordinator, tmp_path):
    """Verify that circular symlinks don't cause infinite loops.

    SAFETY: os.walk() with followlinks=False (default) prevents infinite recursion.
    """
    # Create directory structure with circular symlink
    dir_a = tmp_path / "dir_a"
    dir_a.mkdir()
    (dir_a / "file_a.py").write_text("# file in dir_a")

    dir_b = dir_a / "dir_b"
    dir_b.mkdir()
    (dir_b / "file_b.py").write_text("# file in dir_b")

    # Create circular symlink: dir_b/link_to_a -> dir_a
    link_to_a = dir_b / "link_to_a"
    try:
        link_to_a.symlink_to(dir_a)
    except (OSError, NotImplementedError):
        pytest.skip("Symbolic links not supported on this platform")

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Should complete without infinite loop
    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )

    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )

    # Should find exactly 2 files (not infinite due to loop)
    assert len(parallel_files) == 2
    assert len(sequential_files) == 2
    assert set(parallel_files) == set(sequential_files)

    # Verify we found the expected files
    filenames = {f.name for f in parallel_files}
    assert "file_a.py" in filenames
    assert "file_b.py" in filenames


@pytest.mark.asyncio
async def test_parallel_discovery_fallback_signal(coordinator, tmp_path):
    """Verify parallel discovery returns None for fallback to sequential."""
    patterns = ["**/*.py"]
    exclude_patterns = []

    # Test with too few directories (should return None for fallback)
    single_dir = tmp_path / "single"
    single_dir.mkdir()
    (single_dir / "file.py").write_text("# file")

    files = await coordinator._discover_files_parallel(
        tmp_path, patterns, exclude_patterns
    )

    # Should return None due to < 4 top-level dirs (signals fallback)
    assert files is None

    # Test with enough directories (should return files)
    for i in range(5):
        dir_path = tmp_path / f"dir{i}"
        dir_path.mkdir()
        (dir_path / "file.py").write_text(f"# file {i}")

    files = await coordinator._discover_files_parallel(
        tmp_path, patterns, exclude_patterns
    )

    # Should succeed with parallel and return files
    assert files is not None
    assert len(files) >= 5  # At least 5 files from the directories


@pytest.mark.asyncio
async def test_config_values_respected(coordinator, tmp_path):
    """Verify that config values for parallel discovery are actually used."""
    # Create directory structure with 6 top-level dirs (above default threshold of 4)
    for i in range(6):
        dir_path = tmp_path / f"dir{i}"
        dir_path.mkdir()
        (dir_path / "file.py").write_text(f"# file {i}")

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Test 1: With default config (min_dirs=4), should use parallel
    files = await coordinator._discover_files_parallel(
        tmp_path, patterns, exclude_patterns
    )
    assert files is not None, "Should use parallel with 6 dirs (>= 4 threshold)"

    # Test 2: Modify config to require 10 dirs, should fallback
    coordinator.config.indexing.min_dirs_for_parallel = 10
    files = await coordinator._discover_files_parallel(
        tmp_path, patterns, exclude_patterns
    )
    assert files is None, "Should fallback with 6 dirs (< 10 threshold)"

    # Test 3: Restore to 4, should use parallel again
    coordinator.config.indexing.min_dirs_for_parallel = 4
    files = await coordinator._discover_files_parallel(
        tmp_path, patterns, exclude_patterns
    )
    assert files is not None, "Should use parallel with 6 dirs (>= 4 threshold)"
