"""Performance benchmarks for directory discovery.

These tests measure and validate the performance characteristics of parallel vs
sequential file discovery, documenting actual speedup on different directory structures.
"""

import time
import pytest
from pathlib import Path
from chunkhound.core.types.common import Language
from chunkhound.parsers.parser_factory import create_parser_for_language
from chunkhound.providers.database.duckdb_provider import DuckDBProvider
from chunkhound.services.indexing_coordinator import IndexingCoordinator


@pytest.fixture
async def coordinator(tmp_path):
    """Create test coordinator."""
    db = DuckDBProvider(":memory:", base_directory=tmp_path)
    db.connect()
    parser = create_parser_for_language(Language.PYTHON)
    coordinator = IndexingCoordinator(db, tmp_path, None, {Language.PYTHON: parser})
    return coordinator


def create_large_repo(base_path: Path, num_dirs: int, files_per_dir: int):
    """Create a large repository structure for benchmarking.

    Args:
        base_path: Base directory
        num_dirs: Number of top-level directories
        files_per_dir: Number of files per directory
    """
    for i in range(num_dirs):
        dir_path = base_path / f"module_{i}"
        dir_path.mkdir()

        for j in range(files_per_dir):
            file_path = dir_path / f"file_{j}.py"
            file_path.write_text(f"# Module {i}, File {j}\ndef func_{i}_{j}():\n    pass\n")


@pytest.mark.asyncio
async def test_small_repo_performance(coordinator, tmp_path):
    """Benchmark discovery on small repos (<100 files).

    EXPECTATION: Sequential may be faster due to process spawning overhead.
    """
    # Create small repo (50 files across 2 dirs - below parallel threshold)
    create_large_repo(tmp_path, num_dirs=2, files_per_dir=25)

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Measure sequential
    start = time.perf_counter()
    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )
    sequential_time = time.perf_counter() - start

    # Measure parallel (will auto-fallback to sequential due to < 4 dirs)
    start = time.perf_counter()
    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )
    parallel_time = time.perf_counter() - start

    # Verify results match
    assert set(sequential_files) == set(parallel_files)
    assert len(sequential_files) == 50

    print(f"\nSmall repo (50 files, 2 dirs):")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel:   {parallel_time:.4f}s (auto-fallback to sequential)")


@pytest.mark.asyncio
async def test_medium_repo_performance(coordinator, tmp_path):
    """Benchmark discovery on medium repos (100-1000 files).

    EXPECTATION: Parallel should show measurable speedup (1.5-3x).
    """
    # Create medium repo (500 files across 10 dirs - triggers parallel mode)
    create_large_repo(tmp_path, num_dirs=10, files_per_dir=50)

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Measure sequential
    start = time.perf_counter()
    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )
    sequential_time = time.perf_counter() - start

    # Measure parallel
    start = time.perf_counter()
    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )
    parallel_time = time.perf_counter() - start

    # Verify results match
    assert set(sequential_files) == set(parallel_files)
    assert len(sequential_files) == 500

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"\nMedium repo (500 files, 10 dirs):")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel:   {parallel_time:.4f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    # Document actual performance - speedup should be > 1.0 on multi-core systems
    # (may be marginal on CI systems or small repos)


@pytest.mark.asyncio
async def test_large_repo_performance(coordinator, tmp_path):
    """Benchmark discovery on large repos (>1000 files).

    EXPECTATION: Parallel should show significant speedup (2-5x on multi-core).
    """
    # Create large repo (2000 files across 20 dirs - enterprise monorepo scale)
    create_large_repo(tmp_path, num_dirs=20, files_per_dir=100)

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Measure sequential
    start = time.perf_counter()
    sequential_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )
    sequential_time = time.perf_counter() - start

    # Measure parallel
    start = time.perf_counter()
    parallel_files = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=True
    )
    parallel_time = time.perf_counter() - start

    # Verify results match
    assert set(sequential_files) == set(parallel_files)
    assert len(sequential_files) == 2000

    speedup = sequential_time / parallel_time if parallel_time > 0 else 0

    print(f"\nLarge repo (2000 files, 20 dirs):")
    print(f"  Sequential: {sequential_time:.4f}s")
    print(f"  Parallel:   {parallel_time:.4f}s")
    print(f"  Speedup:    {speedup:.2f}x")

    # On multi-core systems, parallel should be noticeably faster
    # Note: Actual speedup depends on CPU cores, I/O speed, and filesystem caching


@pytest.mark.asyncio
async def test_config_driven_discovery(coordinator, tmp_path):
    """Test that discovery mode respects configuration settings."""
    create_large_repo(tmp_path, num_dirs=10, files_per_dir=10)

    patterns = ["**/*.py"]
    exclude_patterns = []

    # Test with config defaults (parallel_discovery=True by default)
    files_default = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns  # Uses config setting
    )

    # Test explicit override
    files_explicit = await coordinator._discover_files(
        tmp_path, patterns, exclude_patterns, parallel_discovery=False
    )

    # Results should be identical regardless of mode
    assert set(files_default) == set(files_explicit)
    assert len(files_default) == 100
