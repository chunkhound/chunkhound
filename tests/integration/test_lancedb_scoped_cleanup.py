"""Regression tests for provider-portable scoped cleanup behavior."""

from pathlib import Path

from chunkhound.core.models import File
from chunkhound.core.types.common import Language
from chunkhound.services.indexing_coordinator import IndexingCoordinator


def test_lancedb_scoped_orphan_cleanup_preserves_files_outside_directory(
    lancedb_provider, tmp_path: Path
) -> None:
    """Scoped cleanup should only remove stale files from the selected subtree."""
    src_dir = tmp_path / "src"
    other_dir = tmp_path / "other"
    src_dir.mkdir()
    other_dir.mkdir()

    current_file = src_dir / "keep.py"
    current_file.write_text("def keep():\n    return 1\n")

    lancedb_provider.insert_file(
        File(
            path="src/keep.py",
            mtime=1.0,
            language=Language.PYTHON,
            size_bytes=10,
        )
    )
    lancedb_provider.insert_file(
        File(
            path="src/stale.py",
            mtime=1.0,
            language=Language.PYTHON,
            size_bytes=10,
        )
    )
    lancedb_provider.insert_file(
        File(
            path="other/keep.py",
            mtime=1.0,
            language=Language.PYTHON,
            size_bytes=10,
        )
    )

    coordinator = IndexingCoordinator(
        database_provider=lancedb_provider,
        base_directory=tmp_path,
    )

    orphaned_count = coordinator._cleanup_orphaned_files(src_dir, [current_file], [])

    assert orphaned_count == 1
    assert lancedb_provider.get_file_by_path("src/keep.py") is not None
    assert lancedb_provider.get_file_by_path("src/stale.py") is None
    assert lancedb_provider.get_file_by_path("other/keep.py") is not None
