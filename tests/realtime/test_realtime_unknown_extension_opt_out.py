"""Realtime opt-out contract for RealtimePathFilter.should_index().

Mirrors the discovery-side semantics of passes_extension_filter(): files with
unknown extensions pass when the caller opted out — via
index_unknown_files=True, or via the literal "**/*" sentinel passed directly
in include_patterns (not via config).
"""

from pathlib import Path

from chunkhound.core.config.config import Config
from chunkhound.services.realtime_path_filter import (
    RealtimePathFilter,
    RealtimePathFilterSettings,
)


def _config(root: Path, tmp_path: Path, *, index_unknown_files: bool) -> Config:
    return Config(
        **{
            "database": {"provider": "duckdb", "path": str(tmp_path / "db.duckdb")},
            "indexing": {
                "include": ["**/*.py"],
                "exclude": [],
                "exclude_sentinel": ".gitignore",
                "index_unknown_files": index_unknown_files,
            },
            "target_dir": root,
        }
    )


def _unknown_extension_file(root: Path) -> Path:
    path = root / "data.dat"
    path.write_text("payload\n", encoding="utf-8")
    return path


def test_index_unknown_files_opt_out_admits_unknown_extension(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    target = _unknown_extension_file(root)

    path_filter = RealtimePathFilter(
        config=_config(root, tmp_path, index_unknown_files=True),
        root_path=root,
    )

    assert path_filter.should_index(target) is True


def test_include_all_sentinel_admits_unknown_extension(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    target = _unknown_extension_file(root)

    # Sentinel passed directly via settings, not via config include list.
    path_filter = RealtimePathFilter(
        config=None,
        root_path=root,
        settings=RealtimePathFilterSettings(include_patterns=("**/*",)),
    )

    assert path_filter.should_index(target) is True


def test_without_opt_out_unknown_extension_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    target = _unknown_extension_file(root)

    path_filter = RealtimePathFilter(
        config=_config(root, tmp_path, index_unknown_files=False),
        root_path=root,
    )

    assert path_filter.should_index(target) is False
