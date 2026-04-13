"""Tests for _fsync_directory contract: fsync + fd cleanup on POSIX, no-op on Windows."""

import os
from pathlib import Path

import pytest

from chunkhound.providers.database.duckdb_provider import _fsync_directory

_MODULE = "chunkhound.providers.database.duckdb_provider"


def raise_oserror(fd: int) -> None:
    raise OSError("disk error")


class TestFsyncDirectory:
    @pytest.fixture(autouse=True)
    def _force_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(f"{_MODULE}.IS_WINDOWS", False)

    def test_fsyncs_real_directory(self, tmp_path: Path) -> None:
        _fsync_directory(tmp_path)  # must not raise

    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            _fsync_directory(tmp_path / "nonexistent")

    def test_closes_fd_on_fsync_failure(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        close_calls: list[int] = []
        original_close = os.close
        monkeypatch.setattr("os.fsync", raise_oserror)
        monkeypatch.setattr("os.close", lambda fd: (close_calls.append(fd), original_close(fd)))

        with pytest.raises(OSError, match="disk error"):
            _fsync_directory(tmp_path)

        assert len(close_calls) == 1, "fd must be closed even when fsync fails"


class TestFsyncDirectoryWindows:
    def test_noop_on_windows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(f"{_MODULE}.IS_WINDOWS", True)
        assert _fsync_directory(tmp_path) is None
