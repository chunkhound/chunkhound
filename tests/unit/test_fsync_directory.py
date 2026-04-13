"""Tests for _fsync_directory platform-agnostic contract."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chunkhound.providers.database.duckdb_provider import _fsync_directory


class TestFsyncDirectory:
    def test_valid_directory_succeeds(self, tmp_path: Path) -> None:
        assert _fsync_directory(tmp_path) is None

    @patch("chunkhound.providers.database.duckdb_provider.IS_WINDOWS", False)
    def test_raises_on_missing_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent"
        with pytest.raises(OSError):
            _fsync_directory(missing)

    @patch("chunkhound.providers.database.duckdb_provider.IS_WINDOWS", False)
    def test_closes_fd_on_fsync_failure(self, tmp_path: Path) -> None:
        with patch.object(os, "fsync", side_effect=OSError("disk error")), \
             patch.object(os, "close") as mock_close, \
             patch.object(os, "open", return_value=42):
            with pytest.raises(OSError, match="disk error"):
                _fsync_directory(tmp_path)
            mock_close.assert_called_once_with(42)
