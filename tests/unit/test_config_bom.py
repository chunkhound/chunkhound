"""Config files saved by Windows editors and PowerShell 5.1's
`Set-Content -Encoding utf8` begin with a UTF-8 BOM. `json.load` fails on the
BOM before any setting is read, so every config reader must accept it.

The same `utf-8-sig` readers must also surface decode failures (e.g. a UTF-16
file) as friendly config errors, not bare UnicodeDecodeError tracebacks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.core.config.config import Config

BOM = b"\xef\xbb\xbf"
MARKER = "**/bom-marker/**"
BOM_CONFIG = BOM + b'{"indexing": {"exclude": ["**/bom-marker/**"]}}'


def _write_bom_config(path: Path) -> Path:
    path.write_bytes(BOM_CONFIG)
    return path


def test_local_config_with_utf8_bom_is_loaded(
    tmp_path: Path, clean_environment
) -> None:
    _write_bom_config(tmp_path / ".chunkhound.json")

    assert MARKER in Config(target_dir=tmp_path).indexing.exclude


def test_explicit_config_with_utf8_bom_is_loaded(
    tmp_path: Path, monkeypatch, clean_environment
) -> None:
    config_file = _write_bom_config(tmp_path / "custom.json")
    monkeypatch.setenv("CHUNKHOUND_CONFIG_FILE", str(config_file))

    assert MARKER in Config(target_dir=tmp_path).indexing.exclude


def test_global_config_with_utf8_bom_is_loaded(
    tmp_path: Path, monkeypatch, clean_environment
) -> None:
    global_config = _write_bom_config(tmp_path / "global.json")
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(global_config))

    assert MARKER in Config(target_dir=tmp_path).indexing.exclude


def test_utf16_config_raises_friendly_value_error(
    tmp_path: Path, clean_environment
) -> None:
    # UTF-16 files can't be decoded by utf-8-sig; users must get a friendly
    # "Invalid JSON" error instead of a raw UnicodeDecodeError traceback.
    (tmp_path / ".chunkhound.json").write_bytes(
        '{"indexing": {"exclude": []}}'.encode("utf-16")
    )

    with pytest.raises(ValueError, match="Invalid JSON in config file"):
        Config(target_dir=tmp_path)
