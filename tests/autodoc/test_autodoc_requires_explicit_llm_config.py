from __future__ import annotations

from pathlib import Path

import pytest

from chunkhound.api.cli.commands import autodoc as autodoc_command
from chunkhound.core.config.config import Config


class _CaptureFormatter:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_autodoc_cleanup_does_not_auto_select_codex_from_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formatter = _CaptureFormatter()
    cfg = Config(target_dir=tmp_path)

    monkeypatch.setattr(autodoc_command, "_has_llm_env", lambda: False)
    monkeypatch.setattr(autodoc_command, "_codex_available", lambda: True)

    manager = autodoc_command._resolve_llm_manager(
        config=cfg, cleanup_mode="llm", formatter=formatter
    )

    assert manager is None
    assert any("No LLM provider configured" in msg for msg in formatter.warnings)

