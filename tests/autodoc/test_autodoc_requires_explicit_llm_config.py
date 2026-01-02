from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands.autodoc_cleanup import resolve_cleanup_config_and_llm_manager
from chunkhound.api.cli.commands.autodoc_errors import AutoDocCLIExit
from chunkhound.core.config.config import Config


class _CaptureFormatter:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def test_autodoc_cleanup_requires_explicit_llm_config(tmp_path: Path) -> None:
    formatter = _CaptureFormatter()
    cfg = Config(target_dir=tmp_path)
    cfg.llm = None  # type: ignore[assignment]

    args = SimpleNamespace(
        cleanup_mode="llm",
        cleanup_batch_size=1,
        cleanup_max_tokens=512,
        audience="balanced",
    )

    with pytest.raises(AutoDocCLIExit) as excinfo:
        resolve_cleanup_config_and_llm_manager(args=args, config=cfg, formatter=formatter)  # type: ignore[arg-type]

    assert excinfo.value.exit_code == 2
    assert any("AutoDoc cleanup requires an LLM provider" in msg for msg in excinfo.value.errors)

