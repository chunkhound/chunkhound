import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.helpers import DummyPipe, DummyProc


@pytest.mark.asyncio
async def test_codex_overlay_cleanup(monkeypatch, tmp_path: Path):
    """Ensure Codex provider cleans up overlay CODEX_HOME after exec.

    We stub out the subprocess call and force a known overlay path using
    `_build_overlay_home()`. After `_run_exec()` returns, the overlay should
    be removed regardless of success.
    """
    from chunkhound.providers.llm.codex_cli_provider import CodexCLIProvider
    # Force provider to consider Codex available
    monkeypatch.setattr(CodexCLIProvider, "_codex_available", lambda self: True, raising=True)

    # Create a deterministic overlay directory
    overlay_dir = tmp_path / "overlay-home"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch build to return our overlay path
    requested_model = {}

    def _fake_overlay_home(self, model_override=None):
        requested_model["value"] = model_override
        return str(overlay_dir)

    monkeypatch.setattr(CodexCLIProvider, "_build_overlay_home", _fake_overlay_home, raising=True)

    # Stub out subprocess creation to avoid calling real codex
    async def _fake_create_subprocess_exec(*args, **kwargs):  # noqa: ANN001
        return DummyProc(rc=0, out=b"OK", err=b"", stdin=DummyPipe())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec, raising=True)

    with patch.object(
        CodexCLIProvider,
        "get_highest_priority_available_model",
        return_value="test-discovered-model",
    ):
        prov = CodexCLIProvider(model="codex")

        # Sanity: overlay exists before run
        assert overlay_dir.exists()

        # Run via argv path (short content) and ensure success
        out = await prov._run_exec("ping", cwd=None, max_tokens=16, timeout=10, model="codex")  # type: ignore[attr-defined]
        assert out.strip() == "OK"

    # Overlay should be cleaned up by provider
    assert not overlay_dir.exists(), "overlay CODEX_HOME was not cleaned up"
    # "codex" alias should resolve to the mocked discovered model
    assert requested_model.get("value") == "test-discovered-model"
