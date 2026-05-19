from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc_autorun as autorun
from chunkhound.api.cli.commands.autodoc_errors import AutoDocCLIExitError
from chunkhound.core.config.config import Config


def test_confirm_autorun_exits_on_decline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        autorun,
        "code_mapper_autorun_prereq_summary",
        lambda **_kwargs: (True, [], []),
    )
    monkeypatch.setattr(autorun.prompts, "prompt_yes_no", lambda *_a, **_k: False)

    with pytest.raises(AutoDocCLIExitError) as excinfo:
        autorun.confirm_autorun_and_validate_prereqs(
            config=SimpleNamespace(),  # type: ignore[arg-type]
            config_path=None,
            question="Proceed?",
            decline_error="nope",
            decline_exit_code=7,
        )

    assert excinfo.value.exit_code == 7
    assert excinfo.value.errors == ("nope",)


def test_confirm_autorun_exits_on_prereq_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[bool, list[str], list[str]]] = [
        (False, ["database"], ["- missing database"]),
        (False, ["database"], ["- missing database"]),
    ]

    def fake_summary(**_kwargs):  # type: ignore[no-untyped-def]
        return calls.pop(0)

    monkeypatch.setattr(autorun, "code_mapper_autorun_prereq_summary", fake_summary)
    monkeypatch.setattr(autorun.prompts, "prompt_yes_no", lambda *_a, **_k: True)

    with pytest.raises(AutoDocCLIExitError) as excinfo:
        autorun.confirm_autorun_and_validate_prereqs(
            config=SimpleNamespace(),  # type: ignore[arg-type]
            config_path=Path("cfg.json"),
            question="Proceed?",
            decline_error="nope",
            decline_exit_code=7,
            prereq_failure_exit_code=3,
        )

    assert excinfo.value.exit_code == 3
    assert any("prerequisites are missing" in msg for msg in excinfo.value.errors)
    assert "- missing database" in excinfo.value.errors


def test_autorun_llm_prereqs_accept_local_openai_compatible_llm_without_api_key(
    clean_environment,
    tmp_path: Path,
) -> None:
    cfg = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "model": "llama3.2",
            "base_url": "http://localhost:11434/v1",
        }
    )

    missing, details = autorun._code_mapper_autorun_llm_prereqs(cfg)

    assert missing == []
    assert details == []


def test_autorun_llm_prereqs_require_explicit_model_for_custom_endpoint(
    clean_environment,
    tmp_path: Path,
) -> None:
    cfg = Config(
        target_dir=tmp_path,
        llm={
            "provider": "openai",
            "base_url": "http://localhost:11434/v1",
        }
    )

    missing, details = autorun._code_mapper_autorun_llm_prereqs(cfg)

    assert missing == ["llm"]
    assert any("explicit model selection required" in detail for detail in details)
