from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc_autorun as autorun
from chunkhound.api.cli.commands.autodoc_errors import AutoDocCLIExitError


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
