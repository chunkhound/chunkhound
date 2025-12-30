from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chunkhound.api.cli.commands import autodoc as autodoc_cmd


class _FakeFormatter:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


def test_confirm_autorun_exits_on_decline(monkeypatch) -> None:
    monkeypatch.setattr(
        autodoc_cmd,
        "_code_mapper_autorun_prereq_summary",
        lambda **_kwargs: (True, [], []),
    )
    monkeypatch.setattr(autodoc_cmd, "_prompt_yes_no", lambda *_a, **_k: False)

    formatter = _FakeFormatter()
    with pytest.raises(SystemExit) as excinfo:
        autodoc_cmd._confirm_autorun_and_validate_prereqs(
            config=SimpleNamespace(),  # type: ignore[arg-type]
            config_path=None,
            formatter=formatter,  # type: ignore[arg-type]
            question="Proceed?",
            decline_error="nope",
            decline_exit_code=7,
        )

    assert excinfo.value.code == 7
    assert formatter.errors == ["nope"]


def test_confirm_autorun_exits_on_prereq_failure(monkeypatch) -> None:
    calls: list[tuple[bool, list[str], list[str]]] = [
        (False, ["database"], ["- missing database"]),
        (False, ["database"], ["- missing database"]),
    ]

    def fake_summary(**_kwargs):
        return calls.pop(0)

    monkeypatch.setattr(
        autodoc_cmd,
        "_code_mapper_autorun_prereq_summary",
        fake_summary,
    )
    monkeypatch.setattr(autodoc_cmd, "_prompt_yes_no", lambda *_a, **_k: True)

    captured: dict[str, object] = {}

    def fake_exit(*, formatter, details, exit_code):  # type: ignore[no-untyped-def]
        captured["details"] = details
        raise SystemExit(exit_code)

    monkeypatch.setattr(autodoc_cmd, "_exit_autodoc_autorun_prereq_failure", fake_exit)

    formatter = _FakeFormatter()
    with pytest.raises(SystemExit) as excinfo:
        autodoc_cmd._confirm_autorun_and_validate_prereqs(
            config=SimpleNamespace(),  # type: ignore[arg-type]
            config_path=Path("cfg.json"),
            formatter=formatter,  # type: ignore[arg-type]
            question="Proceed?",
            decline_error="nope",
            decline_exit_code=7,
            prereq_failure_exit_code=3,
        )

    assert excinfo.value.code == 3
    assert captured["details"] == ["- missing database"]
