"""Unit-test conftest: disable auto-compaction dispatch.

The 1/N random-sampling in serial_executor.py would introduce
non-deterministic compaction during arbitrary test operations.
We disable the sampling unconditionally here so all tests in
this directory are deterministic.
"""

import os
import subprocess

import pytest

_CLI_VERSION_PROBES = frozenset({"codex", "opencode"})


def _stub_cli_version_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real ``codex``/``opencode --version`` calls. Every other command runs."""
    real_run = subprocess.run

    def _run(args, **kwargs):  # noqa: ANN001, ANN003
        if (
            isinstance(args, (list, tuple))
            and len(args) >= 2
            and args[-1] == "--version"
            and os.path.splitext(os.path.basename(str(args[-2])).lower())[0]
            in _CLI_VERSION_PROBES
        ):
            return subprocess.CompletedProcess(args, 0)
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _run)


@pytest.fixture(autouse=True)
def _block_cli_version_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructors probe ``--version``. A resolver stub must not launch a CLI."""
    _stub_cli_version_probe(monkeypatch)


@pytest.fixture(autouse=True)
def _disable_auto_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent auto-compaction dispatch from firing during tests."""
    monkeypatch.setattr(
        "chunkhound.providers.database.serial_executor.COMPACT_SAMPLE_INTERVAL", 0
    )
