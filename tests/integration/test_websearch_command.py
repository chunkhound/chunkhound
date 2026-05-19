"""Integration tests for the CLI ``websearch_command`` entry point.

Mirrors the in-process patching style of ``tests/unit/test_websearch_mcp_errors.py``:
imports the async command directly, stubs ``search`` / ``fetch_and_save``
/ ``subprocess.run`` / ``_build_quickresearch_argv``, and asserts on exit
codes, tmpdir creation, and subprocess invocation flags.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import urllib.error
from pathlib import Path

import pytest

from chunkhound.api.cli.commands import websearch as ws_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_results() -> list[tuple[str, str, str]]:
    return [
        ("Title A", "https://a.invalid/", "desc a"),
        ("Title B", "https://b.invalid/", "desc b"),
        ("Title C", "https://c.invalid/", "desc c"),
    ]


def _make_args(**overrides) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "query": "q",
        "limit": 30,
        "verbose": False,
        "debug": False,
        "config": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


async def _noop_fetch_and_save(
    urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
):
    return None


def _stub_search(results):
    def _inner(query, limit=30, progress_callback=None):
        return list(results)

    return _inner


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def patched(monkeypatch):
    """Patch the argv builder and capture tmpdir creations."""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def capturing_mkdtemp(*args, **kwargs):
        p = real_mkdtemp(*args, **kwargs)
        created.append(p)
        return p

    def fake_argv(args, tmpdir, config):
        return ["/bin/true"]

    monkeypatch.setattr(ws_mod, "_build_quickresearch_argv", fake_argv)
    monkeypatch.setattr(ws_mod.tempfile, "mkdtemp", capturing_mkdtemp)
    return {"tmpdirs": created}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websearch_command_urlerror_exits_1(monkeypatch, patched) -> None:
    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(ws_mod, "search", raise_url_error)

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(_make_args(), config=None)

    assert exc.value.code == 1
    # Nothing should have been fetched or run.
    assert patched["tmpdirs"] == []


@pytest.mark.asyncio
async def test_websearch_command_empty_results_returns_without_exit(
    monkeypatch, patched
) -> None:
    monkeypatch.setattr(ws_mod, "search", _stub_search([]))

    errors: list[str] = []
    monkeypatch.setattr(
        ws_mod.RichOutputFormatter, "error", lambda self, msg: errors.append(msg)
    )

    # No SystemExit — the command reports an error and returns.
    await ws_mod.websearch_command(_make_args(query="zero-hits"), config=None)

    # Empty results short-circuit before mkdtemp.
    assert patched["tmpdirs"] == []
    # The error must surface the query so the user knows what failed.
    assert len(errors) == 1
    assert "'zero-hits'" in errors[0]


@pytest.mark.asyncio
async def test_websearch_command_happy_path_runs_subprocess(
    monkeypatch, patched
) -> None:
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _noop_fetch_and_save)

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)

        class _R:
            returncode = 0
            stdout = ""

        return _R()

    monkeypatch.setattr(ws_mod.subprocess, "run", fake_run)

    await ws_mod.websearch_command(_make_args(), config=None)

    assert captured["cmd"] == ["/bin/true"]
    assert captured.get("check") is True
    assert captured.get("stdin") is subprocess.DEVNULL
    env = captured.get("env")
    assert isinstance(env, dict) and env.get("CHUNKHOUND_NO_PROMPTS") == "1"
    assert len(patched["tmpdirs"]) == 1
    assert Path(patched["tmpdirs"][0]).name.startswith("chunkhound_websearch_")


@pytest.mark.asyncio
async def test_websearch_command_subprocess_failure_exits_with_returncode(
    monkeypatch, patched
) -> None:
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _noop_fetch_and_save)

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(ws_mod.subprocess, "run", fake_run)

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(_make_args(), config=None)

    assert exc.value.code == 2


@pytest.mark.asyncio
async def test_websearch_command_fetch_and_save_receives_only_urls(
    monkeypatch, patched
) -> None:
    results = _default_results()
    monkeypatch.setattr(ws_mod, "search", _stub_search(results))
    received: dict[str, object] = {}

    async def capturing_fetch(
        urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
    ):
        received["urls"] = list(urls)
        received["tmpdir"] = tmpdir

    monkeypatch.setattr(ws_mod, "fetch_and_save", capturing_fetch)
    monkeypatch.setattr(
        ws_mod.subprocess, "run",
        lambda cmd, **kw: type("R", (), {"returncode": 0, "stdout": ""})(),
    )

    await ws_mod.websearch_command(_make_args(), config=None)

    assert received["urls"] == [url for _, url, _ in results]
    assert isinstance(received["tmpdir"], Path)
    assert str(received["tmpdir"]) == patched["tmpdirs"][0]
