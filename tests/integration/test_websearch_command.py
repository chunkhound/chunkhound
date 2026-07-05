"""Integration tests for the CLI ``websearch_command`` entry point.

Mirrors the in-process patching style of ``tests/unit/test_websearch_mcp_errors.py``:
imports the async command directly, stubs ``fetch_and_save`` / ``subprocess.run``
/ ``_build_quickresearch_argv`` / ``setup_llm_manager`` on the command module,
stubs ``search`` on ``websearch_core`` (its definition module — see note at the
first patch site), and asserts on exit codes, tmpdir creation, and subprocess
invocation flags.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import urllib.error
from pathlib import Path

import pytest

from chunkhound.api.cli.commands import websearch as ws_mod
from chunkhound.utils import websearch_core as ws_core_mod

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
    monkeypatch.setattr(ws_mod, "setup_llm_manager", lambda formatter, config: None)
    return {"tmpdirs": created}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_websearch_command_urlerror_exits_1(monkeypatch, patched) -> None:
    def raise_url_error(*args, **kwargs):
        raise urllib.error.URLError("boom")

    # Patched at definition site: ``search_multi`` binds ``search`` from its
    # own module scope, so patching ``ws_mod.search`` would not intercept it.
    monkeypatch.setattr(ws_core_mod, "search", raise_url_error)

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(_make_args(), config=None)

    assert exc.value.code == 1
    # Nothing should have been fetched or run.
    assert patched["tmpdirs"] == []


@pytest.mark.asyncio
async def test_websearch_command_empty_results_exits_10(
    monkeypatch, patched
) -> None:
    monkeypatch.setattr(ws_core_mod, "search", _stub_search([]))

    errors: list[str] = []
    monkeypatch.setattr(
        ws_mod.RichOutputFormatter, "error", lambda self, msg: errors.append(msg)
    )

    with pytest.raises(SystemExit) as exc:
        await ws_mod.websearch_command(_make_args(query="zero-hits"), config=None)

    assert exc.value.code == 10
    # Empty results short-circuit before mkdtemp.
    assert patched["tmpdirs"] == []
    # The error must surface the query so the user knows what failed.
    assert len(errors) == 1
    assert "'zero-hits'" in errors[0]


@pytest.mark.asyncio
async def test_websearch_command_happy_path_runs_subprocess(
    monkeypatch, patched
) -> None:
    monkeypatch.setattr(ws_core_mod, "search", _stub_search(_default_results()))
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
    monkeypatch.setattr(ws_core_mod, "search", _stub_search(_default_results()))
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
    monkeypatch.setattr(ws_core_mod, "search", _stub_search(results))
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


@pytest.fixture
def patched_capturing(monkeypatch):
    """Run the real argv builder and capture the cmd handed to subprocess.run."""
    captured: dict[str, object] = {}
    tmpdirs: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def capturing_mkdtemp(*args, **kwargs):
        p = real_mkdtemp(*args, **kwargs)
        tmpdirs.append(p)
        return p

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["kwargs"] = kwargs

        class _R:
            returncode = 0
            stdout = ""

        return _R()

    monkeypatch.setattr(ws_core_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _noop_fetch_and_save)
    monkeypatch.setattr(ws_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(ws_mod.tempfile, "mkdtemp", capturing_mkdtemp)
    monkeypatch.setattr(ws_mod, "setup_llm_manager", lambda formatter, config: None)
    return {"captured": captured, "tmpdirs": tmpdirs}


@pytest.mark.asyncio
async def test_websearch_command_forwards_boolean_optional_negative(
    patched_capturing,
) -> None:
    """``--no-llm-ssl-verify`` must reach the child as the negative flag."""
    from chunkhound.core.config.config import Config

    args = _make_args(llm_ssl_verify=False)
    await ws_mod.websearch_command(args, config=Config())

    cmd = patched_capturing["captured"]["cmd"]
    assert isinstance(cmd, list)
    assert "--no-llm-ssl-verify" in cmd
    assert "--llm-ssl-verify" not in cmd
    assert "True" not in cmd and "False" not in cmd


@pytest.mark.asyncio
async def test_websearch_command_forwards_config_flags_end_to_end(
    patched_capturing,
) -> None:
    """Scalar flags and a positive boolean-optional round-trip through the real builder."""
    from chunkhound.core.config.config import Config

    args = _make_args(
        llm_base_url="https://example.test",
        llm_api_key="sekret",
        llm_synthesis_model="gpt-x",
        provider="openai",
        ssl_verify=True,
    )
    await ws_mod.websearch_command(args, config=Config())

    cmd = patched_capturing["captured"]["cmd"]
    assert isinstance(cmd, list) and cmd
    assert "_quickresearch" in cmd
    # Multi-alias dests forward under option_strings[0] (e.g. --provider, not
    # --embedding-provider); both are valid argparse aliases for the same dest.
    for flag, value in [
        ("--llm-base-url", "https://example.test"),
        ("--llm-api-key", "sekret"),
        ("--llm-synthesis-model", "gpt-x"),
        ("--provider", "openai"),
    ]:
        assert flag in cmd, f"missing {flag} in {cmd}"
        assert cmd[cmd.index(flag) + 1] == value
    assert "--ssl-verify" in cmd
    assert "True" not in cmd and "False" not in cmd
    env = patched_capturing["captured"]["kwargs"]["env"]
    assert env["CHUNKHOUND_NO_PROMPTS"] == "1"
    assert env["CHUNKHOUND_QUICKRESEARCH_QUIET"] == "1"


def test_build_quickresearch_argv_unit_boolean_optional(tmp_path) -> None:
    """Direct call into ``_build_quickresearch_argv`` — no async, no subprocess."""
    from chunkhound.core.config.config import Config

    args = _make_args(llm_ssl_verify=False, ssl_verify=True)
    cmd = ws_mod._build_quickresearch_argv(args, tmp_path, Config())
    assert "--no-llm-ssl-verify" in cmd
    assert "--ssl-verify" in cmd
    assert "--llm-ssl-verify" not in cmd
    assert "True" not in cmd and "False" not in cmd
