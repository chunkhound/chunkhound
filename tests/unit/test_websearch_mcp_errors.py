"""Error-path tests for the websearch MCP tool.

Directly invokes ``websearch_impl`` in-process, stubbing the lazy-imported
CLI helpers and ``asyncio.create_subprocess_exec`` so the error matrix
(URLError, empty results, partial-fetch warnings, subprocess non-zero exit,
timeout, cancellation cleanup) can be asserted without hitting the network
or spawning real subprocesses.
"""

from __future__ import annotations

import asyncio
import tempfile
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from chunkhound.mcp_server import tools as tools_mod
from chunkhound.mcp_server.common import MCPError
from chunkhound.utils import websearch_core as ws_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process."""

    def __init__(
        self,
        stdout: bytes = b"ANSWER",
        stderr: bytes = b"",
        returncode: int | None = 0,
        hang: bool = False,
    ) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self._hang = hang
        # returncode=None means "still running" for the finally kill-check.
        self.returncode: int | None = None if hang else returncode
        self._final_returncode = returncode
        self.kill = MagicMock(side_effect=self._on_kill)

    def _on_kill(self) -> None:
        self.returncode = -9

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            # Stays alive until cancelled or the task is torn down.
            await asyncio.sleep(60)
        self.returncode = self._final_returncode
        return self._stdout, self._stderr

    async def wait(self) -> int:
        self.returncode = self.returncode if self.returncode is not None else -9
        return self.returncode


def _make_fake_exec(proc: _FakeProc):
    proc.exec_called = asyncio.Event()

    async def fake_exec(*args, **kwargs):
        proc.exec_called.set()
        return proc

    return fake_exec


def _stub_search(results):
    def _inner(query, limit=30, progress_callback=None):
        return list(results)

    return _inner


async def _stub_fetch_and_save_noop(
    urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
):
    return None


def _stub_build_argv(query, tmpdir, config, parent_pid):
    # Argv content is irrelevant — subprocess is patched out.
    return ["/bin/true"]


def _default_results():
    return [
        ("Title A", "https://a.invalid/", "desc a"),
        ("Title B", "https://b.invalid/", "desc b"),
        ("Title C", "https://c.invalid/", "desc c"),
    ]


@pytest.fixture
def patched(monkeypatch):
    """Common patches: a safe argv stub and captured mkdtemp paths."""
    captured: dict[str, list] = {"tmpdirs": []}
    real_mkdtemp = tempfile.mkdtemp

    def capture_mkdtemp(*args, **kwargs):
        p = real_mkdtemp(*args, **kwargs)
        captured["tmpdirs"].append(p)
        return p

    monkeypatch.setattr(ws_mod, "build_quickresearch_argv_core", _stub_build_argv)
    monkeypatch.setattr(
        "chunkhound.mcp_server.tools.tempfile.mkdtemp", capture_mkdtemp
    )
    return captured


# ---------------------------------------------------------------------------
# §6 error matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_urlerror_from_search_raises_mcperror(monkeypatch, patched):
    def _raise(*args, **kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(ws_mod, "search", _raise)

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="q",
        )

    assert "Web search failed" in str(exc.value)
    assert "boom" in str(exc.value)


@pytest.mark.asyncio
async def test_empty_results_raises_mcperror(monkeypatch, patched):
    monkeypatch.setattr(ws_mod, "search", _stub_search([]))

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="zero-hits",
        )

    assert "No results found" in str(exc.value)
    assert "zero-hits" in str(exc.value)


@pytest.mark.asyncio
async def test_partial_fetch_warnings_render_bullets(monkeypatch, patched):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))

    async def fetch_with_warnings(
        urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
    ):
        assert warning_callback is not None
        warning_callback("Failed to fetch https://a.invalid/: TimeoutError: x")
        warning_callback("Failed to fetch https://b.invalid/: ValueError: y")

    monkeypatch.setattr(ws_mod, "fetch_and_save", fetch_with_warnings)

    fake_proc = _FakeProc(stdout=b"ANSWER", returncode=0)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    result = await tools_mod.websearch_impl(
        embedding_manager=None,
        llm_manager=None,
        config=None,
        query="partial",
    )

    assert "> **Fetch warnings:**" in result
    assert result.count("\n> - ") == 2
    assert "ANSWER" in result


@pytest.mark.asyncio
async def test_multiline_warning_keeps_blockquote(monkeypatch, patched):
    """A multi-line warning must have ``> `` prefixed to every line."""
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))

    multiline = (
        "Chrome verification failed; falling back to urllib:\n"
        "  - /usr/bin/google-chrome is v100; need >=124\n"
        "(Upgrade Google Chrome to >=124 to enable rich page fetches.)"
    )

    async def fetch_with_multiline_warning(
        urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
    ):
        assert warning_callback is not None
        warning_callback(multiline)

    monkeypatch.setattr(ws_mod, "fetch_and_save", fetch_with_multiline_warning)

    fake_proc = _FakeProc(stdout=b"ANSWER", returncode=0)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    result = await tools_mod.websearch_impl(
        embedding_manager=None,
        llm_manager=None,
        config=None,
        query="multiline",
    )

    # First line gets the bullet; subsequent lines get a bare blockquote prefix.
    assert "> - Chrome verification failed; falling back to urllib:" in result
    assert "\n>   - /usr/bin/google-chrome is v100; need >=124" in result
    assert "\n> (Upgrade Google Chrome to >=124" in result
    # No continuation line escapes the blockquote.
    assert "\n  - /usr/bin/google-chrome" not in result
    assert "\n(Upgrade Google Chrome" not in result


@pytest.mark.asyncio
async def test_subprocess_nonzero_exit_raises_mcperror(monkeypatch, patched):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(
        stdout=b"",
        stderr=b"bad config traceback line-N",
        returncode=2,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="bad",
        )

    msg = str(exc.value)
    assert "Research subprocess failed (exit 2)" in msg
    assert "bad config traceback line-N" in msg


@pytest.mark.asyncio
async def test_subprocess_nonzero_exit_strips_traceback_frames(
    monkeypatch, patched
):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(
        stdout=b"",
        stderr=(
            b"Traceback (most recent call last):\n"
            b"  File '/tmp/x.py', line 1, in <module>\n"
            b"    raise RuntimeError('boom')\n"
            b"RuntimeError: boom"
        ),
        returncode=2,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="bad-traceback",
        )

    msg = str(exc.value)
    assert "Research subprocess failed (exit 2)" in msg
    assert "RuntimeError: boom" in msg
    assert "Traceback" not in msg
    assert "  File '/tmp/x.py'" not in msg
    assert "raise RuntimeError('boom')" not in msg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stderr_bytes, expected_substrings, unexpected_substrings",
    [
        pytest.param(
            b"Traceback (most recent call last):\n  File '/x.py', line 1\n    raise SystemExit(1)\n",
            ["SystemExit(1)"],
            ["Traceback", "File '/x.py'", "raise SystemExit(1)"],
            id="all-traceback-no-error-line",
        ),
        pytest.param(
            b"Command not found: foobar\n",
            ["Command not found"],
            [],
            id="plain-error-no-traceback",
        ),
        pytest.param(
            b"",
            ["exit 2"],
            [],
            id="empty-stderr",
        ),
    ],
)
async def test_subprocess_nonzero_exit_strips_traceback_variants(
    monkeypatch, patched, stderr_bytes, expected_substrings, unexpected_substrings
):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(
        stdout=b"",
        stderr=stderr_bytes,
        returncode=2,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="edge-case",
        )

    msg = str(exc.value)
    for expected in expected_substrings:
        assert expected in msg, f"Expected {expected!r} in {msg!r}"
    for unexpected in unexpected_substrings:
        assert unexpected not in msg, f"Expected {unexpected!r} NOT in {msg!r}"


@pytest.mark.asyncio
async def test_subprocess_stderr_truncates_long_output(monkeypatch, patched):
    """Subprocess stderr >500 chars truncates in the MCPError message."""
    long_stderr = b"xYz info line\n" * 60  # ~1020 chars

    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(
        stdout=b"",
        stderr=long_stderr,
        returncode=2,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="long-stderr",
        )

    msg = str(exc.value)
    assert len(msg) <= 600  # "Research subprocess failed (exit 2): " prefix + truncated content
    # Verify content preserved (not blank)
    assert "xYz" in msg


@pytest.mark.asyncio
async def test_timeout_raises_mcperror_and_cleans_up(monkeypatch, patched):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)
    monkeypatch.setattr(ws_mod, "websearch_timeout", lambda: 0.05)

    fake_proc = _FakeProc(hang=True)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    with pytest.raises(MCPError) as exc:
        await tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="slow",
        )

    assert "timed out" in str(exc.value)
    # finally block must have killed the hung process and wiped the tempdir.
    assert fake_proc.kill.called
    assert patched["tmpdirs"], "mkdtemp was not captured"
    for p in patched["tmpdirs"]:
        assert not Path(p).exists(), f"tempdir {p} should be removed"


@pytest.mark.asyncio
async def test_cancellation_kills_subprocess_and_cleans_tempdir(
    monkeypatch, patched
):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(hang=True)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    task = asyncio.create_task(
        tools_mod.websearch_impl(
            embedding_manager=None,
            llm_manager=None,
            config=None,
            query="cancel-me",
        )
    )

    # Wait until websearch_impl has assigned `proc` and is about to enter
    # communicate(); cancelling earlier would skip the finally-block kill.
    await asyncio.wait_for(fake_proc.exec_called.wait(), timeout=2.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_proc.kill.called
    assert patched["tmpdirs"], "mkdtemp was not captured"
    for p in patched["tmpdirs"]:
        assert not Path(p).exists(), f"tempdir {p} should be removed"


# ---------------------------------------------------------------------------
# parse_mcp_arguments: MCP passes all values as strings
# ---------------------------------------------------------------------------


def test_parse_mcp_arguments_accepts_string_typed_limit():
    """String-typed limit must be tolerated (MCP passes all values as strings)."""
    from chunkhound.mcp_server.common import parse_mcp_arguments

    parsed = parse_mcp_arguments({"limit": "0", "query": "q"})
    assert parsed["limit"] == 0

    parsed = parse_mcp_arguments({"limit": "999", "query": "q"})
    assert parsed["limit"] == 999


@pytest.mark.asyncio
async def test_answer_rewrites_filenames_to_source_urls(monkeypatch, patched):
    monkeypatch.setattr(ws_mod, "search", _stub_search(_default_results()))

    async def populate_mapping(
        urls, tmpdir, progress_callback=None, warning_callback=None, mapping=None
    ):
        assert mapping is not None
        mapping["a.invalid_.md"] = "https://a.invalid/"
        mapping["b.invalid_.pdf"] = "https://b.invalid/"

    monkeypatch.setattr(ws_mod, "fetch_and_save", populate_mapping)

    fake_proc = _FakeProc(
        stdout=b"see a.invalid_.md and b.invalid_.pdf for details",
        returncode=0,
    )
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    result = await tools_mod.websearch_impl(
        embedding_manager=None,
        llm_manager=None,
        config=None,
        query="rewrite",
    )

    assert "https://a.invalid/" in result
    assert "https://b.invalid/" in result
    assert "a.invalid_.md" not in result
    assert "b.invalid_.pdf" not in result


@pytest.mark.asyncio
async def test_limit_clamped_to_range(monkeypatch, patched):
    seen: list[int] = []

    def capturing_search(query, limit=30, progress_callback=None):
        seen.append(limit)
        return _default_results()

    monkeypatch.setattr(ws_mod, "search", capturing_search)
    monkeypatch.setattr(ws_mod, "fetch_and_save", _stub_fetch_and_save_noop)

    fake_proc = _FakeProc(stdout=b"ANSWER", returncode=0)
    monkeypatch.setattr(
        "asyncio.create_subprocess_exec", _make_fake_exec(fake_proc)
    )

    await tools_mod.websearch_impl(
        embedding_manager=None,
        llm_manager=None,
        config=None,
        query="q",
        limit=0,
    )
    await tools_mod.websearch_impl(
        embedding_manager=None,
        llm_manager=None,
        config=None,
        query="q",
        limit=999,
    )

    assert seen == [1, 100]
