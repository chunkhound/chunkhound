"""Tests for BaseCLIProvider double-wrap guard and CLI binary resolution."""

import re
import sys
from pathlib import Path

import pytest

from chunkhound.providers.llm.base_cli_provider import (
    BaseCLIProvider,
    build_cli_argv,
    escape_cmd_argument,
    resolve_cli_binary,
)


class _StubCLIProvider(BaseCLIProvider):
    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        return ""  # empty → triggers RuntimeError in complete()

    def _get_provider_name(self) -> str:
        return "stub"


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete():
    """RuntimeError from empty-response check must pass through unwrapped in complete()."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("test")

    msg = str(exc.value)
    assert "LLM returned empty response" in msg
    assert "LLM completion failed" not in msg


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """RuntimeError from empty-response check must pass through unwrapped in complete_structured()."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete_structured("test", json_schema={"type": "object"})

    msg = str(exc.value)
    assert "LLM structured completion returned empty response" in msg
    assert "LLM structured completion failed" not in msg


def test_resolve_cli_binary_uses_which(monkeypatch, tmp_path: Path):
    """shutil.which result is returned (simulates PATHEXT finding .cmd)."""
    fake = tmp_path / "claude.cmd"
    fake.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: str(fake) if name == "claude" else None,
    )
    assert resolve_cli_binary("claude") == str(fake)


def test_resolve_cli_binary_prefers_env_path(monkeypatch, tmp_path: Path):
    """Env override path wins when the file exists."""
    fake = tmp_path / "my-claude.exe"
    fake.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CHUNKHOUND_TEST_BIN", str(fake))
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: pytest.fail("which should not run when env path exists"),
    )
    assert resolve_cli_binary("claude", env_var="CHUNKHOUND_TEST_BIN") == str(
        fake.resolve()
    )


def test_resolve_cli_binary_missing_raises(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: None,
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_cli_binary("definitely-missing-cli-xyz")


def test_build_cli_argv_wraps_cmd_on_windows(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    argv = build_cli_argv(r"C:\Users\me\AppData\Roaming\npm\claude.cmd", "--print")
    assert argv[0] == "cmd.exe"
    assert argv[1:4] == ["/d", "/s", "/c"]
    # Single escaped command string (not multi-arg after /c)
    assert len(argv) == 5
    assert r"C:\Users\me\AppData\Roaming\npm\claude.cmd" in argv[4]
    assert "--print" in argv[4]


def test_build_cli_argv_wraps_bat_on_windows(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    argv = build_cli_argv(r"D:\tools\tool.bat", "run")
    assert argv[0] == r"C:\Windows\System32\cmd.exe"
    assert argv[1:4] == ["/d", "/s", "/c"]
    assert "tool.bat" in argv[4]


def test_build_cli_argv_quotes_metacharacters_for_cmd(monkeypatch):
    """System-prompt-like args must not inject shell commands via &."""
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    dangerous = "hello & calc.exe"
    argv = build_cli_argv(
        r"C:\npm\claude.cmd",
        "--append-system-prompt",
        dangerous,
    )
    cmdline = argv[4]
    # Quoted as a single cmd token (escape_cmd_argument contract).
    assert escape_cmd_argument(dangerous) in cmdline
    assert re.search(r'"[^"]*&[^"]*"', cmdline)
    # Without quoting, & would split commands — ensure free unquoted form absent
    assert " --append-system-prompt hello & " not in f" {cmdline} "


def test_build_cli_argv_doubles_percent_for_cmd(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    argv = build_cli_argv(r"C:\npm\claude.cmd", "--x", "%PATH%")
    # % becomes %% so cmd does not expand env vars from untrusted argv
    assert "%%PATH%%" in argv[4]


def test_build_cli_argv_quotes_spaces_in_binary_path(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    path = r"C:\Program Files\npm\claude.cmd"
    argv = build_cli_argv(path, "--print")
    assert f'"{path}"' in argv[4]


def test_escape_cmd_argument_does_not_quote_8dot3_short_paths():
    """Windows short paths use ``~``; quoting them breaks ``cmd /s /c``."""
    short = r"C:\Users\USER~1\AppData\Roaming\npm\claude.cmd"
    assert escape_cmd_argument(short) == short
    assert not escape_cmd_argument(short).startswith('"')


def test_build_cli_argv_8dot3_short_path_cmdline_does_not_start_with_quote(
    monkeypatch,
):
    """``cmd /s`` strips first+last quote when the /c string starts with ``"``.

    An 8.3 path like ``...\\USER~1\\...`` must stay unquoted so /s does not
    mangle the line into ``...claude.cmd" --print ...``.
    """
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    short = r"C:\Users\USER~1\AppData\Roaming\npm\claude.cmd"
    argv = build_cli_argv(short, "--print", "--model", "haiku")
    assert argv[1:4] == ["/d", "/s", "/c"]
    cmdline = argv[4]
    assert not cmdline.startswith('"'), cmdline
    assert cmdline.startswith(short)
    assert "--print" in cmdline
    assert "--model haiku" in cmdline


def test_build_cli_argv_no_wrap_for_exe(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    argv = build_cli_argv(r"C:\tools\claude.exe", "--print")
    assert argv == [r"C:\tools\claude.exe", "--print"]


def test_resolve_cli_binary_env_missing_falls_back_to_which(monkeypatch, tmp_path: Path):
    fake = tmp_path / "claude.cmd"
    fake.write_text("@echo off\n", encoding="utf-8")
    monkeypatch.setenv("CHUNKHOUND_TEST_BIN", str(tmp_path / "missing.exe"))
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: str(fake) if name == "claude" else None,
    )
    # Missing env path is not is_file; which("missing...") fails; then which(name)
    # For name=claude after env cand fails via which of full path...
    # env cand which may return None; then name "claude" via which works.
    assert resolve_cli_binary("claude", env_var="CHUNKHOUND_TEST_BIN") == str(fake)


def test_resolve_cli_binary_ignores_cwd_file_named_like_binary(
    monkeypatch, tmp_path: Path
):
    """Bare name must not pick a same-named file only because CWD contains it."""
    decoy = tmp_path / "claude"
    decoy.write_text("not a real binary", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: None,
    )
    with pytest.raises(FileNotFoundError):
        resolve_cli_binary("claude")


@pytest.mark.skipif(sys.platform == "win32", reason="posix path shape")
def test_build_cli_argv_posix_passthrough():
    argv = build_cli_argv("/usr/local/bin/claude", "--print")
    assert argv == ["/usr/local/bin/claude", "--print"]
