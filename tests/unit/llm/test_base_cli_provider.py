"""Tests for BaseCLIProvider double-wrap guard and CLI binary resolution."""

import subprocess
import sys
from pathlib import Path

import pytest

from chunkhound.interfaces.llm_provider import (
    PROVIDER_MANAGED_OUTPUT,
    OutputLimitCapability,
    OutputLimitMetadata,
)
from chunkhound.providers.llm.base_cli_provider import (
    BaseCLIProvider,
    build_cli_argv,
    resolve_cli_binary,
)


class _CapturingCLIProvider(BaseCLIProvider):
    def __init__(self, metadata: OutputLimitMetadata = OutputLimitMetadata()):
        super().__init__()
        self._metadata = metadata
        self.received_limits: list[int | None] = []

    @property
    def output_limit_metadata(self) -> OutputLimitMetadata:
        return self._metadata

    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        self.received_limits.append(max_completion_tokens)
        return '{"ok": true}'

    def _get_provider_name(self) -> str:
        return "capturing-stub"


class _StubCLIProvider(BaseCLIProvider):
    async def _run_cli_command(
        self, prompt: str, system=None, max_completion_tokens=None, timeout=None
    ) -> str:
        return ""  # empty → triggers RuntimeError in complete()

    def _get_provider_name(self) -> str:
        return "stub"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected_limit"),
    [
        (OutputLimitMetadata(), 8192),
        (
            OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED),
            None,
        ),
        (
            OutputLimitMetadata(
                declared_max_tokens=32768,
                declared_max_source="provider declaration",
            ),
            32768,
        ),
    ],
)
async def test_provider_managed_output_is_normalized_before_cli_boundary(
    metadata: OutputLimitMetadata, expected_limit: int | None
) -> None:
    provider = _CapturingCLIProvider(metadata)
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    response = await provider.complete(
        "test", max_completion_tokens=PROVIDER_MANAGED_OUTPUT
    )

    assert provider.received_limits == [expected_limit]
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_explicit_and_default_output_limits_remain_numeric() -> None:
    provider = _CapturingCLIProvider(
        OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED)
    )
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    await provider.complete("default")
    await provider.complete("explicit", max_completion_tokens=123)

    assert provider.received_limits == [4096, 123]


@pytest.mark.asyncio
async def test_structured_provider_managed_output_is_normalized() -> None:
    provider = _CapturingCLIProvider(
        OutputLimitMetadata(omission=OutputLimitCapability.SUPPORTED)
    )
    provider.configure_synthesis_output_limit_policy(
        output_limits_enabled=False, fallback_tokens=8192
    )

    result = await provider.complete_structured(
        "test",
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        max_completion_tokens=PROVIDER_MANAGED_OUTPUT,
    )

    assert result == {"ok": True}
    assert provider.received_limits == [None]


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete():
    """Pass through an empty-response RuntimeError from complete unwrapped."""
    provider = _StubCLIProvider()

    with pytest.raises(RuntimeError) as exc:
        await provider.complete("test")

    msg = str(exc.value)
    assert "LLM returned empty response" in msg
    assert "LLM completion failed" not in msg


@pytest.mark.asyncio
async def test_internal_runtime_error_not_double_wrapped_complete_structured():
    """Pass through an empty-response RuntimeError from structured completion."""
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
    binary = r"C:\Users\me\AppData\Roaming\npm\claude.cmd"
    argv = build_cli_argv(binary, "--print")
    assert argv[0] == "cmd.exe"
    assert argv[1:5] == ["/d", "/v:off", "/s", "/c"]
    assert argv[5:] == [binary, "--print"]


def test_build_cli_argv_wraps_bat_on_windows(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")
    binary = r"D:\tools\tool.bat"
    argv = build_cli_argv(binary, "run")
    assert argv[0] == r"C:\Windows\System32\cmd.exe"
    assert argv[1:5] == ["/d", "/v:off", "/s", "/c"]
    assert argv[5:] == [binary, "run"]


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
    assert argv[5:] == [r"C:\npm\claude.cmd", "--append-system-prompt", dangerous]
    rendered = subprocess.list2cmdline(argv)
    assert '"hello & calc.exe"' in rendered
    assert " --append-system-prompt hello & " not in f" {rendered} "


def test_build_cli_argv_rejects_percent_quote_and_newline(monkeypatch):
    """cmd and a .cmd shim rewrite %, quotes, and newlines."""
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    binary = r"C:\npm\claude.cmd"
    with pytest.raises(RuntimeError, match="%"):
        build_cli_argv(binary, "--x", "%PATH%")
    with pytest.raises(RuntimeError, match="quote"):
        build_cli_argv(binary, "--x", 'say "hi"')
    with pytest.raises(RuntimeError, match="newline"):
        build_cli_argv(binary, "--x", "line1\nline2")


def test_build_cli_argv_rejects_unshortened_spaced_batch_path(monkeypatch):
    """cmd /s splits a quoted long path, so an unshortened path must not launch."""
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    path = r"C:\Program Files\npm\claude.cmd"
    with pytest.raises(RuntimeError, match="8.3"):
        build_cli_argv(path, "--print")


def test_build_cli_argv_exe_keeps_freeform_text(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    argv = build_cli_argv(
        r"C:\tools\claude.exe",
        'say "hi"',
        "%PATH%",
        "a\nb",
    )
    assert argv == [r"C:\tools\claude.exe", 'say "hi"', "%PATH%", "a\nb"]


def test_build_cli_argv_rejects_spaceless_cmd_operator(monkeypatch):
    """A spaceless operator is rejected. A trailing space would change the value."""
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    with pytest.raises(RuntimeError, match="command operator"):
        build_cli_argv(r"C:\npm\claude.cmd", "--x", "a&whoami")


def test_build_cli_argv_8dot3_short_path_stays_its_own_unquoted_token(
    monkeypatch,
):
    """``cmd /s`` strips the first and last quote when /c starts with one.

    An 8.3 path like ``...\\USER~1\\...`` must stay an unquoted token.
    """
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    monkeypatch.delenv("COMSPEC", raising=False)
    short = r"C:\Users\USER~1\AppData\Roaming\npm\claude.cmd"
    argv = build_cli_argv(short, "--print", "--model", "haiku")
    assert argv[1:5] == ["/d", "/v:off", "/s", "/c"]
    assert argv[5:] == [short, "--print", "--model", "haiku"]
    rendered = subprocess.list2cmdline(argv)
    assert f"/c {short} --print" in rendered


def test_build_cli_argv_no_wrap_for_exe(monkeypatch):
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.sys.platform", "win32"
    )
    argv = build_cli_argv(r"C:\tools\claude.exe", "--print")
    assert argv == [r"C:\tools\claude.exe", "--print"]


def test_resolve_cli_binary_env_missing_does_not_fall_back(
    monkeypatch, tmp_path: Path
):
    """A set override that does not exist must not run the PATH binary."""
    fake = tmp_path / "claude.cmd"
    fake.write_text("@echo off\n", encoding="utf-8")
    missing = str(tmp_path / "missing.exe")
    monkeypatch.setenv("CHUNKHOUND_TEST_BIN", missing)
    monkeypatch.setattr(
        "chunkhound.providers.llm.base_cli_provider.shutil.which",
        lambda name: str(fake) if name == "claude" else None,
    )
    with pytest.raises(FileNotFoundError, match="CHUNKHOUND_TEST_BIN"):
        resolve_cli_binary("claude", env_var="CHUNKHOUND_TEST_BIN")


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
