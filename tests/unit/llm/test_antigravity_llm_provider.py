import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

# Mock google-antigravity SDK modules if not installed in the current environment
try:
    import google.antigravity
except ImportError:
    # Preserve a real `google` namespace if one already exists (e.g. when
    # google-genai is installed but google-antigravity is not); only create a
    # mock `google` module when none is present. This avoids clobbering an
    # unrelated `google` package for the duration of the test session.
    google_mock = sys.modules.get("google")
    if google_mock is None:
        import types

        google_mock = types.ModuleType("google")
        sys.modules["google"] = google_mock
    google_antigravity_mock = MagicMock()
    google_antigravity_hooks_mock = MagicMock()
    google_antigravity_types_mock = MagicMock()

    class MockBuiltinTools:
        # Must expose every tool the provider references in _build_agent_config()
        # so SDK tests stay hermetic when google-antigravity is not installed.
        LIST_DIR = "LIST_DIR"
        SEARCH_DIR = "SEARCH_DIR"
        FIND_FILE = "FIND_FILE"
        VIEW_FILE = "VIEW_FILE"
        FINISH = "FINISH"
        CREATE_FILE = "CREATE_FILE"

    class _FakeCapabilitiesConfig:
        # Real fake (not a bare MagicMock) so tests can assert on the kwargs the
        # provider passes; a MagicMock would expose auto-generated child mocks
        # instead of the actual values.
        def __init__(self, *, enabled_tools=None, enable_subagents=False, **kwargs):
            self.enabled_tools = (
                list(enabled_tools) if enabled_tools is not None else []
            )
            self.enable_subagents = enable_subagents
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _FakeLocalAgentConfig:
        # Mirrors the real SDK config: stores kwargs as attributes and, like the
        # real LocalAgentConfig, exposes response_schema as a string.
        def __init__(self, **kwargs):
            self.system_instructions = kwargs.get("system_instructions")
            for key, value in kwargs.items():
                setattr(self, key, value)
            response_schema = kwargs.get("response_schema")
            if response_schema is not None and not isinstance(response_schema, str):
                self.response_schema = str(response_schema)

    google_antigravity_types_mock.BuiltinTools = MockBuiltinTools
    google_antigravity_mock.CapabilitiesConfig = _FakeCapabilitiesConfig
    google_antigravity_mock.LocalAgentConfig = _FakeLocalAgentConfig
    google_mock.antigravity = google_antigravity_mock
    sys.modules["google.antigravity"] = google_antigravity_mock
    sys.modules["google.antigravity.hooks"] = google_antigravity_hooks_mock
    sys.modules["google.antigravity.types"] = google_antigravity_types_mock

import pytest

from chunkhound.interfaces.llm_provider import LLMResponse
from chunkhound.providers.llm.antigravity_cli_provider import AntigravityCLIProvider
from chunkhound.providers.llm.antigravity_llm_provider import (
    SDK_AVAILABLE,
    AntigravityLLMProvider,
)

# Guard against pytest collection order: if an earlier-collected test imported
# `chunkhound.llm_manager` (which imports this provider at module load) before this
# file injected the fake SDK above, the provider module was cached with
# SDK_AVAILABLE=False. When we are running against the injected fake SDK, force the
# module flag/Agent binding so constructor guards and SDK paths exercise the fake
# regardless of collection order. No-op when the real SDK is installed.
if not SDK_AVAILABLE:
    import chunkhound.providers.llm.antigravity_llm_provider as _antigravity_module

    _antigravity_module.SDK_AVAILABLE = True
    _antigravity_module.Agent = sys.modules["google.antigravity"].Agent
    SDK_AVAILABLE = True


@pytest.fixture
def mock_antigravity_agent():
    """Mock the Agent class inside the provider module."""
    with patch(
        "chunkhound.providers.llm.antigravity_llm_provider.Agent", create=True
    ) as mock:
        yield mock


@pytest.fixture
def mock_subprocess():
    """Mock subprocess exec calls for CLI provider.

    Also stubs ``shutil.which`` so binary lookup succeeds without a real
    ``agy``/``antigravity`` on PATH — keeping CLI tests hermetic. Tests that
    need custom lookup behavior (e.g. binary fallback) nest their own
    ``patch("shutil.which")``, which overrides this default.

    Additionally neutralizes the process-tree cleanup that ``_run_cli_command``
    always performs in its ``finally`` block: ``os.killpg`` (Unix) and
    ``subprocess.run`` (Windows ``taskkill``) are patched so a mocked process
    can never send a real OS signal — a mock ``pid`` would otherwise reach
    ``os.killpg`` as a non-int and be coerced to ``1``, signalling process
    group 1 (the init group) on a root CI runner. The real cleanup contract is
    covered by ``test_cli_timeout_cleanup`` and the orphan-cleanup integration
    test. Tests that assert on cleanup nest their own patches, which win inside
    their ``with`` block.
    """
    def _fake_which(name, *_args, **_kwargs):
        # Resolve ``taskkill`` to itself so the Windows process-tree cleanup
        # builds ``["taskkill", ...]``; stub the CLI binary lookup otherwise.
        if name == "taskkill":
            return "taskkill"
        return "/usr/local/bin/agy"

    with (
        patch("asyncio.create_subprocess_exec") as mock,
        patch("shutil.which", side_effect=_fake_which),
        patch("os.killpg", create=True),
        patch("subprocess.run"),
    ):
        yield mock


def _make_agent_mock(mock_agent_class: MagicMock) -> MagicMock:
    """Return mock agent instance entered via async context manager."""
    return mock_agent_class.return_value.__aenter__.return_value


def _make_sdk_response(
    text: str,
    thoughts: str = "",
    prompt_tokens: int = 10,
    candidates_tokens: int = 20,
    thoughts_tokens: int = 5,
) -> tuple[MagicMock, MagicMock]:
    """Helper to create nested Agent responses with usage metadata."""
    response = MagicMock()
    # Mock text() as async function
    response.text = AsyncMock(return_value=text)
    # Mock thoughts property
    response.thoughts = thoughts

    # Mock structured_output as async function
    response.structured_output = AsyncMock(return_value={"key": "val"})

    # Nest usage metadata
    total_usage = MagicMock()
    total_usage.total_token_count = prompt_tokens + candidates_tokens + thoughts_tokens
    total_usage.prompt_token_count = prompt_tokens
    total_usage.candidates_token_count = candidates_tokens
    total_usage.thoughts_token_count = thoughts_tokens

    conversation = MagicMock()
    conversation.total_usage = total_usage

    return response, conversation


# --- SDK Provider Tests ---


@pytest.mark.asyncio
async def test_sdk_missing_api_key_raises(mock_antigravity_agent):
    """No api_key must fail fast at construction so MCP filters out LLM tools
    instead of advertising them. Auth is the resolved CHUNKHOUND_LLM_API_KEY,
    matching CLI validation."""
    with pytest.raises(ValueError, match="API key"):
        AntigravityLLMProvider(api_key=None, target_dir="/tmp/chunkhound-test")


@pytest.mark.asyncio
async def test_sdk_complete_success(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello from Antigravity!", thoughts="Let me think..."
    )

    # Mock complete/chat
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    result = await provider.complete("Test prompt", system="Test system")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from Antigravity!"
    assert result.tokens_used == 35  # 10 + 20 + 5
    assert result.model == "gemini-3.5-flash"

    # Verify mock call details
    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")
    assert config_passed.model == "gemini-3.5-flash"
    assert config_passed.system_instructions == "Test system"


@pytest.mark.asyncio
async def test_sdk_complete_error_propagation(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    # Simulate API connection failure
    agent_mock.chat = AsyncMock(side_effect=Exception("API connection refused"))

    with pytest.raises(RuntimeError, match="API connection refused"):
        await provider.complete("hello")


@pytest.mark.asyncio
async def test_sdk_structured_success(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("{}", thoughts="Structured output...")

    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    # Simple JSON schema
    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)

    assert isinstance(result, dict)
    assert result == {"key": "val"}

    # Verify mock call details
    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")

    # Verify pydantic class is compiled and passed as response_schema
    assert isinstance(config_passed.response_schema, str)
    assert "DynamicResponseModel" in config_passed.response_schema


@pytest.mark.asyncio
async def test_sdk_structured_result_not_blocked_by_slow_text(mock_antigravity_agent):
    """A valid structured response with usage data must return promptly even when
    ``res.text()`` hangs: text output is only a fallback and must never be on the
    critical path of a successful structured completion."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    # Valid structured output + populated usage metadata (so the token-estimate
    # fallback that reads text() is skipped).
    resp_mock, conv_mock = _make_sdk_response("{}")

    async def _hang(*_args, **_kwargs):
        await asyncio.sleep(30)
        return "{}"

    resp_mock.text = AsyncMock(side_effect=_hang)
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    # With text() hanging 30s, a small request timeout would trip only if the
    # provider awaited it; the structured result must come back instead.
    result = await provider.complete_structured(
        "Structured prompt", json_schema=schema, timeout=1
    )

    assert result == {"key": "val"}
    resp_mock.text.assert_not_awaited()


@pytest.mark.asyncio
async def test_sdk_complete_token_estimation_includes_thoughts(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello from Antigravity!", thoughts="Thinking deeply..."
    )

    # Simulate failed token usage retrieval by setting conversation to None
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = None

    result = await provider.complete("Test prompt", system="Test system")

    # Estimated tokens: len(prompt) // 4 + len(system) // 4
    # + len(content + thoughts) // 4
    # prompt = "Test prompt" (11 chars) -> 2 tokens
    # system = "Test system" (11 chars) -> 2 tokens
    # content = "Hello from Antigravity!" (23 chars)
    # thoughts = "Thinking deeply..." (18 chars)
    # content + thoughts = 41 chars -> 10 tokens
    # Total estimated = 2 + 2 + 10 = 14 tokens.
    assert result.tokens_used == 14


@pytest.mark.asyncio
async def test_sdk_complete_reports_component_tokens_when_total_missing(
    mock_antigravity_agent,
):
    """When the SDK exposes component token counts but no populated
    total_token_count, the provider must sum the components instead of
    discarding them for a character-length estimate."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello!", prompt_tokens=50, candidates_tokens=100, thoughts_tokens=25
    )
    # Components are present but the total is missing/zero.
    conv_mock.total_usage.total_token_count = 0
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    result = await provider.complete("prompt")

    # 50 (prompt) + 100 (candidates) + 25 (thoughts) = 175, not a len//4 estimate.
    assert result.tokens_used == 175


# --- CLI Provider Tests ---


@pytest.mark.asyncio
async def test_cli_complete_success(mock_subprocess):
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"Hello from CLI!", b"")
    mock_subprocess.return_value = mock_process

    # Set some test env vars to verify scrubbing and redirection
    with patch("shutil.which", return_value="/usr/local/bin/agy"):
        with patch.dict(
            os.environ,
            {
                "CHUNKHOUND_TEST": "1",
                "GOOGLE_APPLICATION_CREDENTIALS": "abc",
                "PATH": "/usr/bin",
                "HOME": "/home/user",
                "USERPROFILE": "C:\\Users\\user",
                "APPDATA": "C:\\Users\\user\\AppData\\Roaming",
                "LOCALAPPDATA": "C:\\Users\\user\\AppData\\Local",
            },
        ):
            result = await provider.complete("CLI prompt")

    assert isinstance(result, LLMResponse)
    assert result.content == "Hello from CLI!"
    assert result.model == "gemini-3.5-flash"
    assert result.tokens_used > 0

    # Assert subprocess call arguments
    mock_subprocess.assert_called_once()
    cmd_args = mock_subprocess.call_args.args
    assert cmd_args[0].endswith("agy") or cmd_args[0].endswith("antigravity")
    assert "--sandbox" in cmd_args
    # --print must NOT be used: it would leak the prompt onto argv and make agy
    # consume the following flag as the prompt (agy reads the prompt from stdin).
    assert "--print" not in cmd_args
    assert "--model" in cmd_args
    assert cmd_args[cmd_args.index("--model") + 1] == "gemini-3.5-flash"

    # Assert subprocess run CWD and env isolation
    call_kwargs = mock_subprocess.call_args.kwargs
    cwd_passed = call_kwargs.get("cwd")
    assert cwd_passed is not None
    assert "chunkhound-antigravity-" in os.path.basename(cwd_passed)

    env_passed = call_kwargs.get("env")
    assert env_passed is not None
    assert "CHUNKHOUND_TEST" not in env_passed
    assert "GOOGLE_APPLICATION_CREDENTIALS" not in env_passed
    assert env_passed.get("PATH") == "/usr/bin"
    assert env_passed.get("HOME") == "/home/user"
    assert env_passed.get("USERPROFILE") == "C:\\Users\\user"
    assert env_passed.get("APPDATA") == "C:\\Users\\user\\AppData\\Roaming"
    assert env_passed.get("LOCALAPPDATA") == "C:\\Users\\user\\AppData\\Local"


@pytest.mark.asyncio
async def test_cli_prompt_sent_via_stdin_not_argv(mock_subprocess):
    """The prompt (and merged system instructions) must be delivered over stdin,
    never as a command-line argument — otherwise workspace source snippets, file
    paths, and secrets leak into process listings (`ps`). agy reads the piped,
    non-TTY stdin as the prompt when no prompt flag is passed."""
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"ok", b"")
    mock_subprocess.return_value = mock_process

    prompt = "SENSITIVE-PROMPT-XYZ"
    system = "SYS-INSTR-ABC"
    with patch("shutil.which", return_value="/usr/local/bin/agy"):
        await provider.complete(prompt, system=system)

    # The prompt must not appear anywhere in argv.
    cmd_args = mock_subprocess.call_args.args
    assert not any(prompt in str(arg) for arg in cmd_args)
    assert not any(system in str(arg) for arg in cmd_args)

    # The merged prompt must be piped via stdin instead.
    stdin_input = mock_process.communicate.call_args.kwargs.get("input")
    assert stdin_input is not None
    decoded = stdin_input.decode("utf-8")
    assert prompt in decoded
    assert system in decoded


@pytest.mark.asyncio
async def test_cli_temp_dir_removed_on_cancellation(mock_subprocess):
    """If the task is cancelled while the process tree is being torn down, the
    per-call temp working directory must still be removed (no leak)."""
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = None  # forces the kill branch in the finally block
    # Simulate the task being cancelled while the subprocess call is pending.
    mock_process.communicate.side_effect = asyncio.CancelledError()
    mock_subprocess.return_value = mock_process

    with (
        patch("shutil.which", return_value="/usr/local/bin/agy"),
        # Reproduce the reported bug shape: cancellation re-raised from the
        # shielded process-tree teardown must not skip temp-dir removal.
        patch.object(
            provider,
            "_kill_process_tree",
            AsyncMock(side_effect=asyncio.CancelledError()),
        ),
    ):
        with pytest.raises(asyncio.CancelledError):
            await provider._run_cli_command("prompt")

    cwd_passed = mock_subprocess.call_args.kwargs.get("cwd")
    assert cwd_passed is not None
    assert "chunkhound-antigravity-" in os.path.basename(cwd_passed)
    assert not os.path.exists(cwd_passed), "temp dir leaked after cancellation"


@pytest.mark.asyncio
async def test_cli_debug_log_redacts_prompt(mock_subprocess):
    """The CLI debug log must never contain prompt text (which can carry source
    snippets, paths, or secrets) — only the binary and flags, with the prompt
    summarized by length."""
    cli = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"ok", b"")
    mock_subprocess.return_value = mock_process

    secret = "super-secret-token-42"
    with (
        patch(
            "chunkhound.providers.llm.antigravity_cli_provider.logger.debug"
        ) as mock_debug,
        patch("shutil.which", return_value="/usr/local/bin/agy"),
    ):
        await cli.complete(secret)

    logged = " ".join(
        str(call.args[0]) for call in mock_debug.call_args_list if call.args
    )
    assert secret not in logged, "prompt content leaked into the CLI debug log"
    assert "--sandbox" in logged
    # Prompt travels via stdin, so it is neither on argv nor in the log; the log
    # only notes its length.
    assert "stdin" in logged


@pytest.mark.asyncio
async def test_cli_complete_failure(mock_subprocess):
    """A nonzero exit surfaces a curated error, never the CLI's raw output.

    The stderr can echo the piped prompt, source snippets, and paths (only
    token-shaped secrets are redacted). It must not reach the user-visible
    RuntimeError or the default-level log; the sanitized detail is available
    only at debug level.
    """
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = 1
    mock_process.communicate.return_value = (
        b"",
        b"Authentication failed: api_key=secret_1234567890",
    )
    mock_subprocess.return_value = mock_process

    with (
        patch(
            "chunkhound.providers.llm.antigravity_cli_provider.logger.debug"
        ) as mock_debug,
        pytest.raises(RuntimeError) as excinfo,
    ):
        await provider.complete("CLI prompt")

    # User-visible error: exit code only, no CLI output / secrets.
    message = str(excinfo.value)
    assert message == "Antigravity CLI command failed (exit code 1)"
    assert "secret_1234567890" not in message
    assert "api_key" not in message

    # The stderr detail is available only at debug, with token-shaped secrets
    # redacted by the shared sanitizer.
    logged = " ".join(
        str(call.args[0]) for call in mock_debug.call_args_list if call.args
    )
    assert "api_key=[REDACTED]" in logged
    assert "secret_1234567890" not in logged


@pytest.mark.asyncio
async def test_cli_scopes_temp_env_to_sandbox_dir(mock_subprocess):
    """Child temp-file APIs are scoped to the per-call sandbox cwd.

    TMPDIR/TMP/TEMP are pointed at the temp cwd so anything the CLI writes via
    tempfile.* lands in the sandbox dir that cleanup removes, rather than the
    inherited system temp location.
    """
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"ok", b"")
    mock_subprocess.return_value = mock_process

    with patch("shutil.which", return_value="/usr/local/bin/agy"):
        await provider.complete("prompt")

    kwargs = mock_subprocess.call_args[1]
    env, cwd = kwargs["env"], kwargs["cwd"]
    assert "chunkhound-antigravity-" in cwd
    assert env["TMPDIR"] == cwd
    assert env["TMP"] == cwd
    assert env["TEMP"] == cwd


@pytest.mark.asyncio
async def test_cli_binary_fallback(mock_subprocess):
    with patch("shutil.which") as mock_which:

        def side_effect(cmd):
            if cmd == "agy":
                return None
            if cmd == "antigravity":
                return "/usr/local/bin/antigravity"
            return None

        mock_which.side_effect = side_effect

        provider = AntigravityCLIProvider(model="gemini-3.5-flash")
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Hello from fallback CLI!", b"")
        mock_subprocess.return_value = mock_process

        result = await provider.complete("CLI prompt")

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from fallback CLI!"
        mock_subprocess.assert_called_once()
        cmd_args = mock_subprocess.call_args.args
        assert cmd_args[0].endswith("antigravity")


@pytest.mark.asyncio
async def test_cli_timeout_cleanup(mock_subprocess):
    import sys
    import signal

    provider = AntigravityCLIProvider(model="gemini-3.5-flash", timeout=2)

    mock_process = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = None
    mock_process.wait = AsyncMock(return_value=0)
    mock_subprocess.return_value = mock_process

    with (
        patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()),
        patch("os.killpg", create=True) as mock_killpg,
        patch("subprocess.run") as mock_run,
    ):
        with pytest.raises(RuntimeError, match="timed out after 2s"):
            await provider.complete("CLI prompt")

    if sys.platform == "win32":
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "taskkill" in args
        assert "12345" in args
    else:
        mock_killpg.assert_any_call(12345, signal.SIGTERM)

    # The per-call temp working directory must be removed even on the timeout path.
    cwd_passed = mock_subprocess.call_args.kwargs.get("cwd")
    assert cwd_passed is not None
    assert not os.path.exists(cwd_passed), "temp dir leaked after timeout"


@pytest.mark.asyncio
async def test_cli_windows_cleanup_never_masks_primary_error():
    """On Windows, best-effort process reaping must never raise: it runs in the
    caller's ``finally``, so an escaping psutil error (e.g. AccessDenied) would
    replace the curated timeout/cancel error. ``_kill_process_tree`` must swallow
    it. Runs on any OS by forcing the win32 branch."""
    import types

    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    process = AsyncMock()
    process.pid = 12345
    process.returncode = None  # still running -> Windows branch does not early-return
    process.wait = AsyncMock(return_value=0)

    class _FakeError(Exception):
        pass

    class _FakeAccessDenied(_FakeError):
        pass

    class _FakeNoSuchProcess(_FakeError):
        pass

    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Error = _FakeError
    fake_psutil.AccessDenied = _FakeAccessDenied
    fake_psutil.NoSuchProcess = _FakeNoSuchProcess

    def _raise_access_denied(_pid):
        raise _FakeAccessDenied("denied")

    fake_psutil.Process = _raise_access_denied

    taskkill_result = MagicMock()
    taskkill_result.returncode = 1  # taskkill fails -> psutil fallback runs
    taskkill_result.stderr = b"access denied"

    with (
        patch(
            "chunkhound.providers.llm.antigravity_cli_provider.sys.platform",
            "win32",
        ),
        patch("shutil.which", return_value="taskkill"),
        patch("subprocess.run", return_value=taskkill_result),
        patch.dict(sys.modules, {"psutil": fake_psutil}),
    ):
        # Must return without raising despite the psutil AccessDenied during
        # the best-effort fallback reap.
        await provider._kill_process_tree(process)


@pytest.mark.asyncio
async def test_sdk_timeout(mock_antigravity_agent):
    """A per-call timeout override must bound ``complete()`` and surface as a
    RuntimeError even when the SDK chat call hangs — and it must honor the
    override, not the larger default request timeout."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        timeout=30,
        target_dir="/tmp/chunkhound-test",
    )

    class _HangingChatAgent:
        def __init__(self, *_args, **_kwargs):
            self.conversation = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def chat(self, _prompt):
            await asyncio.sleep(30)

    mock_antigravity_agent.side_effect = _HangingChatAgent

    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(RuntimeError, match="timed out after"):
        await provider.complete("Test prompt", timeout=0.3)
    elapsed = loop.time() - start

    # Must use the 0.3s override, not the 30s default, and return control.
    assert elapsed < 5, f"complete() did not honor the timeout override ({elapsed}s)"


@pytest.mark.asyncio
async def test_sdk_hung_teardown_returns_control(mock_antigravity_agent):
    """A stuck ``Agent.__aexit__`` that ignores cancellation must not extend the
    request indefinitely: ``complete()`` bounds teardown by CLEANUP_GRACE_TIMEOUT
    and returns control as a timeout error rather than hanging on the unwind."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        timeout=30,
        target_dir="/tmp/chunkhound-test",
    )
    # Shrink the teardown grace so the test is fast.
    provider.CLEANUP_GRACE_TIMEOUT = 0.3

    response, conversation = _make_sdk_response("done")

    class _HungTeardownAgent:
        def __init__(self, *_args, **_kwargs):
            self.conversation = conversation

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            # Resist cancellation for well beyond the grace window, then let go
            # so the abandoned task can eventually finish (no session hang).
            deadline = asyncio.get_running_loop().time() + 3.0
            while asyncio.get_running_loop().time() < deadline:
                try:
                    await asyncio.sleep(3.0)
                except asyncio.CancelledError:
                    continue
            return False

        async def chat(self, _prompt):
            return response

    mock_antigravity_agent.side_effect = _HungTeardownAgent

    loop = asyncio.get_running_loop()
    start = loop.time()
    with pytest.raises(RuntimeError, match="timed out after"):
        await provider.complete("hi", timeout=0.3)
    elapsed = loop.time() - start

    # request_timeout (0.3) + grace (0.3) with headroom; must NOT wait the full
    # 3s teardown before returning control to the caller.
    assert elapsed < 2, f"hung teardown was not abandoned within grace ({elapsed}s)"

    # The still-running teardown must be RETAINED (owned by the provider), not
    # silently orphaned: it stays tracked while it runs past the grace window.
    assert len(provider._abandoned_teardowns) == 1

    # Once the stuck __aexit__ finally releases (~3s), the done-callback drains
    # the tracked task, so the provider does not leak the reference forever.
    drain_deadline = loop.time() + 5.0
    while provider._abandoned_teardowns and loop.time() < drain_deadline:
        await asyncio.sleep(0.1)
    assert not provider._abandoned_teardowns, (
        "abandoned teardown task was never drained after teardown completed"
    )


@pytest.mark.asyncio
async def test_sdk_late_raising_teardown_is_drained(mock_antigravity_agent):
    """A teardown that ignores cancellation past the grace window and then RAISES
    must have its late exception drained (retrieved) — not left to surface as an
    unhandled "exception was never retrieved" — and the real cleanup failure must
    be logged so operators can see it."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        timeout=30,
        target_dir="/tmp/chunkhound-test",
    )
    provider.CLEANUP_GRACE_TIMEOUT = 0.3

    response, conversation = _make_sdk_response("done")

    class _LateRaisingTeardownAgent:
        def __init__(self, *_args, **_kwargs):
            self.conversation = conversation

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            # Resist cancellation past the grace window, then fail the unwind.
            deadline = asyncio.get_running_loop().time() + 0.8
            while asyncio.get_running_loop().time() < deadline:
                try:
                    await asyncio.sleep(0.8)
                except asyncio.CancelledError:
                    continue
            raise RuntimeError("teardown blew up")

        async def chat(self, _prompt):
            return response

    mock_antigravity_agent.side_effect = _LateRaisingTeardownAgent

    loop = asyncio.get_running_loop()
    with patch(
        "chunkhound.providers.llm.antigravity_llm_provider.logger.warning"
    ) as mock_warning:
        with pytest.raises(RuntimeError, match="timed out after"):
            await provider.complete("hi", timeout=0.3)

        # Task is retained while it runs past the grace window.
        assert len(provider._abandoned_teardowns) == 1

        # Once the stuck __aexit__ finally raises, the done-callback must drain
        # the task (removing it) — proving the late exception was retrieved.
        drain_deadline = loop.time() + 5.0
        while provider._abandoned_teardowns and loop.time() < drain_deadline:
            await asyncio.sleep(0.05)
        assert not provider._abandoned_teardowns, (
            "late-raising teardown task was never drained"
        )

    # The real cleanup failure is surfaced (retrieved, not dropped).
    warnings = " ".join(
        str(c.args[0]) for c in mock_warning.call_args_list if c.args
    )
    assert "ultimately failed" in warnings
    assert "teardown blew up" in warnings


@pytest.mark.asyncio
async def test_batch_complete_cancels_siblings_on_failure(mock_antigravity_agent):
    """A failed batch prompt must cancel still-running siblings so their
    ``Agent`` sessions tear down via ``__aexit__`` instead of lingering as
    orphan tasks that keep touching the workspace after the caller has the error.
    """
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_complete(prompt, **_kwargs):
        if prompt == "fail":
            # Yield once so both prompts start before the failure propagates
            # through ``asyncio.gather``.
            await asyncio.sleep(0)
            raise RuntimeError("boom")
        started.set()
        try:
            # Simulate a sibling blocked inside ``res.text()``.
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return LLMResponse(
            content="slow-done",
            tokens_used=0,
            model="gemini-3.5-flash",
            finish_reason="stop",
        )

    provider.complete = fake_complete  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="boom"):
        await provider.batch_complete(["fail", "slow"])

    assert started.is_set(), "slow sibling should have entered its session"
    assert cancelled.is_set(), (
        "slow sibling should be cancelled (session torn down) on batch failure, "
        "not left running as an orphan task"
    )


@pytest.mark.asyncio
async def test_health_check_uses_short_timeout(mock_antigravity_agent):
    """Health checks must use a short readiness timeout (HEALTH_CHECK_TIMEOUT),
    not the full request timeout, so an unhealthy SDK session fails fast."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    assert provider.HEALTH_CHECK_TIMEOUT == 30

    fake_response = LLMResponse(
        content="pong",
        tokens_used=0,
        model="gemini-3.5-flash",
        finish_reason="stop",
    )
    mock_complete = AsyncMock(return_value=fake_response)
    provider.complete = mock_complete  # type: ignore[assignment]

    result = await provider.health_check()

    mock_complete.assert_awaited_once()
    assert mock_complete.call_args.kwargs["timeout"] == provider.HEALTH_CHECK_TIMEOUT
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_sdk_security_constraints(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello!")

    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    # Spy on the SDK config constructor to capture the RAW policy list we hand it,
    # before the SDK normalizes it. `LocalAgentConfig.policies` after construction
    # is SDK-version/platform-specific (the SDK expands our deny_all()+allow() list
    # into synthetic per-tool rules and may rename/relocate the deny baseline), so
    # asserting on it couples the test to SDK mechanics and breaks across platforms.
    import google.antigravity as _ga

    captured_policies = {}
    real_config_cls = _ga.LocalAgentConfig

    def _config_spy(**kwargs):
        captured_policies["raw"] = kwargs.get("policies")
        return real_config_cls(**kwargs)

    with patch.object(_ga, "LocalAgentConfig", _config_spy):
        await provider.complete("Test prompt")

    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")

    assert config_passed.capabilities is not None
    assert not config_passed.capabilities.enable_subagents
    enabled_tools = config_passed.capabilities.enabled_tools
    assert enabled_tools is not None
    from google.antigravity.types import BuiltinTools

    # Read-only capability contract: exactly the five inspection tools are
    # enabled, and no write/exec-capable tool leaks in.
    for readonly in (
        BuiltinTools.LIST_DIR,
        BuiltinTools.SEARCH_DIR,
        BuiltinTools.FIND_FILE,
        BuiltinTools.VIEW_FILE,
        BuiltinTools.FINISH,
    ):
        assert readonly in enabled_tools
    assert BuiltinTools.CREATE_FILE not in enabled_tools
    assert len(enabled_tools) == 5

    assert config_passed.workspaces == ["/tmp/chunkhound-test"]

    # Policy safety contract, asserted on the RAW list we construct (captured
    # above) rather than the SDK-normalized `config.policies`. Structurally the
    # list is a deny-all baseline plus exactly the five read-only allows — a
    # fixed six-entry list we fully own, stable across SDK versions/platforms.
    raw_policies = captured_policies["raw"]
    assert raw_policies is not None
    assert len(raw_policies) == 6  # deny_all() + 5 read-only allow()s

    # Decision/tool semantics are only meaningful with the real SDK installed
    # (under the injected fake SDK the policy helpers return MagicMocks). Where
    # the real SDK is present, verify the deny baseline and that ONLY read-only
    # inspection tools are approved — never a write/exec-capable tool.
    if not isinstance(_ga, MagicMock):
        deny_tools = {p.tool for p in raw_policies if p.decision.name == "DENY"}
        approved_tools = {
            p.tool for p in raw_policies if p.decision.name == "APPROVE"
        }
        assert "*" in deny_tools  # deny-all baseline present
        assert approved_tools == {
            "list_directory",
            "search_directory",
            "find_file",
            "view_file",
            "finish",
        }
        assert "create_file" not in approved_tools


@pytest.mark.asyncio
async def test_sdk_structured_validation(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)

    response = MagicMock()
    response.structured_output = AsyncMock(return_value={"confidence": 2.0})
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["confidence"],
    }

    with pytest.raises(
        RuntimeError,
        match="validation failed|exceeds|maximum|greater than|is greater than",
    ):
        await provider.complete_structured("Structured prompt", json_schema=schema)


def test_cli_synthesis_concurrency():
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")
    assert provider.get_synthesis_concurrency() == 1


@pytest.mark.asyncio
async def test_cli_max_completion_tokens_warning(mock_subprocess):
    provider = AntigravityCLIProvider(model="gemini-3.5-flash")

    mock_process = AsyncMock()
    mock_process.returncode = 0
    mock_process.communicate.return_value = (b"Response from CLI", b"")
    mock_subprocess.return_value = mock_process

    with patch(
        "chunkhound.providers.llm.antigravity_cli_provider.logger.warning"
    ) as mock_warn:
        await provider.complete("prompt", max_completion_tokens=512)
        mock_warn.assert_called_once_with(
            "Antigravity CLI does not support limiting output tokens "
            "via max_completion_tokens. "
            "Requested limit of 512 is ignored."
        )


@pytest.mark.asyncio
async def test_sdk_structured_fallback_validation_failure(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)

    response = MagicMock()
    # Remove structured_output attribute to trigger fallback path
    if hasattr(response, "structured_output"):
        delattr(response, "structured_output")

    # Mock text() returning schema-invalid JSON
    response.text = AsyncMock(return_value='{"confidence": 2.0}')
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
        },
        "required": ["confidence"],
    }

    # Verify that the schema validation error propagates out as a RuntimeError
    with pytest.raises(
        RuntimeError,
        match="validation failed|exceeds|maximum|greater than|is greater than",
    ):
        await provider.complete_structured("Structured prompt", json_schema=schema)


def test_sdk_root_ref_schema_compiles_to_referenced_model():
    """A root-level $ref schema must compile to the referenced model, not an
    empty one, so the SDK receives the correct structured-output shape."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    schema = {
        "$ref": "#/$defs/Result",
        "$defs": {
            "Result": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
    }

    model = provider._compile_schema_to_pydantic(schema)
    assert "name" in model.model_fields


@pytest.mark.asyncio
async def test_sdk_structured_required_null_preserved(mock_antigravity_agent):
    """A required field returned as null must survive serialization and pass
    validation, rather than being dropped and failing the schema's `required`."""
    from pydantic import BaseModel

    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    class _Model(BaseModel):
        a: str | None  # required (no default) but nullable
        b: str

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    response.structured_output = AsyncMock(return_value=_Model(a=None, b="x"))
    response.text = AsyncMock(return_value="")
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "null"]},
            "b": {"type": "string"},
        },
        "required": ["a", "b"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    assert result == {"a": None, "b": "x"}


@pytest.mark.asyncio
async def test_sdk_structured_optional_nullable_null_preserved(mock_antigravity_agent):
    """An OPTIONAL field whose schema permits null must keep an explicit null the
    model returned, not silently drop it. Uses the raw-dict production path (the
    real SDK returns json.loads output)."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    response.structured_output = AsyncMock(return_value={"a": None})
    response.text = AsyncMock(return_value="")
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {"a": {"type": ["string", "null"]}},
        "required": [],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    # Optional + nullable: the explicit null is a valid answer and must survive.
    assert result == {"a": None}


@pytest.mark.asyncio
async def test_sdk_structured_root_allof_nullable_null_preserved(
    mock_antigravity_agent,
):
    """An optional nullable field declared only under a bare root ``allOf``
    branch must keep an explicit null. Earlier the cleanup only ``$ref``-resolved
    the root and dropped it (root-composition cleanup)."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    response.structured_output = AsyncMock(return_value={"a": None})
    response.text = AsyncMock(return_value="")
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "$defs": {
            "Base": {
                "type": "object",
                "properties": {"a": {"type": ["string", "null"]}},
                "required": [],
            }
        },
        "allOf": [{"$ref": "#/$defs/Base"}],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    # Optional + nullable under a bare root allOf: the explicit null must survive.
    assert result == {"a": None}


@pytest.mark.asyncio
async def test_sdk_structured_nested_optional_none_dropped(mock_antigravity_agent):
    """A nested optional field returned as null must be dropped (treated as
    absent) rather than serialized as null and rejected by the validator."""
    from pydantic import BaseModel

    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    class _Inner(BaseModel):
        name: str | None = None  # optional, non-nullable in the JSON schema

    class _Outer(BaseModel):
        profile: _Inner

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    response.structured_output = AsyncMock(
        return_value=_Outer(profile=_Inner(name=None))
    )
    response.text = AsyncMock(return_value="")
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            }
        },
        "required": ["profile"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    # Nested optional None is dropped; the object validates as absent-name.
    assert result == {"profile": {}}


@pytest.mark.asyncio
async def test_sdk_structured_dict_nested_optional_none_dropped(mock_antigravity_agent):
    """Real SDK returns a raw dict (json.loads), not a pydantic model; a nested
    optional field returned as null must still be dropped rather than serialized
    as null and rejected by the validator."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    response.structured_output = AsyncMock(return_value={"profile": {"name": None}})
    response.text = AsyncMock(return_value="")
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            }
        },
        "required": ["profile"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    # Nested optional None is dropped even on the raw-dict path.
    assert result == {"profile": {}}


@pytest.mark.asyncio
async def test_sdk_structured_text_fallback_drops_optional_none(mock_antigravity_agent):
    """When structured_output is unavailable and the model returns JSON text with
    an optional field as null, the text fallback must drop the optional None
    (treated as absent) rather than fail strict schema validation — matching the
    dict/model paths."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    response = MagicMock()
    # Force the text-fallback path: no structured_output attribute at all.
    del response.structured_output
    response.text = AsyncMock(return_value='{"profile": {"name": null}}')
    response.thoughts = ""
    agent_mock.chat = AsyncMock(return_value=response)
    agent_mock.conversation = MagicMock()

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": [],
            }
        },
        "required": ["profile"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)
    # Optional None dropped on the text path too.
    assert result == {"profile": {}}


@pytest.mark.asyncio
async def test_sdk_usage_stats_tracking(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    # Check initial stats
    initial_stats = provider.get_usage_stats()
    assert initial_stats == {
        "requests_made": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "Hello!", prompt_tokens=50, candidates_tokens=100, thoughts_tokens=25
    )
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    # Complete request
    await provider.complete("Prompt")

    # Check updated stats
    updated_stats = provider.get_usage_stats()
    assert updated_stats == {
        "requests_made": 1,
        "total_tokens": 175,
        "prompt_tokens": 50,
        "completion_tokens": 125,  # candidates (100) + thoughts (25)
    }


@pytest.mark.asyncio
async def test_sdk_health_check_healthy(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("pong")
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    health = await provider.health_check()
    assert health == {
        "status": "healthy",
        "provider": "antigravity-sdk",
        "model": "gemini-3.5-flash",
        "test_response": "pong",
    }


@pytest.mark.asyncio
async def test_sdk_health_check_unhealthy(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    agent_mock.chat = AsyncMock(side_effect=Exception("Connection failed"))

    health = await provider.health_check()
    assert health["status"] == "unhealthy"
    assert health["provider"] == "antigravity-sdk"
    assert "Connection failed" in health["error"]


@pytest.mark.asyncio
async def test_sdk_max_completion_tokens_warning(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello!")
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    with patch(
        "chunkhound.providers.llm.antigravity_llm_provider.logger.warning"
    ) as mock_warn:
        await provider.complete("Prompt", max_completion_tokens=100)
        mock_warn.assert_called_once()
        assert "max_completion_tokens" in mock_warn.call_args[0][0]


def test_sdk_compile_schema_to_pydantic_constraints(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )
    schema = {
        "type": "object",
        "properties": {
            "num_val": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 5.0,
                "exclusiveMinimum": 0.5,
                "exclusiveMaximum": 5.5,
            },
            "str_val": {
                "type": "string",
                "minLength": 2,
                "maxLength": 10,
                "pattern": "^[a-z]+$",
            },
            "arr_val": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 1,
                "maxItems": 3,
            },
            "enum_val": {"type": "string", "enum": ["red", "blue"]},
            "enum_arr": {
                "type": "array",
                "items": {"type": "string", "enum": ["apple", "banana"]},
            },
        },
        "required": ["num_val", "enum_val"],
        "additionalProperties": False,
    }

    model_cls = provider._compile_schema_to_pydantic(schema)

    # Assert model constraints by trying to validate valid and invalid inputs
    from pydantic import ValidationError

    # Valid model instance
    valid_instance = model_cls(
        num_val=3.0,
        str_val="abc",
        arr_val=[1, 2],
        enum_val="red",
        enum_arr=["apple", "banana"],
    )
    assert valid_instance.num_val == 3.0
    assert valid_instance.str_val == "abc"
    assert valid_instance.arr_val == [1, 2]
    assert valid_instance.enum_val == "red"
    assert valid_instance.enum_arr == ["apple", "banana"]

    # Invalid numeric bounds (minimum / exclusiveMinimum / maximum / exclusiveMaximum)
    with pytest.raises(ValidationError):
        model_cls(num_val=0.5, enum_val="red")  # fails exclusiveMinimum
    with pytest.raises(ValidationError):
        model_cls(num_val=5.5, enum_val="red")  # fails exclusiveMaximum
    with pytest.raises(ValidationError):
        model_cls(num_val=6.0, enum_val="red")  # fails maximum

    # Invalid string length
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="a", enum_val="red")
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="abcdefghiklmn", enum_val="red")

    # Invalid string pattern
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, str_val="abc1", enum_val="red")

    # Invalid array items size
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, arr_val=[], enum_val="red")
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, arr_val=[1, 2, 3, 4], enum_val="red")

    # Invalid enum value
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="green")

    # Invalid array enum value
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="red", enum_arr=["orange"])

    # Forbidden extra properties (additionalProperties=False)
    with pytest.raises(ValidationError):
        model_cls(num_val=3.0, enum_val="red", extra_field="forbidden")


def test_sdk_compile_schema_allof_merges_intersection(mock_antigravity_agent):
    """allOf is an intersection: subschemas merge into one model requiring all fields."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )
    schema = {
        "$defs": {
            "Base": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        },
        "allOf": [{"$ref": "#/$defs/Base"}],
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    model_cls = provider._compile_schema_to_pydantic(schema)

    from pydantic import ValidationError

    # Merged model accepts all required fields from both subschemas.
    valid = model_cls(id=1, name="chunkhound")
    assert valid.id == 1
    assert valid.name == "chunkhound"

    # Missing the field contributed by the $ref'd subschema must fail.
    with pytest.raises(ValidationError):
        model_cls(name="chunkhound")
    # Missing the field contributed by the parent subschema must fail.
    with pytest.raises(ValidationError):
        model_cls(id=1)


@pytest.mark.parametrize("comp_key", ["anyOf", "oneOf"])
def test_sdk_root_union_schema_fails_closed(mock_antigravity_agent, comp_key):
    """A root anyOf/oneOf schema raises a clear ChunkHound error, not a cryptic
    SDK config-build failure. The SDK's response_schema needs a single model
    class, not a typing.Union, so this is rejected up front."""
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )
    schema = {
        comp_key: [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "integer"}}},
        ]
    }

    with pytest.raises(ValueError, match=f"root '{comp_key}'"):
        provider._compile_schema_to_pydantic(schema)


@pytest.mark.asyncio
async def test_sdk_target_dir_propagation(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/test/project/root",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response("Hello")
    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    await provider.complete("Test prompt")

    mock_antigravity_agent.assert_called_once()
    config_passed = mock_antigravity_agent.call_args[1].get("config")
    assert config_passed.workspaces == ["/test/project/root"]


@pytest.mark.asyncio
async def test_sdk_structured_pydantic_model(mock_antigravity_agent):
    provider = AntigravityLLMProvider(
        api_key="test-api-key",
        model="gemini-3.5-flash",
        target_dir="/tmp/chunkhound-test",
    )

    agent_mock = _make_agent_mock(mock_antigravity_agent)
    resp_mock, conv_mock = _make_sdk_response(
        "{}", thoughts="Structured output Pydantic..."
    )

    # Mock structured_output to return a mock object behaving like Pydantic model
    mock_pydantic_instance = MagicMock()
    mock_pydantic_instance.model_dump = MagicMock(return_value={"key": "val"})
    # Delete dict method to make sure model_dump is called
    if hasattr(mock_pydantic_instance, "dict"):
        delattr(mock_pydantic_instance, "dict")

    resp_mock.structured_output = AsyncMock(return_value=mock_pydantic_instance)

    agent_mock.chat = AsyncMock(return_value=resp_mock)
    agent_mock.conversation = conv_mock

    schema = {
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    result = await provider.complete_structured("Structured prompt", json_schema=schema)

    assert isinstance(result, dict)
    assert result == {"key": "val"}
    mock_pydantic_instance.model_dump.assert_called_once()
