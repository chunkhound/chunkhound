"""Base CLI provider for LLM providers that use command-line interfaces.

This base class contains shared logic for CLI-based providers
(ClaudeCode, Codex, OpenCode) to avoid code duplication and ensure
consistent behavior.
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shutil
import subprocess
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.llm_config import DEFAULT_LLM_TIMEOUT
from chunkhound.core.utils import estimate_tokens_llm
from chunkhound.interfaces.llm_provider import (
    LLMProvider,
    LLMResponse,
    OutputLimitIntent,
)
from chunkhound.utils.json_extraction import parse_and_validate_structured_json

# cmd operators that split a command when the token is not quoted.
# ``~`` is not one of them: 8.3 short paths use it and must stay unquoted.
_CMD_OPERATORS = frozenset("&|<>^()")


def resolve_cli_binary(
    name: str,
    *,
    env_var: str | None = None,
) -> str:
    """Resolve a CLI executable path for ``create_subprocess_exec``.

    Interactive shells on Windows find ``claude.cmd`` / ``codex.cmd`` via
    PATHEXT. ``asyncio.create_subprocess_exec`` / CreateProcess with a bare
    name does not — often only ``.exe`` is tried, causing WinError 2.

    ``shutil.which`` respects PATHEXT and returns the full path to the shim.

    Args:
        name: Default command name (e.g. ``claude``).
        env_var: Optional env var override (absolute path or name on PATH).

    Returns:
        Absolute path to the executable (or override path that exists / which found).

    Raises:
        FileNotFoundError: If no matching binary is found.
    """
    candidates: list[str] = []
    if env_var:
        env_val = os.environ.get(env_var)
        if env_val and env_val.strip():
            candidates.append(env_val.strip())
    candidates.append(name)

    for cand in candidates:
        path = Path(cand)
        # Only treat as an explicit filesystem path when it has a directory
        # component (or is absolute). Bare names always go through which/PATH.
        if (path.is_absolute() or os.path.dirname(cand)) and path.is_file():
            return str(path.resolve())
        found = shutil.which(cand)
        if found:
            return found

    hint = f" (checked env {env_var} and PATH)" if env_var else " (checked PATH)"
    raise FileNotFoundError(
        f"CLI binary {name!r} not found{hint}. "
        f"Install it or ensure it is on PATH. On Windows npm shims are often "
        f"{name}.cmd — use a shell or this resolver, not a bare name with "
        f"CreateProcess."
    )


def is_windows_batch_shim(binary: str) -> bool:
    """True when CreateProcess must go through cmd.exe to run this file."""
    if sys.platform != "win32":
        return False
    lower = binary.lower()
    return lower.endswith(".cmd") or lower.endswith(".bat")


def _windows_short_path(path: str) -> str:
    """8.3 form when the OS can make one. A missing name comes back unchanged."""
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(path, buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        return path
    return buffer.value


def _cmd_argv_blocker(arg: str) -> str | None:
    """Text a ``.cmd`` ``%*`` forward cannot carry without changing it."""
    if "\n" in arg or "\r" in arg:
        return "a newline"
    if "%" in arg:
        return "'%'"
    if '"' in arg:
        return "a quote"
    return None


def _prepare_cmd_token(arg: str) -> str:
    """One argv token for ``cmd /c``.

    ``list2cmdline`` quotes whitespace. A bare ``&`` would start another
    command, so those tokens gain a trailing space and the child keeps it.
    Newlines, ``%``, and quotes are rejected: cmd rewrites them.
    """
    blocked = _cmd_argv_blocker(arg)
    if blocked is not None:
        raise RuntimeError(
            f"Windows batch CLI arguments cannot contain {blocked}. "
            "Pass that text via stdin or a file."
        )
    if arg == "" or any(char in arg for char in " \t"):
        return arg
    if any(char in arg for char in _CMD_OPERATORS):
        return arg + " "
    return arg


def _batch_binary_token(binary: str) -> str:
    """Binary path that ``cmd /s /c`` will not split.

    ``/s`` strips the first and last quote when ``/c`` starts with one, so a
    spaced path cannot sit next to another quoted argument. Prefer the 8.3
    name. ``os.name`` guards ``ctypes.windll``, which a ``sys.platform``
    monkeypatch does not provide.
    """
    path = _windows_short_path(binary) if os.name == "nt" else binary
    blocked = _cmd_argv_blocker(path)
    if blocked is not None:
        raise RuntimeError(f"CLI path contains {blocked}, which cmd.exe cannot pass.")
    if any(char in path for char in " \t"):
        raise RuntimeError(
            "Windows batch CLIs in a path with spaces need an 8.3 short name. "
            f"Could not shorten {binary!r}."
        )
    return path


def build_cli_argv(binary: str, *args: str) -> list[str]:
    """Argv for ``create_subprocess_exec``.

    A bare ``.cmd`` name raises WinError 2, and a direct ``.cmd`` path can
    raise WinError 193. Those shims run as tokens after ``cmd /d /s /c``.
    """
    if is_windows_batch_shim(binary):
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        tokens = [
            _batch_binary_token(binary),
            *(_prepare_cmd_token(arg) for arg in args),
        ]
        return [comspec, "/d", "/s", "/c", *tokens]
    return [binary, *args]


async def terminate_cli_process(process: asyncio.subprocess.Process) -> None:
    """Kill a CLI subprocess, including the Windows process tree when needed.

    When the child is ``cmd.exe`` wrapping a ``.cmd`` shim, ``process.kill()``
    only stops cmd and can leave Node/CLI grandchildren running.
    """
    if process.returncode is not None:
        return
    if sys.platform == "win32" and process.pid:
        taskkill_ok = False
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                check=False,
                timeout=10,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            taskkill_ok = result.returncode == 0
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            taskkill_ok = False
        if not taskkill_ok:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except (asyncio.TimeoutError, ProcessLookupError):
            pass
        return
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        await process.wait()
    except ProcessLookupError:
        pass


class BaseCLIProvider(LLMProvider):
    """Base class for CLI-based LLM providers.

    Subclasses must implement:
    - _run_cli_command(): Execute the actual CLI command
    - _get_provider_name(): Return the provider name string
    """

    # Constants
    HEALTH_CHECK_TIMEOUT = 30  # Seconds to wait for health check

    UNSUPPORTED_FLAG_MARKERS = (
        "unexpected argument",
        "unknown option",
        "unrecognized option",
        "no such option",
        "invalid option",
        "unknown flag",
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "default",
        base_url: str | None = None,
        timeout: int = DEFAULT_LLM_TIMEOUT,
        max_retries: int = 3,
    ):
        """Initialize base CLI provider.

        Args:
            api_key: API key (may not be used by CLI providers)
            model: Model name to use
            base_url: Base URL (may not be used by CLI providers)
            timeout: Request timeout in seconds (defaults to DEFAULT_LLM_TIMEOUT)
            max_retries: Number of retry attempts for failed requests
        """
        self._model = model
        self._timeout = timeout
        self._max_retries = max_retries

        # Usage tracking (estimates since CLIs don't return token counts)
        self._requests_made = 0
        self._estimated_tokens_used = 0
        self._estimated_prompt_tokens = 0
        self._estimated_completion_tokens = 0

    @abstractmethod
    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run CLI command and return output.

        This method must be implemented by subclasses to execute their
        specific CLI command.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        ...

    @abstractmethod
    def _get_provider_name(self) -> str:
        """Get the provider name for this CLI provider.

        Returns:
            Provider name (e.g., "claude-code-cli", "codex-cli", "opencode-cli")
        """
        ...

    @property
    def name(self) -> str:
        """Provider name."""
        return self._get_provider_name()

    @property
    def model(self) -> str:
        """Model name."""
        return self._model

    @property
    def timeout(self) -> int:
        """Request timeout in seconds."""
        return self._timeout

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            LLMResponse with content and estimated token usage
        """
        try:
            output_limit = self.resolve_synthesis_output_limit(max_completion_tokens)
            content = await self._run_cli_command(
                prompt, system, output_limit.max_tokens, timeout
            )

            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    f"{self.name} returned empty content "
                    f"(model={self._model}, prompt_length={len(prompt)})"
                )
                raise RuntimeError(
                    f"LLM returned empty response from {self.name}. This may "
                    "indicate a CLI error, authentication issue, or model refusal."
                )

            # Track usage (estimates since CLI doesn't return token counts)
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            return LLMResponse(
                content=content,
                tokens_used=total_tokens,
                model=self._model,
                finish_reason="stop",  # CLI doesn't provide this
            )

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"{self.name} completion failed: {e}")
            raise RuntimeError(f"LLM completion failed: {e}") from e

    async def complete_structured(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        system: str | None = None,
        max_completion_tokens: int | OutputLimitIntent = 4096,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON completion conforming to the given schema.

        Since CLI providers don't support native JSON schema validation,
        we include the schema in the prompt and request JSON output.

        Args:
            prompt: The user prompt
            json_schema: JSON Schema definition for structured output
            system: Optional system prompt
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout in seconds (overrides default)

        Returns:
            Parsed JSON object

        Raises:
            RuntimeError: If output is not valid JSON or doesn't match schema
        """
        # Build structured prompt with schema
        structured_prompt = (
            "Please respond with ONLY valid JSON that conforms to "
            f"this schema:\n\n{json.dumps(json_schema, indent=2)}\n\n"
            f"User request: {prompt}\n\n"
            "Respond with JSON only, no additional text."
        )

        try:
            output_limit = self.resolve_synthesis_output_limit(max_completion_tokens)
            content = await self._run_cli_command(
                structured_prompt, system, output_limit.max_tokens, timeout
            )

            # Validate content is not empty
            if not content or not content.strip():
                logger.error(
                    f"{self.name} structured completion returned empty content"
                )
                raise RuntimeError(
                    f"LLM structured completion returned empty response from "
                    f"{self.name}"
                )

            # Track usage
            self._requests_made += 1
            prompt_tokens = self.estimate_tokens(structured_prompt)
            if system:
                prompt_tokens += self.estimate_tokens(system)
            completion_tokens = self.estimate_tokens(content)
            total_tokens = prompt_tokens + completion_tokens

            self._estimated_prompt_tokens += prompt_tokens
            self._estimated_completion_tokens += completion_tokens
            self._estimated_tokens_used += total_tokens

            parsed = parse_and_validate_structured_json(content, json_schema)

            return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output as JSON: {e}")
            logger.debug(f"Raw output: {content if 'content' in locals() else 'N/A'}")
            raise RuntimeError(f"Invalid JSON in structured output: {e}") from e
        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"{self.name} structured completion failed: {e}")
            raise RuntimeError(f"LLM structured completion failed: {e}") from e

    async def batch_complete(
        self,
        prompts: list[str],
        system: str | None = None,
        max_completion_tokens: int = 4096,
    ) -> list[LLMResponse]:
        """Generate completions for multiple prompts.

        Note: CLI doesn't support true batch API, so we run sequentially
        to avoid overwhelming the CLI or rate limits.
        """
        results = []
        for prompt in prompts:
            result = await self.complete(prompt, system, max_completion_tokens)
            results.append(result)
        return results

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Uses rough approximation since we don't have direct tokenizer access.
        Most models use ~4 characters per token.
        """
        return estimate_tokens_llm(text)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check by attempting a simple completion.

        This will naturally detect if the CLI is missing or incompatible.
        """
        try:
            response = await self.complete(
                "Say 'OK'",
                max_completion_tokens=10,
                timeout=self.HEALTH_CHECK_TIMEOUT,
            )
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": response.content[:50],
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.name,
                "error": str(e),
            }

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics (estimates since CLI doesn't return actual counts)."""
        return {
            "requests_made": self._requests_made,
            "total_tokens_estimated": self._estimated_tokens_used,
            "prompt_tokens_estimated": self._estimated_prompt_tokens,
            "completion_tokens_estimated": self._estimated_completion_tokens,
        }

    def _merge_prompts(self, prompt: str, system: str | None) -> str:
        """Merge an optional system prompt with the user prompt.

        Uses a standard format shared across CLI providers for consistency.
        """
        if system and system.strip():
            return f"System Instructions:\n{system.strip()}\n\nUser Request:\n{prompt}"
        return prompt

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Returns:
            3 for CLI providers (conservative default)
        """
        return 3
