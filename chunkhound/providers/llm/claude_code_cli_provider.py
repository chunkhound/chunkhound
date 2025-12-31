"""Claude Code CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the Claude Code CLI (claude --print) to enable deep research
using the user's existing Claude subscription instead of API credits.

Note: This provider is configured for vanilla LLM behavior:
- All tools disabled via --tools ""
- MCP servers disabled via empty --mcp-config
- Workspace isolation (runs from temp directory to prevent context gathering)
- Clean API access without workspace overhead
"""

import asyncio
import os
import subprocess
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.text_sanitization import sanitize_error_text


class ClaudeCodeCLIProvider(BaseCLIProvider):
    """Claude Code CLI provider using subprocess calls to claude --print."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize Claude Code CLI provider.

        Args:
            api_key: Not used (subscription-based authentication)
            model: Model name to use (e.g., "claude-sonnet-4-5-20250929")
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        super().__init__(api_key, model, base_url, timeout, max_retries)

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "claude-code-cli"

    def _map_model_to_cli_arg(self, model: str) -> str:
        """Map full model name to CLI model argument.

        The Claude Code CLI accepts full model names directly
        (e.g., "claude-sonnet-4-5-20250929"). Short names like "sonnet-4-5"
        are NOT accepted and result in exit code 1.

        Args:
            model: Full model name (e.g., "claude-sonnet-4-5-20250929")

        Returns:
            CLI model argument (same as input - full model name)
        """
        # Pass through model name as-is - CLI requires full names
        return model

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run claude CLI command and return output.

        Args:
            prompt: User prompt
            system: Optional system prompt (appended to default)
            max_completion_tokens: Maximum tokens to generate
            timeout: Optional timeout override

        Returns:
            CLI output text

        Raises:
            RuntimeError: If CLI command fails
        """
        # Build CLI command
        model_arg = self._map_model_to_cli_arg(self._model)
        cmd = ["claude", "--print", "--model", model_arg, "--output-format", "text"]

        # Disable all tools for vanilla LLM behavior (more future-proof than --disallowedTools)
        cmd.extend(["--tools", ""])

        # Prevent MCP server loading for clean LLM access
        cmd.extend(["--mcp-config", '{"mcpServers":{}}'])
        cmd.append("--strict-mcp-config")  # Ignore user/project MCP configs

        # Prevent session persistence (avoid context bleed between calls)
        cmd.append("--no-session-persistence")

        # Add system prompt if provided (appends to default)
        if system:
            cmd.extend(["--append-system-prompt", system])

        # Note: prompt is passed via stdin, not CLI args (avoids ARG_MAX limit)

        # Set environment for subscription-based auth
        env = os.environ.copy()
        env["CLAUDE_USE_SUBSCRIPTION"] = "true"

        # Remove ANTHROPIC_API_KEY if present to force subscription auth
        env.pop("ANTHROPIC_API_KEY", None)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self._timeout

        # Run command with retry logic
        last_error = None
        for attempt in range(self._max_retries):
            process = None
            try:
                # Create subprocess with neutral CWD to prevent workspace scanning
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=subprocess.PIPE,  # Pass prompt via stdin (avoids ARG_MAX)
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=tempfile.gettempdir(),  # Cross-platform temp directory
                )

                # Wrap communicate() with timeout (this is the long-running part)
                # Pass prompt via stdin to avoid OS ARG_MAX limits (~256KB on macOS)
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=prompt.encode("utf-8")),
                    timeout=request_timeout,
                )

                if process.returncode != 0:
                    raw_err = (stderr or stdout or b"").decode("utf-8", errors="ignore")
                    error_msg = sanitize_error_text(raw_err.strip()) or f"Exit code {process.returncode}"
                    last_error = RuntimeError(
                        f"CLI command failed (exit {process.returncode}): {error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"CLI attempt {attempt + 1} failed, retrying: {error_msg}"
                        )
                        continue
                    raise last_error

                return stdout.decode("utf-8").strip()

            except asyncio.TimeoutError as e:
                # Kill the subprocess if it's still running
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(
                    f"CLI command timed out after {request_timeout}s"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} timed out, retrying")
                    continue
                raise last_error from e

            except Exception as e:
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(f"CLI command failed: {e}")
                if attempt < self._max_retries - 1:
                    logger.warning(f"CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("CLI command failed after retries")
