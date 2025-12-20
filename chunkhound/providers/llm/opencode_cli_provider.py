"""OpenCode CLI LLM provider implementation for ChunkHound deep research.

This provider wraps the OpenCode CLI (opencode run) to enable deep research
using the user's existing OpenCode configuration and access to 75+ LLM providers.

Note: This provider is configured for vanilla LLM behavior:
- Uses default text format for simple, reliable output
- Runs in non-interactive mode via opencode run
- Leverages existing opencode auth login credentials
- Supports all providers/models available via "opencode models"
"""

import asyncio
import os
import subprocess
import tempfile

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider


class OpenCodeCLIProvider(BaseCLIProvider):
    """OpenCode CLI provider using subprocess calls to opencode run."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "opencode/grok-code",
        base_url: str | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        """Initialize OpenCode CLI provider.

        Args:
            api_key: Not used (credentials managed by opencode auth)
            model: Model name to use in provider/model format
                (e.g., "opencode/grok-code")
            base_url: Not used (CLI uses default endpoints)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
        """
        super().__init__(api_key, model, base_url, timeout, max_retries)

        # Check CLI availability
        if not self._opencode_available():
            logger.warning("OpenCode CLI not found in PATH")

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "opencode-cli"

    def _opencode_available(self) -> bool:
        """Check if opencode CLI is available in PATH."""
        try:
            result = subprocess.run(
                ["opencode", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _validate_model_format(self, model: str) -> None:
        """Validate that model follows provider/model format.

        Args:
            model: Model string to validate

        Raises:
            ValueError: If model format is invalid
        """
        if "/" not in model:
            raise ValueError(
                f"Model must be in 'provider/model' format, got: {model}. "
                f"Run 'opencode models' to see available models."
            )

        provider, _ = model.split("/", 1)
        if not provider:
            raise ValueError(f"Provider cannot be empty in model: {model}")

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run opencode CLI command and return output.

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
        # Validate model format
        self._validate_model_format(self._model)

        # Build CLI command
        cmd = [
            "opencode",
            "run",
            "--model",
            self._model,
        ]

        # Add system prompt if provided
        if system:
            cmd.append(system + "\n" + prompt)
        else:
            cmd.append(prompt)

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
                    stdin=subprocess.DEVNULL,  # Prevent stdin inheritance
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),  # Use copy of environment
                    cwd=tempfile.gettempdir(),  # Cross-platform temp directory
                )

                # Wrap communicate() with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=request_timeout,
                )

                if process.returncode != 0:
                    error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                    last_error = RuntimeError(
                        f"OpenCode CLI command failed (exit {process.returncode}): "
                        f"{error_msg}"
                    )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"OpenCode CLI attempt {attempt + 1} failed, "
                            f"retrying: {error_msg}"
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
                    f"OpenCode CLI command timed out after {request_timeout}s"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(
                        f"OpenCode CLI attempt {attempt + 1} timed out, retrying"
                    )
                    continue
                raise last_error from e

            except Exception as e:
                # Kill the subprocess if it's still running on unexpected errors
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()

                last_error = RuntimeError(
                    f"OpenCode CLI command failed: {e}, with command {cmd}"
                )
                if attempt < self._max_retries - 1:
                    logger.warning(f"OpenCode CLI attempt {attempt + 1} failed: {e}")
                    continue
                raise last_error from e

        # Should not reach here, but just in case
        raise last_error or RuntimeError("OpenCode CLI command failed after retries")
