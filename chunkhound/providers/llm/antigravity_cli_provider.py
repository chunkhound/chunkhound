import asyncio
import os
import subprocess
import tempfile
from typing import Any

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider


class AntigravityCLIProvider(BaseCLIProvider):
    """CLI LLM provider wrapping the agy / antigravity CLI command."""

    def _get_provider_name(self) -> str:
        """Get the provider name."""
        return "antigravity-cli"

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        """Run CLI command and return output.

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
        request_timeout = timeout if timeout is not None else self._timeout

        # Build CLI command
        cmd = ["agy", "chat", "--print", "--sandbox", "read-only"]
        if self._model:
            cmd.extend(["--model", self._model])

        # Merge system prompt if provided
        merged_prompt = self._merge_prompts(prompt, system)

        # Clone and sanitize environment variables to prevent plugin hijacking
        env = os.environ.copy()
        keys_to_remove = [
            k for k in env 
            if k.startswith("CHUNKHOUND_") or k.startswith("SDLAIC_") or k.startswith("ENFORCER_")
        ]
        for k in keys_to_remove:
            env.pop(k, None)

        logger.debug(f"Executing CLI command: {' '.join(cmd)} in sandboxed mode")
        
        try:
            # Create subprocess with neutral CWD to prevent local config scans
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=tempfile.gettempdir(),
            )

            # Wait for completion and stream prompt via stdin (bypasses OS arg length limits)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=merged_prompt.encode("utf-8")),
                timeout=request_timeout,
            )

            if process.returncode != 0:
                raw_err = (stderr or stdout or b"").decode("utf-8", errors="ignore").strip()
                logger.error(f"Antigravity CLI command failed (exit code {process.returncode}): {raw_err}")
                raise RuntimeError(raw_err or f"Exit code {process.returncode}")

            return stdout.decode("utf-8", errors="ignore")
            
        except FileNotFoundError as fnf:
            logger.error(f"Antigravity CLI binary not found: {fnf}")
            raise RuntimeError(
                "Antigravity CLI binary ('agy' or 'antigravity') not found. "
                "Ensure it is installed and configured in your system PATH."
            ) from fnf
        except asyncio.TimeoutError as te:
            logger.error(f"Antigravity CLI command timed out after {request_timeout}s")
            raise RuntimeError(f"Antigravity CLI command timed out after {request_timeout}s") from te
        except Exception as e:
            if not isinstance(e, RuntimeError):
                logger.error(f"Antigravity CLI execution failed: {e}")
                raise RuntimeError(f"Antigravity CLI execution failed: {e}") from e
            raise

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations."""
        return 1
