"""Codex CLI LLM provider for ChunkHound.

Wraps `codex exec` to run local-agent synthesis using the user's Codex
credentials and configuration. Designed for the final synthesis step in
code_research; keeps MCP stdio clean by never printing to stdout.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.providers.llm.base_cli_provider import BaseCLIProvider
from chunkhound.utils.text_sanitization import sanitize_error_text


class CodexCLIProvider(BaseCLIProvider):
    """Provider that shells out to `codex exec`.

    Notes
    - Uses stdin to avoid argv length limits for large prompts.
    - Defaults to an isolated, read-only, non-interactive run (no history).
    - Never writes to stdout; only returns captured content to caller.
    """

    # Token estimate (chars per token) — align with CLI providers
    TOKEN_CHARS_RATIO = 4

    # Timeouts used in health checks (seconds)
    VERSION_CHECK_TIMEOUT = 5
    HEALTH_CHECK_TIMEOUT = 30

    def __init__(
        self,
        api_key: str | None = None,  # Unused (CLI handles auth)
        model: str = "codex",
        base_url: str | None = None,  # Unused
        timeout: int = 60,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(api_key, model, base_url, timeout, max_retries)
        self._reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)

        if not self._codex_available():
            logger.warning("Codex CLI not found in PATH (codex)")

    # ----- Internals -----

    def _get_base_codex_home(self) -> Path | None:
        base = os.getenv("CODEX_HOME")
        if base:
            p = Path(base).expanduser()
            return p if p.exists() else None
        # Default location
        default = Path.home() / ".codex"
        return default if default.exists() else None

    def _resolve_model_name(self, requested: str | None) -> str:
        """Resolve requested model name to Codex CLI model identifier."""
        resolved, _source = self.describe_model_resolution(requested)
        return resolved

    @classmethod
    def describe_model_resolution(cls, requested: str | None) -> tuple[str, str]:
        """Return (resolved_model, source) for Codex CLI model selection.

        Notes:
        - The special value "codex" means "use ChunkHound's default".
        - Override defaults via CHUNKHOUND_CODEX_DEFAULT_MODEL.
        """
        env_override = os.getenv("CHUNKHOUND_CODEX_DEFAULT_MODEL")
        # Default to a Codex-optimized reasoning model unless explicitly overridden.
        default_model = env_override.strip() if env_override else "gpt-5.1-codex"
        default_source = (
            "env:CHUNKHOUND_CODEX_DEFAULT_MODEL" if env_override else "default"
        )

        if not requested:
            return default_model, default_source

        model_name = requested.strip()
        if not model_name or model_name.lower() == "codex":
            return default_model, default_source

        return model_name, "explicit"

    def _resolve_reasoning_effort(self, requested: str | None) -> str:
        """Resolve reasoning effort override."""
        effort, _source = self.describe_reasoning_effort_resolution(requested)
        return effort

    @classmethod
    def describe_reasoning_effort_resolution(cls, requested: str | None) -> tuple[str, str]:
        """Return (resolved_effort, source) for Codex CLI reasoning effort selection."""
        env_override = os.getenv("CHUNKHOUND_CODEX_REASONING_EFFORT")
        candidate = requested or env_override
        allowed = {"minimal", "low", "medium", "high", "xhigh"}

        if not candidate:
            return "low", "default"

        effort = candidate.strip().lower()
        if effort not in allowed:
            logger.warning(
                "Unknown Codex reasoning effort '%s'; falling back to 'low'", candidate
            )
            return "low", "fallback"

        if requested:
            return effort, "explicit"
        if env_override:
            return effort, "env:CHUNKHOUND_CODEX_REASONING_EFFORT"
        return effort, "default"

    def _resolve_sandbox_mode(self, requested: str | None = None) -> str:
        """Resolve sandbox mode for codex exec command execution."""
        env_override = os.getenv("CHUNKHOUND_CODEX_SANDBOX_MODE")
        candidate = (requested or env_override or "read-only").strip().lower()
        allowed = {"read-only", "workspace-write", "danger-full-access"}
        if candidate not in allowed:
            logger.warning(
                "Unknown Codex sandbox mode '%s'; falling back to 'read-only'", candidate
            )
            return "read-only"
        return candidate

    def _resolve_approval_policy(self, requested: str | None = None) -> str:
        """Resolve approval policy for codex exec tool execution."""
        env_override = os.getenv("CHUNKHOUND_CODEX_APPROVAL_POLICY")
        # Default to on-request to discourage command execution in non-interactive runs.
        candidate = (requested or env_override or "on-request").strip().lower()
        allowed = {"untrusted", "on-failure", "on-request", "never"}
        if candidate not in allowed:
            logger.warning(
                "Unknown Codex approval policy '%s'; falling back to 'on-request'",
                candidate,
            )
            return "on-request"
        return candidate

    @staticmethod
    def _toml_string(value: str) -> str:
        # Codex parses `-c key=value` with TOML semantics. Use explicit strings to
        # avoid ambiguity across CLI versions.
        return '"' + value.replace('"', '\\"') + '"'

    def _extract_agent_message_from_jsonl(self, stdout_text: str) -> tuple[str | None, dict[str, Any] | None]:
        """Extract final agent message text and usage from `codex exec --json` output."""
        import json

        last_message: str | None = None
        usage: dict[str, Any] | None = None

        for line in stdout_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if obj.get("type") == "item.completed":
                item = obj.get("item") or {}
                if item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        last_message = text
            elif obj.get("type") == "turn.completed":
                u = obj.get("usage")
                if isinstance(u, dict):
                    usage = u

        return last_message, usage

    def _copy_minimal_codex_state(self, base: Path, dest: Path) -> None:
        """Copy minimal auth/session state into destination CODEX_HOME."""
        copy_all = os.getenv("CHUNKHOUND_CODEX_COPY_ALL", "0") == "1"
        max_bytes = int(os.getenv("CHUNKHOUND_CODEX_MAX_COPY_BYTES", "1000000"))

        def _should_copy_dir(name: str) -> bool:
            n = name.lower()
            if copy_all:
                return True
            # Likely auth/session state we may need
            return n in {"sessions", "session", "auth", "profiles", "state"}

        def _should_copy_file(p: Path) -> bool:
            if copy_all:
                return True
            if p.name.lower() == "config.toml":
                return False  # we write our own config below
            if p.suffix.lower() in {".json", ".toml", ".ini"}:
                try:
                    return p.stat().st_size <= max_bytes
                except Exception:
                    return False
            return False

        for item in base.iterdir():
            dest_path = dest / item.name
            try:
                if item.is_dir():
                    if _should_copy_dir(item.name):
                        shutil.copytree(item, dest_path, dirs_exist_ok=False)
                else:
                    if _should_copy_file(item):
                        shutil.copy2(item, dest_path)
            except Exception:
                # Best-effort copy; skip unreadable items
                pass

    def _build_overlay_home(self, model_override: str | None = None) -> str:
        """Create an overlay CODEX_HOME inheriting auth but overriding config.

        - Copies the base CODEX_HOME (if it exists) to a temp dir
        - Replaces config.toml with a minimal one (no MCP, no history persistence)
        - Sets fast model defaults (best effort)
        - Copies only a minimal subset by default to reduce exposure
        """
        overlay = Path(tempfile.mkdtemp(prefix="chunkhound-codex-overlay-"))
        base = self._get_base_codex_home()
        model_name = self._resolve_model_name(model_override or self._model)
        try:
            if base and base.exists():
                self._copy_minimal_codex_state(base, overlay)

            config_path = overlay / "config.toml"
            # Many Codex builds expect top-level `model` keys (not a [model] table).
            cfg_lines = [
                f'model = "{model_name}"',
                f'model_reasoning_effort = "{self._reasoning_effort}"',
                "",
                "[history]",
                'persistence = "none"',
            ]
            config_path.write_text("\n".join(cfg_lines) + "\n", encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to build Codex overlay home: {e}")
        return str(overlay)

    def _codex_available(self) -> bool:
        """Return True if `codex` binary looks available (sync check)."""
        import subprocess

        try:
            res = subprocess.run(
                ["codex", "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.VERSION_CHECK_TIMEOUT,
                check=False,
            )
            # Any exit status implies the binary exists; only ENOENT means absent
            return res.returncode is not None
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    async def _run_exec(
        self,
        content: str,
        *,
        cwd: str | None = None,
        max_tokens: int,
        timeout: int | None,
        model: str | None,
    ) -> str:
        """Run `codex exec` and capture stdout with robust fallbacks."""
        binary = os.getenv("CHUNKHOUND_CODEX_BIN", "codex")
        overlay_home: str | None = None
        config_file_path: str | None = None
        extra_args: list[str] = []

        env = os.environ.copy()
        effective_model = self._resolve_model_name(model or self._model)
        keep_overlay = os.getenv("CHUNKHOUND_CODEX_KEEP_OVERLAY", "0") == "1"

        # Optional verbose diagnostics for large prompts / transport issues.
        debug_codex = os.getenv("CHUNKHOUND_CODEX_DEBUG", "0") == "1"
        json_mode = os.getenv("CHUNKHOUND_CODEX_JSON", "0") == "1"
        content_chars = len(content)
        estimated_tokens = self.estimate_tokens(content)
        if debug_codex:
            logger.debug(
                "Codex CLI request: model=%s, chars=%d, est_tokens=%d, max_tokens=%d",
                effective_model,
                content_chars,
                estimated_tokens,
                max_tokens,
            )

        # Helper to forward selected env keys
        def _forward_env(keys: list[str]) -> None:
            for k in keys:
                v = os.environ.get(k)
                if v is not None:
                    env[k] = v

        auth_keys = [s.strip() for s in os.getenv(
            "CHUNKHOUND_CODEX_AUTH_ENV",
            "OPENAI_API_KEY,CODEX_API_KEY,ANTHROPIC_API_KEY,BEARER_TOKEN",
        ).split(",") if s.strip()]
        passthrough_keys = [s.strip() for s in os.getenv(
            "CHUNKHOUND_CODEX_PASSTHROUGH_ENV",
            "",
        ).split(",") if s.strip()]

        overlay_home = self._build_overlay_home(effective_model)
        env["CODEX_HOME"] = overlay_home
        config_file_path = str(Path(overlay_home) / "config.toml")

        # Reduce risk of slow/hanging command execution by forcing a strict sandbox
        # and a non-interactive approval policy.
        sandbox_mode = self._resolve_sandbox_mode()
        approval_policy = self._resolve_approval_policy()

        # CLI flag covers sandboxing; approval policy is only configurable via -c.
        extra_args += ["--sandbox", sandbox_mode]
        extra_args += [
            "-c",
            f"approval_policy={self._toml_string(approval_policy)}",
        ]

        # Make reasoning effort explicit per-call (helps when using base Codex config overlays).
        extra_args += [
            "-c",
            f"model_reasoning_effort={self._toml_string(self._reasoning_effort)}",
        ]

        # Enforce per-call output cap. Codex CLI supports config overrides via `-c key=value`.
        # Per docs, the key is `model_max_output_tokens` (caps completion length for Responses API).
        # This keeps ChunkHound's `max_completion_tokens` meaningful for the CLI provider and helps
        # prevent long "think+write" runs when the prompt requests overly-large outputs.
        extra_args += ["-c", f"model_max_output_tokens={int(max_tokens)}"]

        override_mode = os.getenv("CHUNKHOUND_CODEX_CONFIG_OVERRIDE", "env").strip().lower()
        if config_file_path:
            if override_mode == "flag":
                flag = os.getenv("CHUNKHOUND_CODEX_CONFIG_FLAG", "--config")
                extra_args += [flag, config_file_path]
            else:
                cfg_key = os.getenv("CHUNKHOUND_CODEX_CONFIG_ENV", "CODEX_CONFIG")
                env[cfg_key] = config_file_path

        _forward_env(auth_keys + passthrough_keys)

        # Non-interactive defaults to avoid hangs in subprocess tools.
        env.setdefault("CI", "1")
        env.setdefault("GIT_TERMINAL_PROMPT", "0")
        env.setdefault("PAGER", "cat")
        env.setdefault("GIT_PAGER", "cat")
        env.setdefault("TERM", "dumb")
        env.setdefault("NO_COLOR", "1")

        request_timeout = timeout if timeout is not None else self._timeout

        # Privacy-first strategy: use stdin by default to avoid argv leaking prompt in process list.
        # If the CLI rejects stdin, we fallback to argv.
        # Legacy behavior can be restored by setting CHUNKHOUND_CODEX_STDIN_FIRST=0, which will
        # use argv for small prompts and switch to stdin only for very large inputs.
        MAX_ARG_CHARS = int(os.getenv("CHUNKHOUND_CODEX_ARG_LIMIT", "200000"))
        stdin_first = os.getenv("CHUNKHOUND_CODEX_STDIN_FIRST", "1") != "0"
        use_stdin = True if stdin_first else (len(content) > MAX_ARG_CHARS)
        if debug_codex:
            logger.debug(
                "Codex CLI transport selection: stdin_first=%s, use_stdin=%s, MAX_ARG_CHARS=%d",
                stdin_first,
                use_stdin,
                MAX_ARG_CHARS,
            )
        # Newer Codex builds require --skip-git-repo-check; default to passing
        # it on the first attempt to avoid noisy negotiation warnings. This
        # can be disabled via CHUNKHOUND_CODEX_SKIP_GIT_CHECK=0 if needed for
        # older binaries.
        add_skip_git = os.getenv("CHUNKHOUND_CODEX_SKIP_GIT_CHECK", "1") != "0"

        last_error: Exception | None = None
        try:
            for attempt in range(self._max_retries):
                proc: asyncio.subprocess.Process | None = None
                try:
                    if use_stdin:
                        if debug_codex:
                            logger.debug(
                                "Codex CLI attempt %d using stdin transport (skip_git=%s)",
                                attempt + 1,
                                add_skip_git,
                            )
                        proc = await asyncio.create_subprocess_exec(
                            binary,
                            "exec",
                            "-",
                            *( ["--json"] if json_mode else [] ),
                            *extra_args,
                            *( ["--skip-git-repo-check"] if add_skip_git else [] ),
                            cwd=cwd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        assert proc.stdin is not None
                        proc.stdin.write(content.encode("utf-8"))
                        await proc.stdin.drain()
                        proc.stdin.close()
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=request_timeout
                        )
                    else:
                        # argv mode
                        if debug_codex:
                            logger.debug(
                                "Codex CLI attempt %d using argv transport (skip_git=%s)",
                                attempt + 1,
                                add_skip_git,
                            )
                        proc = await asyncio.create_subprocess_exec(
                            binary,
                            "exec",
                            content,
                            *( ["--json"] if json_mode else [] ),
                            *extra_args,
                            *( ["--skip-git-repo-check"] if add_skip_git else [] ),
                            cwd=cwd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            env=env,
                        )
                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=request_timeout
                        )

                    if proc.returncode != 0:
                        raw_err = stderr.decode("utf-8", errors="ignore")
                        err = self._sanitize_text(raw_err)
                        if debug_codex:
                            logger.debug(
                                "Codex CLI non-zero exit (attempt %d, use_stdin=%s): %s",
                                attempt + 1,
                                use_stdin,
                                err,
                            )
                        if add_skip_git and self._skip_git_flag_unsupported(err):
                            add_skip_git = False
                            logger.warning(
                                "codex exec does not support --skip-git-repo-check; retrying without flag"
                            )
                            continue
                        err_lower = err.lower()

                        # Skip-git repo check negotiation for newer Codex builds
                        if "skip-git-repo-check" in err and not add_skip_git:
                            add_skip_git = True
                            logger.warning("codex exec requires --skip-git-repo-check; retrying with flag")
                            continue
                        # Some older Codex builds may reject the flag; fall back by removing it.
                        if add_skip_git and "skip-git-repo-check" in err_lower and (
                            "unknown option" in err_lower
                            or "unrecognized option" in err_lower
                            or "unknown flag" in err_lower
                            or "unexpected argument" in err_lower
                        ):
                            add_skip_git = False
                            logger.warning(
                                "codex exec does not support --skip-git-repo-check; retrying without flag"
                            )
                            continue

                        # If stdin failed (e.g., BrokenPipe or codex not reading stdin), fall back to argv with truncation.
                        if use_stdin and ("broken pipe" in err_lower or "stdin" in err_lower):
                            use_stdin = False
                            logger.warning("codex exec stdin not supported; retrying with argv mode")
                            continue
                        last_error = RuntimeError(
                            f"codex exec failed (exit {proc.returncode}): {err}"
                        )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                f"codex exec attempt {attempt + 1} failed: {err}; retrying"
                            )
                            continue
                        raise last_error

                    stdout_text = stdout.decode("utf-8", errors="ignore").strip()
                    if json_mode:
                        message, usage = self._extract_agent_message_from_jsonl(stdout_text)
                        if debug_codex and usage:
                            logger.debug("Codex CLI usage: %s", usage)
                        if message and message.strip():
                            return message.strip()
                        # Fall back to raw output if we couldn't parse an agent message.
                    return stdout_text

                except asyncio.TimeoutError as e:
                    if proc and proc.returncode is None:
                        proc.kill()
                        await proc.wait()
                    last_error = RuntimeError(
                        f"codex exec timed out after {request_timeout}s"
                    )
                    if debug_codex:
                        logger.debug(
                            "Codex CLI timeout on attempt %d after %ds",
                            attempt + 1,
                            request_timeout,
                        )
                    if attempt < self._max_retries - 1:
                        logger.warning(
                            f"codex exec attempt {attempt + 1} timed out; retrying"
                        )
                        continue
                    raise last_error from e
                except (BrokenPipeError, ConnectionResetError) as e:
                    # Treat BrokenPipe/connection reset on stdin as "no stdin support" and fall back to argv
                    if use_stdin:
                        use_stdin = False
                        if debug_codex:
                            logger.debug(
                                "Codex CLI stdin connection lost on attempt %d; switching to argv",
                                attempt + 1,
                            )
                        if attempt < self._max_retries - 1:
                            logger.warning(
                                "codex exec stdin connection lost; retrying with argv mode"
                            )
                            continue
                        raise RuntimeError(
                            "codex exec failed: stdin connection lost and no retries left"
                        ) from e
                    raise
                except OSError as e:
                    # Handle OS-level argv length errors by switching to stdin mode
                    if e.errno == 7:  # Argument list too long
                        if not use_stdin:
                            use_stdin = True
                            if debug_codex:
                                logger.debug(
                                    "Codex CLI argv too long on attempt %d; switching to stdin",
                                    attempt + 1,
                                )
                            logger.warning("codex exec argv too long; retrying with stdin mode")
                            continue
                    raise
                # Let unexpected exceptions propagate; overlay cleanup happens in the outer finally
                finally:
                    # No per-attempt cleanup of overlay; cleanup performed in outer finally
                    pass

        finally:
            # Cleanup temporary resources regardless of success or failure
            try:
                if overlay_home and Path(overlay_home).exists():
                    if keep_overlay:
                        logger.warning(
                            "CHUNKHOUND_CODEX_KEEP_OVERLAY=1; preserving Codex overlay at %s",
                            overlay_home,
                        )
                    else:
                        shutil.rmtree(overlay_home, ignore_errors=True)
            except Exception:
                pass

        raise last_error or RuntimeError("codex exec failed after retries")

    def _sanitize_text(self, s: str, max_len: int | None = None) -> str:
        """Truncate and redact secrets. Delegates to shared utility."""
        limit = max_len or int(os.getenv("CHUNKHOUND_CODEX_LOG_MAX_ERR", "800"))
        return sanitize_error_text(s, max_length=limit)

    def _merge_prompts(self, prompt: str, system: str | None) -> str:
        if system and system.strip():
            return f"System Instructions:\n{system.strip()}\n\nUser Request:\n{prompt}"
        return prompt

    def _skip_git_flag_unsupported(self, err: str) -> bool:
        lowered = err.lower()
        if "--skip-git-repo-check" not in lowered:
            return False
        unsupported_markers = (
            "unexpected argument",
            "unknown option",
            "unrecognized option",
            "no such option",
            "invalid option",
        )
        return any(marker in lowered for marker in unsupported_markers)

    # ----- BaseCLIProvider hooks -----

    def _get_provider_name(self) -> str:
        return "codex-cli"

    async def _run_cli_command(
        self,
        prompt: str,
        system: str | None = None,
        max_completion_tokens: int | None = None,
        timeout: int | None = None,
    ) -> str:
        text = self._merge_prompts(prompt, system)
        max_tokens = max_completion_tokens if max_completion_tokens is not None else 4096
        return await self._run_exec(
            text,
            cwd=None,
            max_tokens=max_tokens,
            timeout=timeout,
            model=self._model,
        )

    async def health_check(self) -> dict[str, Any]:
        if not self._codex_available():
            return {"status": "unhealthy", "provider": self.name, "error": "codex not found"}
        try:
            sample = await self.complete("Say 'OK'", max_completion_tokens=10, timeout=self.HEALTH_CHECK_TIMEOUT)
            return {
                "status": "healthy",
                "provider": self.name,
                "model": self._model,
                "test_response": sample.content[:50],
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "unhealthy", "provider": self.name, "error": str(e)}

    def get_synthesis_concurrency(self) -> int:
        """Get recommended concurrency for parallel synthesis operations.

        Default aligns with Anthropic (5), but remains configurable via env var.
        """
        try:
            v = int(os.getenv("CHUNKHOUND_CODEX_SYNTHESIS_CONCURRENCY", "5"))
        except Exception:
            return 5
        return max(1, min(5, v))
