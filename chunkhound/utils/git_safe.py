from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class GitCommandError(Exception):
    msg: str
    returncode: int | None = None
    stderr: str | None = None

    def __str__(self) -> str:
        base = self.msg
        if self.returncode is not None:
            base += f" (rc={self.returncode})"
        if self.stderr:
            base += f" :: {self.stderr.strip()}"
        return base


def _build_git_env() -> dict[str, str]:
    env = {}
    # Preserve PATH for finding git
    env["PATH"] = os.environ.get("PATH", "")
    # Keep locale deterministic
    env["LC_ALL"] = os.environ.get("LC_ALL", "C")
    # Prevent reading user/system git configs
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Point global/system config to null devices (best-effort cross-platform)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    return env


def get_global_excludes_file() -> Path | None:
    """Get the path to the global gitignore file from git config.

    Reads core.excludesFile from git config (without the isolation env
    that run_git uses) to support global gitignore patterns.

    Returns:
        Path to global excludes file if configured and exists, None otherwise.
    """
    try:
        # Run git config WITHOUT the isolation env to read user config
        proc = subprocess.run(
            ["git", "config", "--global", "--get", "core.excludesFile"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            path = Path(proc.stdout.strip()).expanduser()
            if path.exists():
                return path
    except Exception:
        pass

    # Fallback: check default locations
    for default in [
        Path.home() / ".gitignore_global",
        Path.home() / ".gitignore",
        Path.home() / ".config" / "git" / "ignore",
    ]:
        if default.exists():
            return default

    return None


def run_git(args: Sequence[str], cwd: Path | None, timeout_s: float | None = None) -> subprocess.CompletedProcess:
    cmd = ["git", *list(args)]
    env = _build_git_env()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=env,
            timeout=timeout_s if timeout_s is not None else float(os.environ.get("CHUNKHOUND_GIT_TIMEOUT_SECONDS", "15")),
            text=True,
        )
        return proc
    except subprocess.TimeoutExpired as te:
        raise GitCommandError("git command timeout", None, None) from te
    except Exception as e:
        raise GitCommandError(f"git command failed: {e}") from e

