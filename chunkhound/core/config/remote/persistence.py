"""Backup + write for the remote-config pipeline.

Contract:
- Target file resolution: ``CHUNKHOUND_GLOBAL_CONFIG_FILE`` wins if set;
  otherwise the first existing candidate from
  ``Config._get_global_config_candidates()``. If none exist, fall back to
  the **last** candidate — ``~/.chunkhound.json`` — because that's the
  path the loader will discover on the *next* run, keeping fetch and read
  in agreement.
- Backup: ``shutil.copy2(target, target + ".bak")``. Skipped on first-time
  file creation (nothing to preserve).
- Any ``OSError`` on backup or write is a **fatal** operator-visible
  failure — ``sys.exit(1)`` with a message naming source, target, and errno.
  A silent write failure would let the process keep running against a
  stale on-disk copy while advertising success, so the operator loses
  the "did this apply?" signal at exactly the moment they need it most.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from chunkhound.core.config.config import Config
from chunkhound.utils.atomic_write import write_json_atomically


def resolve_target() -> Path:
    """Return the disk path the pipeline will read/write.

    Mirrors the loader's global-JSON resolution in
    ``Config._apply_global_json`` so fetch and read stay in agreement:
    env var wins, else first existing candidate, else last candidate
    as a write-target fallback for the next-run loader to discover.

    All branches return a resolved (absolute, symlink-normalized) path.
    Chiefly this normalizes a relative ``CHUNKHOUND_GLOBAL_CONFIG_FILE``
    against CWD so the write target doesn't drift with the shell's cwd,
    and it makes the pipeline's target compare equal to the loader's
    ``.resolve()``-stored ``Config.global_config_file`` under a symlinked
    HOME (e.g. macOS ``/tmp`` → ``/private/tmp``).
    """
    env_global = os.getenv("CHUNKHOUND_GLOBAL_CONFIG_FILE")
    if env_global:
        return Path(env_global).resolve()
    candidates = Config._get_global_config_candidates()
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def read_target(target: Path) -> dict[str, Any]:
    """Read the on-disk global JSON, or return an empty dict if absent."""
    if not target.exists():
        return {}
    with open(target) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def backup_and_write(target: Path, working_copy: dict[str, Any]) -> None:
    """Back up ``target`` (if it exists) and write ``working_copy`` as JSON.

    Any OSError escalates to ``sys.exit(1)`` — the operator needs a loud
    signal that persistence broke (permissions, disk full, EROFS, etc.).
    """
    is_new_file = not target.exists()

    if not is_new_file:
        backup = target.with_suffix(target.suffix + ".bak")
        try:
            shutil.copy2(target, backup)
        except OSError as exc:
            sys.stderr.write(
                f"remote-config: fatal — backup {target} -> {backup} "
                f"failed (errno={exc.errno}): {exc}\n"
            )
            sys.exit(1)
        # Match the 0o600 the primary write applies below — the backup
        # holds the same secret-bearing contents (embedding.api_key,
        # etc.), so preserving a laxer pre-existing mode via copy2
        # would leave a world-readable snapshot next to a locked-down
        # primary. Best-effort, mirroring atomic_write's own private-mode
        # fallback; skipped on Windows (no POSIX mode bits).
        if sys.platform != "win32":
            try:
                os.chmod(backup, 0o600)
            except OSError:
                pass

    try:
        write_json_atomically(
            target,
            working_copy,
            private=True,
            indent=2,
            sort_keys=True,
            trailing_newline=True,
        )
    except OSError as exc:
        sys.stderr.write(
            f"remote-config: fatal — write {target} failed "
            f"(errno={exc.errno}): {exc}\n"
        )
        sys.exit(1)
