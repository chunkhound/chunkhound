from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from loguru import logger

from chunkhound.core.config.config import Config


def apply_code_mapper_workspace_overrides(
    *,
    config: Config,
    args: argparse.Namespace,
) -> None:
    """Apply Code Mapper-specific workspace overrides to config in-place."""
    cfg_override: Path | None = None
    if getattr(args, "config", None):
        try:
            cfg_override = Path(args.config).expanduser().resolve()
        except (OSError, RuntimeError):
            cfg_override = None
    if cfg_override is None:
        config_file_env = os.getenv("CHUNKHOUND_CONFIG_FILE")
        if config_file_env:
            try:
                cfg_override = Path(config_file_env).expanduser().resolve()
            except (OSError, RuntimeError):
                cfg_override = None

    if cfg_override is None:
        return

    workspace_root = cfg_override.parent
    if not workspace_root.exists():
        return

    config.target_dir = workspace_root

    # Do NOT override an explicitly configured database path.
    explicit_db_cli = bool(
        getattr(args, "db", None) or getattr(args, "database_path", None)
    )
    explicit_db_in_file = False
    try:
        raw = json.loads(cfg_override.read_text(encoding="utf-8"))
        db = raw.get("database") if isinstance(raw, dict) else None
        explicit_db_in_file = isinstance(db, dict) and bool(db.get("path"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        logger.debug(f"Code Mapper: failed to parse config override: {exc}")
        explicit_db_in_file = False

    if not explicit_db_cli and not explicit_db_in_file:
        config.database.path = workspace_root / ".chunkhound" / "db"
