#!/usr/bin/env python3
"""Dump ChunkHound config resolution for a project directory.

Prints which global/local/explicit config files were loaded, a short LLM
summary, and the full resolved Config as JSON (secrets redacted by default).

Uses the same Config(args=...) discovery as CLI/MCP for project path and
--config (env + global + local + explicit). Does not replay other CLI flags
(--db, --llm-*, provider overrides, etc.).

For relative paths inside the config (e.g. database.path), Config resolves
them against process cwd — prefer ``cd`` into the project (matching MCP cwd)
rather than only passing --path from elsewhere.

Examples:
  uv run python scripts/dump_resolved_config.py
  uv run python scripts/dump_resolved_config.py --path F:\\path\\to\\project
  uv run python scripts/dump_resolved_config.py --config .chunkhound.json
  uv run python scripts/dump_resolved_config.py --show-secrets
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, SecretStr


# Keys whose values are masked unless --show-secrets is set.
_SECRET_KEY_NAMES = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "client_secret",
        "authorization",
        "auth_token",
    }
)


def _is_secret_key(key: str) -> bool:
    key_l = key.lower()
    if key_l in _SECRET_KEY_NAMES or key_l.endswith("_api_key"):
        return True
    # Avoid matching non-secrets like max_tokens / target_tokens.
    if key_l.endswith("_secret") or key_l.endswith("_password"):
        return True
    return False


def _unwrap_value(value: Any, *, show_secrets: bool) -> Any:
    """Convert model values to JSON-friendly forms; handle SecretStr explicitly."""
    if isinstance(value, SecretStr):
        raw = value.get_secret_value()
        if show_secrets:
            return raw
        return "***REDACTED***" if raw else raw
    if isinstance(value, BaseModel):
        return _model_to_dict(value, show_secrets=show_secrets)
    if isinstance(value, dict):
        return {
            k: (
                "***REDACTED***"
                if _is_secret_key(str(k))
                and not show_secrets
                and v not in (None, "")
                and not isinstance(v, (dict, list, BaseModel, SecretStr))
                else _unwrap_value(v, show_secrets=show_secrets)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_unwrap_value(item, show_secrets=show_secrets) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _model_to_dict(model: BaseModel, *, show_secrets: bool) -> dict[str, Any]:
    """Dump a Pydantic model with real SecretStr values when requested.

    ``model_dump(mode="json")`` always masks SecretStr as ``**********``;
    this walk uses ``get_secret_value()`` so ``--show-secrets`` works.
    """
    # mode="python" keeps SecretStr instances for us to unwrap.
    raw = model.model_dump(mode="python")
    return _unwrap_value(raw, show_secrets=show_secrets)  # type: ignore[return-value]


def _path_str(p: Path | None) -> str | None:
    return str(p) if p is not None else None


def _build_args(path: Path | None, config: Path | None) -> argparse.Namespace:
    """Minimal namespace so Config uses path/config discovery like CLI/MCP."""
    return argparse.Namespace(
        command=None,
        path=path,
        config=config,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print config load sources and the full resolved ChunkHound config "
            "for a project path (file/env/global discovery via path/--config; "
            "does not replay other CLI flags)."
        )
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Project directory for local .chunkhound.json discovery (default: cwd)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Explicit config file (same as chunkhound --config)",
    )
    parser.add_argument(
        "--show-secrets",
        action="store_true",
        help="Include raw api_key values (default: redact SecretStr fields)",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the resolved config JSON (no human-readable header)",
    )
    args = parser.parse_args(argv)

    target = (args.path or Path.cwd()).resolve()
    if not target.is_dir():
        print(f"error: --path is not a directory: {target}", file=sys.stderr)
        return 2

    explicit_config = args.config.resolve() if args.config is not None else None
    if explicit_config is not None and not explicit_config.is_file():
        print(f"error: --config not found: {explicit_config}", file=sys.stderr)
        return 2

    cwd = Path.cwd().resolve()

    # Import after argv parse so --help works even without a full install.
    from chunkhound.core.config.config import Config

    cli_args = _build_args(path=target, config=explicit_config)
    try:
        config = Config(args=cli_args)
    except Exception as e:
        print(f"error: failed to load Config: {e}", file=sys.stderr)
        return 1

    show_secrets = bool(args.show_secrets)
    resolved: dict[str, Any] = _model_to_dict(config, show_secrets=show_secrets)
    # Tracking fields are exclude=True on Config; re-attach for diagnostics.
    resolved["_sources"] = {
        "target_dir": _path_str(config.target_dir),
        "global_config_file": _path_str(config.global_config_file),
        "local_config_file": _path_str(config.local_config_file),
        "config_file": _path_str(config.config_file),
        "cwd": str(cwd),
        "CHUNKHOUND_CONFIG_FILE": os.getenv("CHUNKHOUND_CONFIG_FILE"),
        "CHUNKHOUND_GLOBAL_CONFIG_FILE": os.getenv("CHUNKHOUND_GLOBAL_CONFIG_FILE"),
    }

    path_cwd_mismatch = (
        config.target_dir is not None and config.target_dir.resolve() != cwd
    )
    if path_cwd_mismatch:
        print(
            "warning: target_dir != cwd. Relative paths in the config "
            "(e.g. database.path) resolve against cwd, not --path. "
            "Prefer: cd into the project (same as typical MCP cwd).",
            file=sys.stderr,
        )

    if not args.json_only:
        llm = config.llm
        print("=== Config load sources ===")
        print(f"cwd:                 {cwd}")
        print(f"target_dir:          {config.target_dir}")
        print(f"global_config_file:  {config.global_config_file}")
        print(f"local_config_file:   {config.local_config_file}")
        print(f"explicit config_file:{config.config_file}")
        print(
            f"CHUNKHOUND_CONFIG_FILE:        "
            f"{os.getenv('CHUNKHOUND_CONFIG_FILE')!r}"
        )
        print(
            f"CHUNKHOUND_GLOBAL_CONFIG_FILE: "
            f"{os.getenv('CHUNKHOUND_GLOBAL_CONFIG_FILE')!r}"
        )
        print()
        print("=== LLM summary ===")
        if llm is None:
            print("llm:                 None  (no llm section resolved)")
            print(
                "note: MCP will hide code_research/websearch; "
                "CLI research will fail at setup."
            )
        else:
            dump = _model_to_dict(llm, show_secrets=show_secrets)
            print(f"provider:            {llm.provider}")
            print(f"utility_provider:    {llm.utility_provider}")
            print(f"synthesis_provider:  {llm.synthesis_provider}")
            print(f"model:               {llm.model}")
            print(f"utility_model:       {llm.utility_model}")
            print(f"synthesis_model:     {llm.synthesis_model}")
            print(f"llm (full section):  {json.dumps(dump, indent=2, default=str)}")
            if llm.provider == "claude-code-cli":
                print(
                    "note: research will spawn bare 'claude' via PATH "
                    "in the MCP process environment."
                )
        print()
        print("=== Command validation ===")
        for cmd in ("research", "mcp", "websearch"):
            errors = config.validate_for_command(cmd)
            if errors:
                print(f"{cmd}: FAIL")
                for err in errors:
                    print(f"  - {err}")
            else:
                print(f"{cmd}: ok")
        print()
        print("=== Full resolved config ===")
        if not show_secrets:
            print("(secrets redacted; pass --show-secrets to include raw values)")
        print()

    print(json.dumps(resolved, indent=2, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
