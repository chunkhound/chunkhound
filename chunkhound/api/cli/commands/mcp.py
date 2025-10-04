"""MCP command module - handles Model Context Protocol server operations."""

import argparse
import json
import os
import sys
from pathlib import Path

from chunkhound.utils.windows_constants import IS_WINDOWS


def _safe_print(text: str) -> None:
    """Print text with safe encoding for all platforms."""
    try:
        # On Windows, ensure UTF-8 encoding for console output
        if IS_WINDOWS:
            # Try to encode as UTF-8 first
            try:
                print(text.encode("utf-8").decode("utf-8"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback to ASCII-safe version
                safe_text = text.encode("ascii", errors="replace").decode("ascii")
                print(safe_text)
        else:
            # Unix systems typically handle UTF-8 better
            print(text)
    except Exception:
        # Final fallback - strip any non-ASCII characters
        safe_text = "".join(c if ord(c) < 128 else "?" for c in text)
        print(safe_text)


async def mcp_command(args: argparse.Namespace, config) -> None:
    """Execute the MCP server command.

    Args:
        args: Parsed command-line arguments containing database path
        config: Pre-validated configuration instance
    """
    # Handle --show-setup flag (display instructions and exit)
    if hasattr(args, "show_setup") and args.show_setup:
        _show_mcp_setup_instructions(args, force_display=True)
        sys.exit(0)

    # Set MCP mode environment early
    os.environ["CHUNKHOUND_MCP_MODE"] = "1"

    # CRITICAL: Import numpy modules early for DuckDB threading safety in MCP mode
    # Must happen before any DuckDB operations in async/threading context
    # See: https://duckdb.org/docs/stable/clients/python/known_issues.html
    try:
        import numpy  # noqa: F401
    except ImportError:
        pass

    # Handle transport selection
    if hasattr(args, "http") and args.http:
        # Show MCP setup instructions for HTTP mode (stdio can't print to stdout)
        _show_mcp_setup_instructions_if_first_run(args)

        # Use HTTP transport via subprocess to avoid event loop conflicts
        import subprocess

        # Use config values instead of hardcoded fallbacks
        # CLI args override config values
        host = getattr(args, "host", None) or config.mcp.host
        port = getattr(args, "port", None) or config.mcp.port

        # Run HTTP server in subprocess
        cmd = [
            sys.executable,
            "-m",
            "chunkhound.mcp.http_server",
            "--host",
            str(host),
            "--port",
            str(port),
        ]

        if hasattr(args, "db") and args.db:
            cmd.extend(["--db", str(args.db)])

        # Set up environment with UTF-8 encoding for Windows compatibility
        from chunkhound.utils.windows_constants import get_utf8_env

        env = get_utf8_env()

        process = subprocess.run(
            cmd,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            encoding="utf-8",
            errors="replace",  # Handle encoding errors gracefully
        )
        sys.exit(process.returncode)
    else:
        # Use stdio transport (default)
        from chunkhound.mcp.stdio import main

        await main(args=args)


def _show_mcp_setup_instructions_if_first_run(args: argparse.Namespace) -> None:
    """Show MCP setup instructions on first run (HTTP mode only)."""
    project_path = Path(args.path)
    config_path = project_path / ".chunkhound.json"

    # Check if this looks like a first run (recent .chunkhound.json)
    if not config_path.exists():
        return

    import time
    file_age_seconds = time.time() - config_path.stat().st_mtime
    if file_age_seconds > 300:  # More than 5 minutes old
        return

    # Only show once by creating a marker file
    marker_path = project_path / ".chunkhound" / ".mcp_setup_shown"
    if marker_path.exists():
        return

    # Show setup instructions
    _show_mcp_setup_instructions(args, force_display=False)

    # Create marker file
    marker_path.parent.mkdir(exist_ok=True)
    try:
        with open(marker_path, "w") as f:
            f.write("MCP setup instructions shown")
    except Exception:
        pass


def _show_mcp_setup_instructions(args: argparse.Namespace, force_display: bool = False) -> None:
    """Show comprehensive MCP setup instructions for all MCP clients.

    Args:
        args: Command arguments containing project path
        force_display: If True, bypass all checks and show instructions
    """
    import shutil

    project_path = Path(args.path)
    is_http = hasattr(args, "http") and args.http

    # Detect installation method
    is_tool_installed = shutil.which("chunkhound") is not None

    # Show setup instructions
    _safe_print("\n" + "=" * 70)
    _safe_print(" ChunkHound MCP Server - Setup Instructions")
    _safe_print("=" * 70)

    if is_http:
        _safe_print(f"\nHTTP Transport Mode: http://localhost:{getattr(args, 'port', 3000)}")
    else:
        _safe_print("\nStdio Transport Mode (default)")

    _safe_print("\n" + "-" * 70)
    _safe_print(" Configuration for Different MCP Clients")
    _safe_print("-" * 70)

    # Claude Code (project-local .mcp.json)
    _safe_print("\n1. Claude Code (Project-Local Configuration)")
    _safe_print("   File: .mcp.json in project root")
    _safe_print("   Scope: This project only")

    if is_tool_installed:
        claude_code_config = {
            "mcpServers": {
                "ChunkHound": {
                    "command": "chunkhound",
                    "args": ["mcp"]
                }
            }
        }
    else:
        claude_code_config = {
            "mcpServers": {
                "ChunkHound": {
                    "command": "uv",
                    "args": ["--directory", str(project_path.absolute()), "run", "chunkhound", "mcp"]
                }
            }
        }

    _safe_print("\n" + json.dumps(claude_code_config, indent=2))

    # Claude Desktop (global config)
    _safe_print("\n2. Claude Desktop (Global Configuration)")
    _safe_print("   File: ~/.claude/claude_desktop_config.json")
    _safe_print("   Scope: All projects (requires absolute path)")

    if is_tool_installed:
        desktop_config = {
            "mcpServers": {
                "chunkhound": {
                    "command": "chunkhound",
                    "args": ["mcp", str(project_path.absolute())]
                }
            }
        }
    else:
        desktop_config = {
            "mcpServers": {
                "chunkhound": {
                    "command": "uv",
                    "args": ["--directory", str(project_path.absolute()), "run", "chunkhound", "mcp", str(project_path.absolute())]
                }
            }
        }

    _safe_print("\n" + json.dumps(desktop_config, indent=2))

    # VS Code (team config)
    _safe_print("\n3. VS Code with Agent Mode (Team Configuration)")
    _safe_print("   File: .vscode/mcp.json in project")
    _safe_print("   Scope: Team/workspace")

    if is_tool_installed:
        vscode_config = {
            "servers": {
                "ChunkHound": {
                    "type": "stdio",
                    "command": "chunkhound",
                    "args": ["mcp"]
                }
            }
        }
    else:
        vscode_config = {
            "servers": {
                "ChunkHound": {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["--directory", str(project_path.absolute()), "run", "chunkhound", "mcp"]
                }
            }
        }

    _safe_print("\n" + json.dumps(vscode_config, indent=2))

    # Installation notes
    _safe_print("\n" + "-" * 70)
    _safe_print(" Notes")
    _safe_print("-" * 70)

    if is_tool_installed:
        _safe_print("\n✓ ChunkHound is installed globally (detected in PATH)")
        _safe_print("  Using: chunkhound command")
    else:
        _safe_print("\n• Running via: uv run (development mode)")
        _safe_print("  Tip: Install globally with: uv tool install chunkhound")
        _safe_print("  Then use simpler configs with just: chunkhound mcp")

    _safe_print(f"\n• Project path: {project_path.absolute()}")
    _safe_print("• Local configs (.mcp.json, .vscode/mcp.json) don't need path arg")
    _safe_print("• Global configs (~/.claude/) require absolute path")

    if is_http:
        _safe_print(f"\n• HTTP mode: Server accessible at http://localhost:{getattr(args, 'port', 3000)}")
        _safe_print("• HTTP mode can print setup instructions safely")

    # Documentation link
    _safe_print("\n" + "-" * 70)
    _safe_print(" Documentation")
    _safe_print("-" * 70)
    _safe_print("\nFor more details, visit:")
    _safe_print("https://github.com/chunkhound/chunkhound")

    # Try to copy the most common config (Claude Code) to clipboard
    try:
        import pyperclip
        pyperclip.copy(json.dumps(claude_code_config, indent=2))
        _safe_print("\n✓ Claude Code config copied to clipboard!")
    except (ImportError, Exception):
        _safe_print("\n• Install pyperclip to enable clipboard copy: pip install pyperclip")

    _safe_print("\n" + "=" * 70 + "\n")

    if not force_display:
        _safe_print(f"Starting MCP server for {project_path.name}...\n")


__all__: list[str] = ["mcp_command"]
