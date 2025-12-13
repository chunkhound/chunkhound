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

    # Use stdio transport (only supported mode)
    from chunkhound.mcp_server.stdio import main

    await main(args=args)


def _show_mcp_setup_instructions(
    args: argparse.Namespace, force_display: bool = False
) -> None:
    """Show comprehensive MCP setup instructions for all MCP clients.

    Args:
        args: Command arguments containing project path
        force_display: If True, bypass all checks and show instructions
    """
    import shutil

    project_path = Path(args.path)

    # Detect installation method
    is_tool_installed = shutil.which("chunkhound") is not None

    # Show setup instructions
    _safe_print("\n" + "=" * 70)
    _safe_print(" ChunkHound MCP Server - Setup Instructions")
    _safe_print("=" * 70)
    _safe_print("\nStdio Transport Mode")

    _safe_print("\n" + "-" * 70)
    _safe_print(" Configuration for Different MCP Clients")
    _safe_print("-" * 70)

    # Claude Code (project-local .mcp.json)
    _safe_print("\n1. Claude Code (Project-Local Configuration)")
    _safe_print("   File: .mcp.json in project root")
    _safe_print("   Scope: This project only")

    if is_tool_installed:
        claude_code_config = {
            "mcpServers": {"ChunkHound": {"command": "chunkhound", "args": ["mcp"]}}
        }
    else:
        claude_code_config = {
            "mcpServers": {
                "ChunkHound": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        str(project_path.absolute()),
                        "run",
                        "chunkhound",
                        "mcp",
                    ],
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
                    "args": ["mcp", str(project_path.absolute())],
                }
            }
        }
    else:
        desktop_config = {
            "mcpServers": {
                "chunkhound": {
                    "command": "uv",
                    "args": [
                        "--directory",
                        str(project_path.absolute()),
                        "run",
                        "chunkhound",
                        "mcp",
                        str(project_path.absolute()),
                    ],
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
                    "args": ["mcp"],
                }
            }
        }
    else:
        vscode_config = {
            "servers": {
                "ChunkHound": {
                    "type": "stdio",
                    "command": "uv",
                    "args": [
                        "--directory",
                        str(project_path.absolute()),
                        "run",
                        "chunkhound",
                        "mcp",
                    ],
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
        _safe_print(
            "\n• Install pyperclip to enable clipboard copy: pip install pyperclip"
        )

    _safe_print("\n" + "=" * 70 + "\n")

    if not force_display:
        _safe_print(f"Starting MCP server for {project_path.name}...\n")


__all__: list[str] = ["mcp_command"]
