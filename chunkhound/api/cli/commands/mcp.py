"""MCP command module - handles Model Context Protocol server operations."""

import argparse
import ipaddress
import json
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.utils.windows_constants import IS_WINDOWS


def _client_url_host(host: str) -> str:
    """Best-effort reachable host for a printed client config URL.

    ``host`` may be a bind-all address (``0.0.0.0``, ``::``) that the
    *server* binds to but that a client can never connect to directly —
    substitute the loopback form so a copy-pasted local config still works.
    IPv6 literals must be bracketed per RFC 3986 to form a valid URL.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host  # hostname, not a literal address — printed as-is
    if addr.is_unspecified:
        addr = ipaddress.ip_address("::1" if addr.version == 6 else "127.0.0.1")
    return f"[{addr}]" if addr.version == 6 else str(addr)


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
        _show_mcp_setup_instructions(args, config, force_display=True)
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

    if config.mcp.transport == "http":
        if getattr(args, "stdio", False):
            logger.warning(
                "chunkhound mcp: --stdio has no effect with --transport http; "
                "starting the HTTP transport."
            )

        # HTTP already multiplexes many concurrent client connections over one
        # port natively, so — unlike stdio — it needs no daemon/ClientProxy
        # IPC bridge to serve multiple clients from one backend process.
        from chunkhound.mcp_server.http_server import main as http_main

        await http_main(args=args, config=config)
        return

    # Daemon mode: route through ClientProxy unless explicitly disabled.
    # --stdio predates the daemon and implies single-process direct mode for
    # backwards compatibility.
    no_daemon = (
        getattr(args, "no_daemon", False)
        or getattr(args, "stdio", False)
        or os.getenv("CHUNKHOUND_DAEMON_MODE", "").lower() == "false"
    )

    if config.database.read_only:
        # Read-only implies single-process stdio: the daemon path coordinates
        # multi-client writes, which is meaningless when the process never writes.
        # Resolved from the merged config so JSON config sources work, not just CLI.
        if not no_daemon:
            logger.warning(
                "chunkhound mcp: read-only mode forces single-process stdio "
                "(daemon coordination is for writers)."
            )
        no_daemon = True

    if no_daemon:
        # Direct path: run StdioMCPServer in this process (single-client mode)
        from chunkhound.mcp_server.stdio import main

        await main(args=args)
    else:
        # Proxy path: find/start daemon, then bridge stdio ↔ socket
        from chunkhound.daemon.client_proxy import ClientProxy

        project_dir = Path(getattr(args, "path", ".")).resolve()
        proxy = ClientProxy(project_dir, args)
        await proxy.run()


def _show_mcp_setup_instructions(
    args: argparse.Namespace, config: Config | None = None, force_display: bool = False
) -> None:
    """Show comprehensive MCP setup instructions for all MCP clients.

    Args:
        args: Command arguments containing project path
        config: Pre-validated configuration instance (for transport-specific hints)
        force_display: If True, bypass all checks and show instructions
    """
    import shutil

    project_path = Path(args.path)

    # Detect installation method
    is_tool_installed = shutil.which("chunkhound") is not None

    is_http_transport = (
        config is not None and getattr(config.mcp, "transport", "stdio") == "http"
    )

    # Show setup instructions
    _safe_print("\n" + "=" * 70)
    _safe_print(" ChunkHound MCP Server - Setup Instructions")
    _safe_print("=" * 70)
    _safe_print(
        "\nHTTP Transport Mode" if is_http_transport else "\nStdio Transport Mode"
    )

    http_client_config: dict[str, Any] = {}
    if is_http_transport and config is not None:
        host = config.mcp.host
        port = config.mcp.port
        display_host = _client_url_host(host)
        _safe_print("\n" + "-" * 70)
        _safe_print(" HTTP Transport Client Configuration")
        _safe_print("-" * 70)
        http_client_config = {
            "type": "http",
            "url": f"http://{display_host}:{port}/mcp",
        }
        if config.mcp.auth_token:
            http_client_config["headers"] = {
                "Authorization": "Bearer <TOKEN>",
            }
        _safe_print("\n" + json.dumps(http_client_config, indent=2))
        if display_host != host:
            _safe_print(
                f"\n• Server is bound to {host} (all interfaces); the URL above "
                "uses the local loopback address. Replace it with the "
                "server's actual reachable hostname/IP for remote clients."
            )
        _safe_print(
            "\n• Generate a token with: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
        _safe_print(
            "• Set the same value via --auth-token on the server and in the "
            "client's Authorization header."
        )

    # The stdio launch commands below (spawning a local subprocess per client)
    # do not apply to HTTP transport, which every client instead reaches via
    # the "type": "http" block already printed above — so skip them here to
    # avoid printing configs that wouldn't actually start the right server.
    claude_code_config: dict[str, Any] = http_client_config if is_http_transport else {}
    if not is_http_transport:
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

    # Try to copy the most relevant config to clipboard (the HTTP client
    # config in HTTP mode, else the most common client's stdio config).
    try:
        import pyperclip

        pyperclip.copy(json.dumps(claude_code_config, indent=2))
        copied_what = "HTTP client" if is_http_transport else "Claude Code"
        _safe_print(f"\n✓ {copied_what} config copied to clipboard!")
    except (ImportError, Exception):
        _safe_print(
            "\n• Install pyperclip to enable clipboard copy: pip install pyperclip"
        )

    _safe_print("\n" + "=" * 70 + "\n")

    if not force_display:
        _safe_print(f"Starting MCP server for {project_path.name}...\n")


__all__: list[str] = ["mcp_command"]
