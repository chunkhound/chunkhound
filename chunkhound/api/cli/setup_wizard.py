"""Interactive setup wizard for ChunkHound first-time configuration"""

import json
import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any

import httpx
from pydantic import SecretStr
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt

from chunkhound.api.cli.env_detector import (
    _detect_local_endpoints,
    _normalize_endpoint_url,
)
from chunkhound.api.cli.utils.rich_output import RichOutputFormatter
from chunkhound.core.config.config import Config
from chunkhound.core.config.embedding_config import EmbeddingConfig
from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory
from chunkhound.core.config.openai_utils import is_official_openai_endpoint
from chunkhound.core.constants import VOYAGE_DEFAULT_MODEL
from chunkhound.version import __version__

logger = logging.getLogger(__name__)

# Global console for consistent colored output
_console = Console()


def console_print(message: str, style: str = None) -> None:
    """Print with colors using Rich console or fallback to plain print."""
    try:
        if style:
            _console.print(f"[{style}]{message}[/{style}]")
        else:
            _console.print(message)
    except Exception:
        # Fallback to plain print if Rich fails
        print(message)


# Browser opening helper
def _open_url_on_empty_input(
    url: str, provider_name: str, formatter: RichOutputFormatter
) -> bool:
    """Open URL in browser when user provides empty input.

    Args:
        url: URL to open
        provider_name: Provider name for user messages
        formatter: Rich formatter for messages

    Returns:
        True if URL was opened or attempted, False otherwise
    """
    try:
        webbrowser.open(url)
        formatter.info(f"Opening {provider_name} page in your browser...")
        return True
    except Exception:
        formatter.warning(f"Could not open browser. Please visit {url} manually.")
        return True  # Still count as attempted


# Input pre-fill helper for fallback mode
def _input_with_prefill(prompt: str, prefill_text: str = "") -> str:
    """Input with pre-filled text using readline (if available).

    Args:
        prompt: The input prompt to display
        prefill_text: Text to pre-fill in the input buffer

    Returns:
        User input string
    """
    try:
        import readline

        def startup_hook():
            readline.insert_text(prefill_text)

        readline.set_startup_hook(startup_hook)
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook(None)  # Critical: remove hook
    except ImportError:
        # readline not available (e.g., Windows), fall back to regular input
        return input(prompt)


# Validation helper functions
def _validate_api_key_format(key: str, prefix: str) -> bool | str:
    """Validate API key format.

    Args:
        key: API key to validate
        prefix: Expected prefix (e.g., 'sk-', 'pa-')

    Returns:
        True if valid, error message string if invalid
    """
    if key.strip().startswith(prefix):
        return True
    return f"API keys for this provider start with '{prefix}'"


def _validate_url_format(url: str) -> bool | str:
    """Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if valid, error message string if invalid
    """
    if url.strip().startswith(("http://", "https://")):
        return True
    return "URL must start with http:// or https://"


def _validate_non_empty(text: str, field_name: str = "Field") -> bool | str:
    """Validate that text is not empty.

    Args:
        text: Text to validate
        field_name: Name of field for error message

    Returns:
        True if valid, error message string if invalid
    """
    if text.strip():
        return True
    return f"{field_name} cannot be empty"


# Simplified Rich-based utility functions
async def rich_confirm(question: str, default: bool = True) -> bool:
    """Interactive Rich-based confirmation with arrow key navigation."""
    return await _rich_confirm_interactive(question, default)


async def _rich_confirm_interactive(question: str, default: bool = True) -> bool:
    """Interactive yes/no confirmation with arrow key navigation."""
    from rich.live import Live
    from rich.text import Text

    from .keyboard import KeyboardInput

    console = _console
    # Default to Yes (True) if default is True, No (False) if default is False
    selected = 0 if default else 1  # 0 = Yes, 1 = No
    keyboard_handler = KeyboardInput()

    # Show the question
    console.print(f"\n[bold]{question}[/bold]")
    console.print()

    def create_display():
        """Create the display with current selection highlighted."""
        text = Text()

        # Yes option
        if selected == 0:
            text.append("▶ Yes  ", style="bold cyan")
        else:
            text.append("  Yes  ", style="dim")

        # No option
        if selected == 1:
            text.append("▶ No", style="bold cyan")
        else:
            text.append("  No", style="dim")

        return text

    with Live(create_display(), auto_refresh=False, console=console) as live:
        # Main input loop
        while True:
            try:
                key = keyboard_handler.getkey()

                if key in ["LEFT", "UP"]:
                    selected = 0  # Select Yes
                    live.update(create_display(), refresh=True)
                elif key in ["RIGHT", "DOWN"]:
                    selected = 1  # Select No
                    live.update(create_display(), refresh=True)
                elif key.lower() == "y":  # 'y' shortcut for Yes
                    live.stop()
                    console.print("Selected: Yes")
                    console.print()  # Add spacing
                    return True
                elif key.lower() == "n":  # 'n' shortcut for No
                    live.stop()
                    console.print("Selected: No")
                    console.print()  # Add spacing
                    return False
                elif key == "ENTER":
                    live.stop()
                    result = selected == 0  # True for Yes, False for No
                    choice_text = "Yes" if result else "No"
                    console.print(f"Selected: {choice_text}")
                    console.print()  # Add spacing
                    return result
                elif key == "ESC":
                    live.stop()
                    raise KeyboardInterrupt("Confirmation cancelled")
                elif key == "CTRL_C":
                    live.stop()
                    raise KeyboardInterrupt("Confirmation cancelled")
                # Ignore other keys

            except KeyboardInterrupt:
                live.stop()
                raise


async def _rich_text_interactive(
    question: str, default: str = "", validate=None, password: bool = False
) -> str:
    """Interactive text input with arrow key support."""
    from .keyboard import KeyboardInput
    from .utils.text_input import TextInputState, create_text_input_display

    console = _console
    keyboard_handler = KeyboardInput()
    console.show_cursor(False)

    # Initialize state
    state = TextInputState(default)

    def create_display():
        """Create the display with current state."""
        return create_text_input_display(question, state, password)

    def handle_validation(text: str) -> bool:
        """Handle validation and update state. Returns True if valid."""
        if not validate:
            return True

        try:
            validation_result = validate(text)
            if validation_result is True:
                return True
            elif isinstance(validation_result, str):
                state.validation_error = validation_result
                return False
            else:
                state.validation_error = "Invalid input"
                return False
        except Exception as e:
            state.validation_error = f"Validation error: {e}"
            return False

    try:
        with Live(create_display(), auto_refresh=False, console=console) as live:
            while True:
                try:
                    key = keyboard_handler.getkey()

                    if key in ["LEFT", "RIGHT", "HOME", "END"]:
                        state.move_cursor(key)
                        live.update(create_display(), refresh=True)
                    elif key in ["BACKSPACE", "DELETE"]:
                        state.delete_char(key)
                        live.update(create_display(), refresh=True)
                    elif key == "ENTER":
                        if handle_validation(state.text):
                            console.print()  # Add spacing
                            return state.text
                        else:
                            live.update(create_display(), refresh=True)
                    elif key in ["ESC", "CTRL_C"]:
                        raise KeyboardInterrupt("Text input cancelled")
                    elif len(key) == 1 and key.isprintable():
                        state.insert_char(key)
                        live.update(create_display(), refresh=True)
                    # Ignore other keys

                except KeyboardInterrupt:
                    raise
    finally:
        console.show_cursor(True)


async def rich_text(question: str, default: str = "", validate=None) -> str:
    """Simple wrapper for Rich text prompt with validation and arrow key support."""
    try:
        # Use our custom interactive text input
        return await _rich_text_interactive(question, default, validate)
    except Exception as e:
        # Log the actual error for debugging
        logger.debug(f"Rich mode failed with {type(e).__name__}: {e}")

        # Show debug info if requested
        if os.getenv("CHUNKHOUND_DEBUG"):
            console_print(
                f"Debug: Rich mode failed - {type(e).__name__}: {e}", "yellow"
            )

        # Fallback to standard input with readline pre-fill when possible
        console_print("Using standard input mode...", "dim")

        def handle_fallback_input():
            """Handle input in fallback mode with proper pre-fill support."""
            # Check if this looks like a sensitive value (API key, token, etc.)
            is_sensitive = default and (
                len(default) > 20
                or any(default.startswith(p) for p in ["sk-", "pa-", "xai-", "key-"])
            )

            if default and is_sensitive:
                # For sensitive values, try readline pre-fill first
                try:
                    return _input_with_prefill(f"{question}: ", default)
                except ImportError:
                    # readline not available (Windows), use clear messaging
                    console_print("API key detected from environment variable", "dim")
                    result = input(
                        f"{question} (press Enter to use detected key, "
                        "or enter new key): "
                    )
                    return result if result.strip() else default
            elif default:
                # For non-sensitive defaults, use Rich's normal behavior
                return Prompt.ask(question, default=default)
            else:
                # No default, use regular input
                return input(f"{question}: ")

        if validate is None:
            return handle_fallback_input()

        # Handle validation in fallback mode
        while True:
            result = handle_fallback_input()
            if not validate:
                return result

            validation_result = validate(result)
            if validation_result is True:
                return result
            elif isinstance(validation_result, str):
                console_print(f"Error: {validation_result}", "red")
            else:
                console_print("Invalid input", "red")


async def rich_select(
    question: str, choices: list[tuple[str, str]] | list[str], default: str = None
) -> str:
    """Interactive Rich-based selection with arrow key navigation."""
    if not choices:
        return ""

    # Normalize choices to (display, value) tuples
    normalized_choices = []
    choice_values = []

    for choice in choices:
        if isinstance(choice, tuple):
            display, value = choice
            normalized_choices.append((display, value))
            choice_values.append(value)
        else:
            normalized_choices.append((choice, choice))
            choice_values.append(choice)

    # If only one choice, return it
    if len(normalized_choices) == 1:
        return choice_values[0]

    # Find default index
    default_index = 0
    if default and default in choice_values:
        default_index = choice_values.index(default)

    return await _rich_select_interactive(question, normalized_choices, default_index)


async def _rich_select_interactive(
    question: str, choices: list[tuple[str, str]], default_index: int = 0
) -> str:
    """Interactive menu with arrow key navigation using asyncio + Rich."""

    from .keyboard import KeyboardInput

    console = _console
    selected = max(0, min(default_index, len(choices) - 1))
    keyboard_handler = KeyboardInput()

    # Show the question
    console.print(f"\n[bold]{question}[/bold]")
    console.print()

    def create_display():
        """Create the display text with current selection."""
        from rich.text import Text

        text = Text()
        for i, (display, _) in enumerate(choices):
            if i == selected:
                text.append("▶ ", style="bold cyan")
                text.append(display, style="bold cyan")
            else:
                text.append("  ")
                text.append(display)
            if i < len(choices) - 1:  # Add newline except for last item
                text.append("\n")
        return text

    with Live(create_display(), auto_refresh=False, console=console) as live:
        # Main input loop
        while True:
            try:
                # Get key input
                key = keyboard_handler.getkey()

                if key == "UP":
                    selected = max(0, selected - 1)
                    live.update(create_display(), refresh=True)
                elif key == "DOWN":
                    selected = min(len(choices) - 1, selected + 1)
                    live.update(create_display(), refresh=True)
                elif key == "ENTER":
                    live.stop()
                    console.print()  # Add spacing
                    return choices[selected][1]
                elif key == "ESC":
                    # With reliable readchar, ESC should work correctly
                    live.stop()
                    raise KeyboardInterrupt("Selection cancelled")
                elif key == "CTRL_C":
                    live.stop()
                    raise KeyboardInterrupt("Selection cancelled")
                elif key.isdigit():
                    # Allow numeric selection as alternative to arrow keys
                    digit = int(key)
                    if 1 <= digit <= len(choices):
                        selected = digit - 1
                        live.stop()
                        console.print()
                        console.print(f"Selected: {choices[selected][0]}")
                        return choices[selected][1]
                # Ignore other keys

            except KeyboardInterrupt:
                live.stop()
                raise


async def _fetch_available_models(
    base_url: str, api_key: str | None = None
) -> tuple[list[str] | None, bool]:
    """Fetch available models from OpenAI-compatible endpoint.

    Args:
        base_url: Base URL of the OpenAI-compatible endpoint
        api_key: Optional API key for authentication

    Returns:
        Tuple of (models_list, needs_auth)
        - models_list: List of model names if successful, None if failed
        - needs_auth: True if failure appears to be authentication-related
    """
    try:
        # Normalize URL - ensure it doesn't end with /v1 if present
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        models_url = f"{url}/v1/models"

        # Prepare headers
        headers = {"Accept": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"

        # Configure HTTP client with SSL handling (same pattern as OpenAI provider)
        client_kwargs = {"timeout": 10.0}  # Increased timeout for corporate networks

        # Apply SSL verification logic - reuse existing pattern
        is_openai_official = is_official_openai_endpoint(base_url)
        if not is_openai_official:
            # For custom endpoints, disable SSL verification
            # These often use self-signed certificates (corporate servers, Ollama)
            client_kwargs["verify"] = False
            logger.debug(f"SSL verification disabled for custom endpoint: {base_url}")

        # Make request with SSL configuration
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.get(models_url, headers=headers)
            response.raise_for_status()

            data = response.json()
            if "data" in data and isinstance(data["data"], list):
                # Extract model names from OpenAI-compatible response format
                models = []
                for model_info in data["data"]:
                    if isinstance(model_info, dict) and "id" in model_info:
                        models.append(model_info["id"])
                    elif isinstance(model_info, str):
                        models.append(model_info)

                return (sorted(models) if models else None, False)

            return (None, False)

    except httpx.HTTPStatusError as e:
        # Check if it's an authentication error
        if e.response.status_code in [401, 403]:
            return (None, True)  # Definitely needs authentication
        elif e.response.status_code in [200, 404]:
            return (None, False)  # Clear no-auth cases (success or not found)
        else:
            # Other HTTP errors (500, etc.) - assume auth needed to be safe
            return (None, True)
    except Exception:
        # Network or parsing error - assume auth needed to be safe
        return (None, True)


def _filter_embedding_models(models: list[str]) -> tuple[list[str], list[str]]:
    """Filter models to identify likely embedding models.

    Args:
        models: List of all available model names

    Returns:
        Tuple of (embedding_models, other_models)
    """
    embedding_keywords = [
        "embed",
        "embedding",
        "sentence",
        "text-embed",
        "nomic-embed",
        "mxbai-embed",
        "bge-",
        "all-minilm",
        "e5-",
        "multilingual-e5",
        "gte-",
    ]

    # Known non-embedding model patterns
    non_embedding_keywords = [
        "gpt",
        "llama",
        "mistral",
        "phi",
        "codellama",
        "vicuna",
        "chat",
        "instruct",
        "code",
        "qwen",
        "gemma",
        "solar",
    ]

    embedding_models = []
    other_models = []

    for model in models:
        model_lower = model.lower()

        # Check if it's likely an embedding model
        is_embedding = any(keyword in model_lower for keyword in embedding_keywords)
        is_non_embedding = any(
            keyword in model_lower for keyword in non_embedding_keywords
        )

        if is_embedding and not is_non_embedding:
            embedding_models.append(model)
        else:
            other_models.append(model)

    return embedding_models, other_models


def _filter_reranking_models(models: list[str]) -> list[str]:
    """Filter models to identify likely reranking models.

    Args:
        models: List of all available model names

    Returns:
        List of reranking models
    """
    reranking_keywords = [
        "rerank",
        "reranker",
        "cross-encoder",
        "cross_encoder",
        "bge-reranker",
        "jina-reranker",
        "mxbai-rerank",
        "ce-esci",  # FlashRank models
    ]

    reranking_models = []

    for model in models:
        model_lower = model.lower()

        # Check if it's likely a reranking model
        if any(keyword in model_lower for keyword in reranking_keywords):
            reranking_models.append(model)

    return reranking_models


def _detect_opencode() -> bool:
    """Check if opencode CLI is installed"""
    try:
        result = subprocess.run(
            ["opencode", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _should_run_setup_wizard(validation_errors: list[str]) -> bool:
    """Check if we should offer interactive setup wizard"""
    # Only run in interactive terminal
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False

    # Check if missing embedding configuration
    for error in validation_errors:
        if any(
            keyword in error.lower()
            for keyword in [
                "embedding provider",
                "api key",
                "provider not configured",
                "no embedding provider configured",
            ]
        ):
            return True

    return False


def _display_detected_configs(
    configs: dict[str, dict[str, Any] | None], formatter: RichOutputFormatter
) -> None:
    """Display detected configurations using Rich components."""
    import rich.box
    from rich.table import Table

    table = Table(show_header=False, box=rich.box.ROUNDED, padding=(0, 1))
    table.add_column("Configuration", style="cyan")

    for provider, config in configs.items():
        if not config:
            continue

        if provider == "voyageai":
            table.add_row("• VoyageAI API key found (VOYAGE_API_KEY)")

        elif provider == "openai":
            table.add_row("• OpenAI API key found (OPENAI_API_KEY)")
            if config.get("base_url"):
                url = config["base_url"]
                if any(host in url.lower() for host in ["localhost", "127.0.0.1"]):
                    table.add_row(f"    Local endpoint: {url}")
                else:
                    table.add_row(f"    Custom endpoint: {url}")
            if config.get("organization"):
                table.add_row(f"    Organization: {config['organization']}")

        elif provider == "local":
            detection_source = config.get("detected_from", "scan")
            provider_name = config.get("provider_name", "Unknown")
            if detection_source == "environment":
                table.add_row(f"• {provider_name} configured via environment")
            else:
                table.add_row(
                        f"• {provider_name} server detected at "
                        f"{config.get('base_url', 'unknown')}"
                )

    if formatter.console is not None:
        formatter.console.print(table)
    else:
        print("Environment Configuration Detected")
        for provider, config in configs.items():
            if config:
                if provider == "voyageai":
                    print("• VoyageAI API key found (VOYAGE_API_KEY)")
                elif provider == "openai":
                    print("• OpenAI API key found (OPENAI_API_KEY)")
                elif provider == "local":
                    provider_name = config.get("provider_name", "Unknown")
                    print(
                    f"• {provider_name} server detected at "
                    f"{config.get('base_url', 'unknown')}"
                    )


async def run_setup_wizard(target_path: Path, args=None) -> Config | None:
    """
    Run the interactive setup wizard to create initial configuration.

    Args:
        target_path: Directory where .chunkhound.json will be created

    Returns:
        Config object if setup completed successfully, None if cancelled
    """
    formatter = RichOutputFormatter()

    # Display welcome screen
    _display_welcome(formatter, target_path)

    # Continue with normal provider selection if no env config or user declined
    provider_choice = await _select_provider()
    if provider_choice == "skip":
        formatter.info(
            "Skipping provider setup. You can configure later in .chunkhound.json"
        )
        return None

    # Configure selected provider
    embedding_config = None
    if provider_choice == "voyageai":
        embedding_config = await _configure_voyageai(formatter)
    elif provider_choice == "openai":
        embedding_config = await _configure_openai(formatter)
    elif provider_choice == "openai_compatible":
        embedding_config = await _configure_openai_compatible(formatter)

    if not embedding_config:
        formatter.warning(
            "Setup cancelled. You can configure later using .chunkhound.json"
        )
        return None

    # Save configuration
    config_path, status = await _save_configuration(
        embedding_config, target_path, formatter
    )
    if status == "saved":
        formatter.success(f"Configuration saved to {config_path}")

        # Run agent setup
        await _run_agent_setup(target_path, formatter)

        print("\nReady to start indexing your codebase!")
        # Return a new config object that will pick up the saved file
        return Config(args=args) if args else Config()
    elif status == "cancelled":
        formatter.info("Configuration not saved")
        return None
    else:  # status == "error"
        return None


def _display_welcome(formatter: RichOutputFormatter, target_path: Path) -> None:
    """Display welcome message with setup information"""
    from rich.panel import Panel
    from rich.table import Table

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column()

    info_table.add_row("Version:", f"[green]{__version__}[/green]")
    info_table.add_row("Directory:", f"[blue]{target_path}[/blue]")

    panel = Panel(
        info_table,
        title="[bold cyan]ChunkHound Setup Wizard[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )
    if formatter.console is not None:
        formatter.console.print(panel)
        formatter.console.print(
            "Configure embedding provider and index your project for semantic search."
        )
    else:
        print("ChunkHound Setup Wizard")
        print(f"Version: {__version__}")
        print(f"Directory: {target_path}")
        print(
            "Configure embedding provider and index your project for semantic search."
        )


async def _select_provider() -> str:
    """Interactive provider selection"""
    choices = [
        ("VoyageAI (Recommended - Best for code)", "voyageai"),
        ("OpenAI", "openai"),
        ("OpenAI-compatible (Ollama, LM Studio, etc.)", "openai_compatible"),
    ]

    return await rich_select(
        "Select your embedding provider:", choices=choices, default="voyageai"
    )


async def _select_agent() -> str:
    """Interactive agent selection for MCP setup"""
    choices = [
        ("Claude Code - Official Anthropic CLI", "claude_code"),
        ("VS Code - Microsoft Visual Studio Code", "vscode"),
        ("OpenCode - Open source coding agent", "opencode"),
        ("Skip - Configure manually later", "skip"),
    ]

    return await rich_select(
        "Which AI agent would you like to configure for ChunkHound?",
        choices=choices,
        default="claude_code",
    )


async def _setup_claude_code(target_path: Path, formatter: RichOutputFormatter) -> bool:
    """Setup ChunkHound MCP integration with Claude Code"""
    formatter.section_header("Claude Code Setup")

    # Path to .mcp.json file in project root
    mcp_path = target_path / ".mcp.json"

    # Read existing configuration
    config = _read_claude_mcp_config(mcp_path)

    # Check if ChunkHound is already configured
    servers = config.setdefault("mcpServers", {})
    if "ChunkHound" in servers:
        formatter.warning("ChunkHound MCP server already configured in .mcp.json")

        overwrite = await rich_confirm(
            "Overwrite existing ChunkHound configuration?", default=False
        )
        if not overwrite:
            formatter.info("Skipping Claude Code configuration")
            return True

    # Add ChunkHound server configuration
    servers["ChunkHound"] = {"command": "chunkhound", "args": ["mcp"], "env": {}}

    # Write updated configuration
    if _write_claude_mcp_config(mcp_path, config):
        formatter.success(f"✓ ChunkHound MCP server added to {mcp_path}")
        print("You can now use ChunkHound tools directly in Claude Code!")
        print(
            "Claude Code will prompt you to approve this project-scoped "
            "server on first use."
        )
        return True
    else:
        formatter.error(f"Failed to write configuration to {mcp_path}")
        _show_manual_claude_instructions(formatter, mcp_path)
        return False


def _show_claude_installation_instructions(formatter: RichOutputFormatter) -> None:
    """Show instructions for installing Claude Code"""
    print("\nTo install Claude Code:")
    formatter.bullet_list(
        [
            "Visit: https://claude.ai/download",
            "Download and install Claude Code",
            "Create .mcp.json file in your project root with ChunkHound configuration",
        ]
    )


def _show_manual_claude_instructions(
    formatter: RichOutputFormatter, mcp_path: Path | None = None
) -> None:
    """Show manual configuration instructions for Claude Code"""
    print("\nTo manually configure ChunkHound in Claude Code:")

    if mcp_path:
        formatter.bullet_list(
            [
                f"Edit or create: {mcp_path}",
                "Add ChunkHound server configuration:",
                '  { "mcpServers": { "ChunkHound": { "command": "chunkhound", "args": ["mcp"], "env": {} } } }',
            ]
        )
    else:
        formatter.bullet_list(
            [
                "Create .mcp.json in your project root",
                'Add: { "mcpServers": { "ChunkHound": { "command": "chunkhound", "args": ["mcp"], "env": {} } } }',
                "Claude Code will prompt to approve the server on first use",
            ]
        )


def _detect_vscode_workspace(target_path: Path) -> Path | None:
    """Detect VS Code workspace by looking for .vscode directory"""
    # Check current directory first
    current_vscode = target_path / ".vscode"
    if current_vscode.exists() and current_vscode.is_dir():
        return current_vscode

    # Check parent directories up to 3 levels
    for parent in [
        target_path.parent,
        target_path.parent.parent,
        target_path.parent.parent.parent,
    ]:
        if parent == target_path:  # Avoid infinite loop
            break
        vscode_dir = parent / ".vscode"
        if vscode_dir.exists() and vscode_dir.is_dir():
            return vscode_dir

    return None


def _read_vscode_mcp_config(mcp_path: Path) -> dict[str, Any]:
    """Read existing VS Code MCP configuration or return empty structure"""
    if mcp_path.exists():
        try:
            with open(mcp_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If file is corrupted or unreadable, start fresh
            pass

    return {"servers": {}}


def _write_vscode_mcp_config(mcp_path: Path, config: dict[str, Any]) -> bool:
    """Write VS Code MCP configuration, creating directory if needed"""
    try:
        mcp_path.parent.mkdir(exist_ok=True)
        with open(mcp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except OSError:
        return False


def _read_claude_mcp_config(mcp_path: Path) -> dict[str, Any]:
    """Read existing Claude Code .mcp.json configuration or return empty structure"""
    if mcp_path.exists():
        try:
            with open(mcp_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If file is corrupted or unreadable, start fresh
            pass

    return {"mcpServers": {}}


def _write_claude_mcp_config(mcp_path: Path, config: dict[str, Any]) -> bool:
    """Write Claude Code .mcp.json configuration"""
    try:
        with open(mcp_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except OSError:
        return False


def _read_opencode_config(opencode_path: Path) -> dict[str, Any]:
    """Read existing opencode.json or return empty structure"""
    if opencode_path.exists():
        try:
            with open(opencode_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If file is corrupted or unreadable, start fresh
            pass

    return {}  # Return empty dict, not {"mcp": {}} since file might not exist


def _write_opencode_config(opencode_path: Path, config: dict[str, Any]) -> bool:
    """Write opencode.json configuration"""
    try:
        with open(opencode_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except OSError:
        return False


async def _setup_vscode(target_path: Path, formatter: RichOutputFormatter) -> bool:
    """Setup ChunkHound MCP integration with VS Code"""
    formatter.section_header("VS Code Setup")

    # Detect workspace
    vscode_dir = _detect_vscode_workspace(target_path)

    if vscode_dir:
        formatter.info(f"✓ VS Code workspace detected: {vscode_dir.parent}")
        mcp_path = vscode_dir / "mcp.json"

        # Read existing configuration
        config = _read_vscode_mcp_config(mcp_path)

        # Check if ChunkHound is already configured
        servers = config.setdefault("servers", {})
        if "ChunkHound" in servers:
            formatter.warning("ChunkHound MCP server already configured in VS Code")

            overwrite = await rich_confirm(
                "Overwrite existing ChunkHound configuration?", default=False
            )
            if not overwrite:
                formatter.info("Skipping VS Code configuration")
                return True

        # Add ChunkHound server configuration
        servers["ChunkHound"] = {"command": "chunkhound", "args": ["mcp", "stdio"]}

        # Write updated configuration
        if _write_vscode_mcp_config(mcp_path, config):
            formatter.success(f"✓ ChunkHound MCP server added to {mcp_path}")
            print("You can now use ChunkHound tools in VS Code with GitHub Copilot!")
            return True
        else:
            formatter.error(f"Failed to write configuration to {mcp_path}")
            _show_manual_vscode_instructions(formatter, mcp_path)
            return False
    else:
        formatter.warning("No VS Code workspace detected")

        # Offer to create workspace configuration
        create_workspace = await rich_confirm(
            "Create .vscode/mcp.json in target directory?", default=True
        )

        if create_workspace:
            vscode_dir = target_path / ".vscode"
            mcp_path = vscode_dir / "mcp.json"

            config = {
                "servers": {
                    "ChunkHound": {"command": "chunkhound", "args": ["mcp", "stdio"]}
                }
            }

            if _write_vscode_mcp_config(mcp_path, config):
                formatter.success(
                    f"✓ Created VS Code workspace with ChunkHound MCP: {mcp_path}"
                )
                print(
                    "You can now use ChunkHound tools in VS Code with GitHub Copilot!"
                )
                return True
            else:
                formatter.error(f"Failed to create {mcp_path}")
                _show_manual_vscode_instructions(formatter, mcp_path)
                return False
        else:
            _show_manual_vscode_instructions(formatter)
            return False


def _show_manual_vscode_instructions(
    formatter: RichOutputFormatter, mcp_path: Path | None = None
) -> None:
    """Show manual configuration instructions for VS Code"""
    print("\nTo manually configure ChunkHound in VS Code:")

    if mcp_path:
        formatter.bullet_list(
            [
                f"Edit: {mcp_path}",
                "Add ChunkHound server configuration:",
                '  "ChunkHound": { "command": "chunkhound", "args": ["mcp", "stdio"] }',
            ]
        )
    else:
        formatter.bullet_list(
            [
                "Create .vscode/mcp.json in your workspace",
                'Add configuration: { "servers": { "ChunkHound": { "command": "chunkhound", "args": ["mcp", "stdio"] } } }',
                "Restart VS Code to load the MCP server",
            ]
        )


def _show_manual_opencode_instructions(
    formatter: RichOutputFormatter, opencode_path: Path | None = None
) -> None:
    """Show manual configuration instructions for OpenCode"""
    print("\nTo manually configure ChunkHound in OpenCode:")

    if opencode_path:
        formatter.bullet_list(
            [
                f"Edit or create: {opencode_path}",
                "Add/merge ChunkHound configuration:",
                '  { "mcp": { "chunkhound": { "type": "local", "command": ["chunkhound", "mcp"] } }, "$schema": "https://opencode.ai/config.json" }',
            ]
        )
    else:
        formatter.bullet_list(
            [
                "Create opencode.json in your project root",
                'Add: { "mcp": { "chunkhound": { "type": "local", "command": ["chunkhound", "mcp"] } }, "$schema": "https://opencode.ai/config.json" }',
            ]
        )


async def _setup_opencode(target_path: Path, formatter: RichOutputFormatter) -> bool:
    """Setup ChunkHound MCP integration with OpenCode"""
    formatter.section_header("OpenCode Setup")

    # Check if OpenCode CLI is installed
    if not _detect_opencode():
        formatter.warning("OpenCode CLI not found")
        print("\nTo install OpenCode:")
        formatter.bullet_list(
            [
                "Visit: https://opencode.ai",
                "Download and install OpenCode CLI",
                "Run: opencode --version to verify installation",
            ]
        )

        # Offer to create configuration anyway
        create_config = await rich_confirm(
            "Create OpenCode configuration anyway?", default=False
        )
        if not create_config:
            return False

    # Path to opencode.json file in project root
    opencode_path = target_path / "opencode.json"

    # Read existing configuration
    config = _read_opencode_config(opencode_path)

    # Check if ChunkHound is already configured
    mcp_config = config.setdefault("mcp", {})
    if "chunkhound" in mcp_config:
        formatter.warning("ChunkHound MCP server already configured in opencode.json")

        overwrite = await rich_confirm(
            "Overwrite existing ChunkHound configuration?", default=False
        )
        if not overwrite:
            formatter.info("Skipping OpenCode configuration")
            return True

    # Add ChunkHound server configuration
    mcp_config["chunkhound"] = {
        "type": "local",
        "command": ["chunkhound", "mcp"]
    }

    # Ensure schema is present
    if "$schema" not in config:
        config["$schema"] = "https://opencode.ai/config.json"

    # Write updated configuration
    if _write_opencode_config(opencode_path, config):
        formatter.success(f"✓ ChunkHound MCP server added to {opencode_path}")
        print("You can now use ChunkHound tools directly in OpenCode!")
        print("OpenCode will automatically detect and load the MCP server.")
        return True
    else:
        formatter.error(f"Failed to write configuration to {opencode_path}")
        _show_manual_opencode_instructions(formatter, opencode_path)
        return False


async def _run_agent_setup(target_path: Path, formatter: RichOutputFormatter) -> None:
    """Run the agent setup step after successful configuration"""
    formatter.section_header("AI Agent Integration")
    print("Connect ChunkHound to your preferred AI agent for seamless code search.\n")

    agent_choice = await _select_agent()

    if agent_choice == "skip":
        formatter.info("Skipped agent setup. You can configure this manually later.")
        print("\nFor manual setup instructions, run: chunkhound mcp --help")
        return

    success = False
    if agent_choice == "claude_code":
        success = await _setup_claude_code(target_path, formatter)
    elif agent_choice == "vscode":
        success = await _setup_vscode(target_path, formatter)
    elif agent_choice == "opencode":
        success = await _setup_opencode(target_path, formatter)

    if not success:
        formatter.warning(
            "Agent setup incomplete. ChunkHound will still work via command line."
        )
        print("Run 'chunkhound mcp' to test the MCP server manually.")


async def _configure_voyageai(formatter: RichOutputFormatter) -> dict[str, Any] | None:
    """Configure VoyageAI provider with signup assistance"""
    print("Excellent choice! VoyageAI offers specialized code embeddings.\n")

    # Check for existing API key
    from .env_detector import _detect_voyageai

    detected_config = _detect_voyageai()
    api_key = None
    already_declined = False

    if detected_config and detected_config.get("api_key"):
        use_detected = await rich_confirm(
            "Found VoyageAI API key in environment. Use it?", default=True
        )
        if use_detected:
            api_key = detected_config["api_key"]
        else:
            already_declined = True

    return await _configure_provider_unified(
        "voyageai",
        api_key=api_key,
        formatter=formatter,
        already_declined_key=already_declined,
    )


async def _configure_openai(formatter: RichOutputFormatter) -> dict[str, Any] | None:
    """Configure OpenAI provider"""

    # Check for existing API key
    from .env_detector import _detect_openai

    detected_config = _detect_openai()
    api_key = None
    already_declined = False

    if detected_config and detected_config.get("api_key"):
        use_detected = await rich_confirm(
            "Found OpenAI API key in environment. Use it?", default=True
        )
        if use_detected:
            api_key = detected_config["api_key"]
        else:
            already_declined = True

    return await _configure_provider_unified(
        "openai",
        api_key=api_key,
        formatter=formatter,
        already_declined_key=already_declined,
    )


async def _configure_openai_compatible(
    formatter: RichOutputFormatter,
) -> dict[str, Any] | None:
    """Configure OpenAI-compatible endpoint"""
    formatter.section_header("OpenAI-Compatible Configuration")

    # Check for running local servers first
    local_endpoint = _detect_local_endpoints()

    if local_endpoint:
        provider_name = local_endpoint["provider_name"]
        detected_url = local_endpoint["base_url"]

        print(f"🎯 Found {provider_name} running locally!")
        print("   Benefits of local servers:")
        formatter.bullet_list(
            [
                "Complete privacy - your code never leaves your machine",
                "No API costs or rate limits",
                "Works offline",
                "Full control over models and performance",
            ]
        )
        print()

        use_detected = await rich_confirm(
            f"Use {provider_name} at {detected_url}?", default=True
        )

        if use_detected:
            return await _configure_provider_unified(
                "openai_compatible", base_url=detected_url, formatter=formatter
            )

    # If no local server detected or user declined, show manual entry
    print("Common OpenAI-compatible providers:")
    formatter.bullet_list(
        [
            "Ollama: http://localhost:11434/v1",
            "LM Studio: http://localhost:1234/v1",
            "vLLM: http://localhost:8000/v1",
            "Any OpenAI-compatible API endpoint",
        ]
    )
    print()

    # Only provide a default URL if no local server was detected
    # If user declined a detected server, leave field empty
    default_url = ""

    base_url = await rich_text(
        "Endpoint URL:",
        default=default_url,
        validate=_validate_url_format,
    )

    normalized_url = _normalize_endpoint_url(base_url.strip())

    # Check for existing API key in environment for the configured URL
    from .env_detector import _detect_openai

    detected_config = _detect_openai()
    detected_api_key = None
    already_declined_key = False

    if detected_config and detected_config.get("api_key"):
        # Check if the detected config has a compatible base URL or no base URL
        detected_base_url = detected_config.get("base_url")
        if not detected_base_url or detected_base_url == normalized_url:
            use_detected = await rich_confirm(
                "Found API key in environment. Use it?", default=True
            )
            if use_detected:
                detected_api_key = detected_config["api_key"]
            else:
                already_declined_key = True

    return await _configure_provider_unified(
        "openai_compatible",
        base_url=normalized_url,
        api_key=detected_api_key,
        formatter=formatter,
        already_declined_key=already_declined_key,
    )


async def _select_compatible_model(
    base_url: str, api_key: str | None, formatter: RichOutputFormatter
) -> tuple[str | None, str | None]:
    """Select a model from available models or manual entry.

    Args:
        base_url: Endpoint URL
        api_key: Optional API key
        formatter: Output formatter

    Returns:
        Tuple of (selected_model, api_key_used)
    """
    # Try to fetch available models
    formatter.safe_progress_indicator("Detecting available models...")
    available_models, needs_auth = await _fetch_available_models(base_url, api_key)

    # If failed and might need auth, try with API key
    current_api_key = api_key
    if available_models is None and needs_auth and api_key is None:
        formatter.warning("Authentication may be required for this endpoint")
        retry_key = await rich_text("API Key (press Enter to skip):", default="")

        if retry_key.strip():
            formatter.safe_progress_indicator("Retrying with authentication...")
            available_models, _ = await _fetch_available_models(
                base_url, retry_key.strip()
            )
            if available_models:
                current_api_key = retry_key.strip()

    if available_models:
        # Filter for embedding models
        embedding_models, other_models = _filter_embedding_models(available_models)

        if embedding_models:
            formatter.success(f"Found {len(embedding_models)} embedding models")

            # Create choices for model selection (skip separators)
            choices = []
            for model in embedding_models:
                choices.append((f"{model} (embedding)", model))

            # Add other models if any (limit to first 10)
            if other_models:
                for model in other_models[:10]:
                    choices.append((f"{model} (other)", model))

            # Add manual entry option
            choices.append(("Enter manually...", "__manual__"))

            selected = await rich_select("Select embedding model:", choices=choices)

            if selected == "__manual__":
                manual_model = await _manual_model_entry()
                return (manual_model, current_api_key)
            elif selected:
                return (selected, current_api_key)
            else:
                return (None, current_api_key)

        elif other_models:
            formatter.warning(
                f"Found {len(other_models)} models, but none appear to be "
                "embedding models"
            )

            # Show available models and offer manual entry
            print("\nAvailable models:")
            for i, model in enumerate(other_models[:10], 1):
                print(f"  {i:2}. {model}")
            if len(other_models) > 10:
                print(f"  ... and {len(other_models) - 10} more")

            manual_model = await _manual_model_entry()
            return (manual_model, current_api_key)
        else:
            formatter.warning("No models found on server")
            manual_model = await _manual_model_entry()
            return (manual_model, current_api_key)
    else:
        # Fall back to manual entry
        formatter.warning("Could not detect available models")
        manual_model = await _manual_model_entry()
        return (manual_model, current_api_key)


async def _manual_model_entry() -> str | None:
    """Handle manual model entry with examples."""
    print("\nCommon embedding models:")
    print("  - dengcao/Qwen3-Embedding-8B:Q5_K_M (Qwen - Best accuracy)")
    print("  - nomic-embed-text (Nomic - Fast)")
    print("  - mxbai-embed-large (MixedBread)")
    print("  - all-minilm-l6-v2 (Sentence Transformers - Lightweight)")
    print()

    model = await rich_text(
        "Enter model name:",
        validate=lambda x: _validate_non_empty(x, "Model name"),
    )

    return model.strip() if model else None


async def _select_reranking_model(
    base_url: str, api_key: str | None, formatter: RichOutputFormatter
) -> str | None:
    """Automatically select a reranking model if available.

    Args:
        base_url: Endpoint URL
        api_key: Optional API key
        formatter: Output formatter

    Returns:
        Selected reranking model name or None if none available
    """
    # Try to detect available models
    formatter.safe_progress_indicator("Checking for reranking models...")
    available_models, _ = await _fetch_available_models(base_url, api_key)

    if available_models:
        reranking_models = _filter_reranking_models(available_models)

        if reranking_models:
            formatter.success(f"Found {len(reranking_models)} reranking models")

            choices = [(model, model) for model in reranking_models]

            # User MUST select a reranking model if available
            selected = await rich_select(
                "Select reranking model (improves search accuracy):", choices=choices
            )

            return selected
        else:
            # No reranking models found - silently skip
            return None
    else:
        # Could not fetch models - silently skip
        return None


async def _validate_detected_config(
    config_data: dict[str, Any], formatter: RichOutputFormatter
) -> bool:
    """Validate configuration detected from environment variables."""
    formatter.safe_progress_indicator("Validating detected configuration...")

    try:
        provider = config_data.get("provider")

        if provider == "voyageai":
            api_key = config_data.get("api_key")
            if not api_key:
                formatter.error("VoyageAI API key not found")
                return False
            return await _validate_voyageai_key(api_key, formatter)

        elif provider == "openai":
            api_key = config_data.get("api_key")
            base_url = config_data.get("base_url")
            model = config_data.get("model", "text-embedding-3-small")

            if not api_key:
                formatter.error("OpenAI API key not found")
                return False

            # If it's a local endpoint, treat it as OpenAI-compatible
            if base_url and any(
                host in base_url.lower()
                for host in ["localhost", "127.0.0.1", "host.docker.internal"]
            ):
                return await _validate_openai_compatible(config_data, formatter)
            else:
                # Official OpenAI endpoint
                return await _validate_openai_key(api_key, model, formatter)

        else:
            formatter.error(f"Unknown provider: {provider}")
            return False

    except Exception as e:
        formatter.error(f"Validation failed: {e}")
        return False


async def _validate_voyageai_key(api_key: str, formatter: RichOutputFormatter) -> bool:
    """Test VoyageAI API key with minimal embedding request"""
    try:
        formatter.info("🔄 Validating API key...")

        # Create a test configuration
        config = EmbeddingConfig(
            provider="voyageai", api_key=SecretStr(api_key), model=VOYAGE_DEFAULT_MODEL
        )

        # Try to create provider and test connection
        from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.create_provider(config)

        # Test with minimal embedding
        await provider.embed(["test connection"])
        formatter.success("API key validated successfully")
        return True

    except Exception as e:
        formatter.error(f"Validation failed: {e}")
        return False


async def _validate_openai_key(
    api_key: str, model: str, formatter: RichOutputFormatter
) -> bool:
    """Test OpenAI API key with minimal embedding request"""
    try:
        formatter.info("🔄 Validating API key...")

        # Create a test configuration
        config = EmbeddingConfig(
            provider="openai", api_key=SecretStr(api_key), model=model
        )

        # Try to create provider and test connection
        from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.create_provider(config)

        # Test with minimal embedding
        await provider.embed(["test connection"])
        formatter.success("API key validated successfully")
        return True

    except Exception as e:
        formatter.error(f"Validation failed: {e}")
        return False


async def _validate_openai_compatible(
    config_data: dict[str, Any], formatter: RichOutputFormatter
) -> bool:
    """Test OpenAI-compatible endpoint connection"""
    try:
        formatter.info("🔄 Testing connection...")

        # Create a test configuration
        config_kwargs = {
            "provider": "openai",
            "base_url": config_data["base_url"],
            "model": config_data["model"],
        }

        if "api_key" in config_data:
            config_kwargs["api_key"] = SecretStr(config_data["api_key"])

        config = EmbeddingConfig(**config_kwargs)

        # Try to create provider and test connection
        from chunkhound.core.config.embedding_factory import EmbeddingProviderFactory

        provider = EmbeddingProviderFactory.create_provider(config)

        # Test with minimal embedding
        await provider.embed(["test connection"])
        formatter.success("Connection validated successfully")
        return True

    except Exception as e:
        formatter.error(f"Connection failed: {e}")
        return False


async def _ensure_api_key(
    provider_type: str,
    base_url: str | None,
    api_key: str | None,
    formatter: RichOutputFormatter,
    already_declined: bool = False,
) -> str | None:
    """
    Ensure we have an API key if needed for the provider.

    Args:
        provider_type: Type of provider (voyageai, openai, openai_compatible)
        base_url: Base URL for the provider (relevant for openai_compatible)
        api_key: Existing API key if any
        formatter: Output formatter
        already_declined: True if user already declined a detected key

    Returns:
        API key string if needed, None if not needed, or None if user cancelled
    """
    provider_info = EmbeddingProviderFactory.get_provider_info(provider_type)

    if provider_info["requires_api_key"] is True:
        # Always require API key
        if not api_key:
            api_key = await _prompt_for_api_key(
                provider_type, formatter, already_declined
            )
        return api_key
    elif provider_info["requires_api_key"] == "auto":
        # Test connection first, prompt for key if needed
        if not api_key:
            # If user already declined a detected key, always give them the option to enter their own
            if already_declined:
                api_key = await _prompt_for_api_key(
                    provider_type, formatter, already_declined
                )
                # Return whatever the user enters (could be None if they skip, which is valid for auto providers)
                return api_key
            else:
                needs_auth = await _test_needs_auth(base_url, formatter)
                if needs_auth:
                    api_key = await _prompt_for_api_key(
                        provider_type, formatter, already_declined
                    )
                    # For openai_compatible, API key is optional - allow empty
                    if not api_key and provider_type != "openai_compatible":
                        return None
        return api_key

    return api_key


async def _prompt_for_api_key(
    provider_type: str, formatter: RichOutputFormatter, already_declined: bool = False
) -> str | None:
    """Prompt user for API key based on provider type.

    Args:
        provider_type: Type of provider
        formatter: Output formatter
        already_declined: True if user already declined a detected key
    """
    provider_info = EmbeddingProviderFactory.get_provider_info(provider_type)
    provider_name = provider_info["display_name"]

    if provider_type == "voyageai":
        formatter.section_header(f"{provider_name} API Key")
        print("Getting Started:")
        formatter.bullet_list(
            [
                "Visit: https://www.voyageai.com",
                "Sign up for a free account (includes free credits)",
                "Find your API key in the dashboard",
            ]
        )
        print()

        url_opened = False  # Track if we've opened the URL

        while True:

            def validate_key(x):
                if not x.strip() and not url_opened:
                    return True  # Allow empty field first time
                return _validate_api_key_format(x, "pa-")

            api_key = await rich_text(
                "Enter your VoyageAI API key:",
                default="",  # No pre-filling - detection handled at higher level
                validate=validate_key,
            )

            if not api_key.strip() and not url_opened:
                url_opened = _open_url_on_empty_input(
                    "https://www.voyageai.com", "VoyageAI", formatter
                )
                continue

            return api_key.strip()

    elif provider_type == "openai":
        formatter.section_header(f"{provider_name} API Key")
        print("You can get an API key from: https://platform.openai.com/api-keys\n")

        url_opened = False  # Track if we've opened the URL

        while True:

            def validate_key(x):
                if not x.strip() and not url_opened:
                    return True  # Allow empty field first time
                return _validate_api_key_format(x, "sk-")

            api_key = await rich_text(
                "Enter your OpenAI API key:",
                default="",  # No pre-filling - detection handled at higher level
                validate=validate_key,
            )

            if not api_key.strip() and not url_opened:
                url_opened = _open_url_on_empty_input(
                    "https://platform.openai.com/api-keys", "OpenAI", formatter
                )
                continue

            return api_key.strip()

    elif provider_type == "openai_compatible":
        formatter.section_header(f"{provider_name} API Key")
        print("API key is optional for most OpenAI-compatible endpoints.")
        print("Leave empty if your endpoint doesn't require authentication.\n")

        api_key = await rich_text(
            "API Key (optional, press Enter to skip):",
            default="",  # No pre-filling - detection handled at higher level
            validate=lambda x: True,  # No validation - allow any input including empty
        )

        return api_key.strip() if api_key.strip() else None

    return None


async def _test_needs_auth(
    base_url: str | None, formatter: RichOutputFormatter
) -> bool:
    """Test if the endpoint needs authentication by making a request without API key."""
    if not base_url:
        return False

    try:
        formatter.safe_progress_indicator(
            "Testing endpoint authentication requirements..."
        )

        # Try to fetch models without API key
        models, needs_auth = await _fetch_available_models(base_url, None)
        return needs_auth
    except Exception:
        # If we can't test, assume auth is needed to be safe
        return True


async def _select_model_unified(
    provider_type: str,
    base_url: str | None,
    api_key: str | None,
    formatter: RichOutputFormatter,
    model_type: str = "embedding",  # "embedding" or "reranking"
) -> tuple[str | None, str | None]:
    """
    Select model - either from API discovery or defaults.

    Args:
        provider_type: Type of provider
        base_url: Base URL for API calls
        api_key: API key for authentication
        formatter: Output formatter
        model_type: Type of model to select ("embedding" or "reranking")

    Returns:
        Tuple of (selected_model, api_key_used)
    """
    provider_info = EmbeddingProviderFactory.get_provider_info(provider_type)

    # Try dynamic discovery for openai_compatible
    if provider_info["supports_model_listing"] and base_url:
        formatter.safe_progress_indicator("Detecting available models...")
        models, needs_auth = await _fetch_available_models(base_url, api_key)

        # If we still need auth, the API key resolution in _configure_provider_unified failed
        # In this case, we should not try to prompt again - just proceed without dynamic discovery
        if needs_auth and not api_key:
            formatter.warning(
                "Could not authenticate with endpoint - proceeding with manual entry"
            )
            models = None

        if models:
            if model_type == "embedding":
                filtered, _ = _filter_embedding_models(
                    models
                )  # Unpack tuple, take first list
            else:
                filtered = _filter_reranking_models(models)

            if filtered:
                choices = [(model, model) for model in filtered]
                selected = await rich_select(
                    f"Select {model_type} model:", choices=choices
                )
                return selected, api_key

    # Use defaults for voyageai and openai
    defaults = (
        provider_info["default_models"]
        if model_type == "embedding"
        else provider_info["default_rerankers"]
    )
    if defaults:
        choices = [(f"{model} - {desc}", model) for model, desc in defaults]
        default = (
            provider_info["default_selection"]
            if model_type == "embedding"
            else provider_info["default_reranker"]
        )

        selected = await rich_select(
            f"Select {model_type} model:", choices=choices, default=default
        )

        return selected, api_key

    # Manual entry fallback for embedding models only
    if model_type == "embedding":
        model = await rich_text(
            "Enter embedding model name:",
            validate=lambda x: _validate_non_empty(x, "Model name"),
        )
        return model.strip(), api_key

    # No reranking model available
    return None, api_key


async def _configure_provider_unified(
    provider_type: str,
    base_url: str | None = None,
    api_key: str | None = None,
    skip_intro: bool = False,
    formatter: RichOutputFormatter | None = None,
    already_declined_key: bool = False,
) -> dict[str, Any] | None:
    """
    Unified provider configuration flow for all provider types.

    Args:
        provider_type: Type of provider (voyageai, openai, openai_compatible)
        base_url: Base URL for API calls (relevant for openai_compatible)
        api_key: Existing API key if any
        skip_intro: Skip intro messages (for auto-detected flows)
        formatter: Output formatter
        already_declined_key: True if user already declined a detected key

    Returns:
        Complete configuration dictionary if successful, None if cancelled
    """
    if not formatter:
        formatter = RichOutputFormatter()

    provider_info = EmbeddingProviderFactory.get_provider_info(provider_type)

    if not skip_intro:
        formatter.section_header(f"{provider_info['display_name']} Configuration")

    # Step 1: Handle API key - ensure we have credentials before model discovery
    api_key = await _ensure_api_key(
        provider_type, base_url, api_key, formatter, already_declined_key
    )
    if not api_key and provider_info["requires_api_key"] is True:
        return None

    # Step 2: Select embedding model
    model, used_api_key = await _select_model_unified(
        provider_type, base_url, api_key, formatter, model_type="embedding"
    )
    if not model:
        return None

    # Update API key if one was discovered during model selection
    if used_api_key and not api_key:
        api_key = used_api_key

    # Step 3: Select reranking model if supported
    rerank_model = None
    if provider_info["supports_reranking"]:
        rerank_model, _ = await _select_model_unified(
            provider_type, base_url, api_key, formatter, model_type="reranking"
        )

    # Step 4: Build configuration
    config_data = {
        "provider": "openai" if provider_type == "openai_compatible" else provider_type,
        "model": model,
    }

    if base_url:
        config_data["base_url"] = base_url
    if api_key:
        config_data["api_key"] = api_key
    if rerank_model:
        config_data["rerank_model"] = rerank_model

    # Step 5: Validate configuration
    if await _validate_provider_config(config_data, formatter):
        return config_data
    else:
        formatter.error("Configuration validation failed. Please try again.")
        return None


async def _validate_provider_config(
    config_data: dict[str, Any], formatter: RichOutputFormatter
) -> bool:
    """Validate provider configuration by testing connection."""
    provider = config_data.get("provider")

    if provider == "voyageai":
        api_key = config_data.get("api_key")
        if not api_key:
            formatter.error("VoyageAI API key not found")
            return False
        return await _validate_voyageai_key(api_key, formatter)

    elif provider == "openai":
        api_key = config_data.get("api_key")
        base_url = config_data.get("base_url")
        model = config_data.get("model", "text-embedding-3-small")

        # Only require API key for official OpenAI endpoints
        if is_official_openai_endpoint(base_url):
            if not api_key:
                formatter.error("OpenAI API key not found")
                return False
            # Official OpenAI endpoint
            return await _validate_openai_key(api_key, model, formatter)
        else:
            # Custom/local endpoints don't require API key
            return await _validate_openai_compatible(config_data, formatter)

    else:
        formatter.error(f"Unknown provider: {provider}")
        return False


async def _save_configuration(
    config_data: dict[str, Any], target_path: Path, formatter: RichOutputFormatter
) -> tuple[Path | None, str]:
    """Save configuration to .chunkhound.json

    Returns:
        Tuple of (config_path, status) where status is "saved", "cancelled", or "error"
    """
    try:
        config_path = target_path / ".chunkhound.json"

        # Create configuration structure
        config = {"embedding": config_data}

        # Show summary before saving
        content = [
            ("Provider", config_data["provider"]),
            ("Model", config_data.get("model", "default")),
            ("Database", ".chunkhound/db"),
        ]
        if "base_url" in config_data:
            content.insert(2, ("Endpoint", config_data["base_url"]))
        if "rerank_model" in config_data:
            # Insert reranking info after model but before database
            insert_pos = len(content) - 1  # Before database
            content.insert(insert_pos, ("Reranking", config_data["rerank_model"]))

        formatter.box_section("Configuration Summary", content)

        # Confirm save
        should_save = await rich_confirm(
            f"\nSave configuration to {config_path}?", default=True
        )

        if not should_save:
            return None, "cancelled"

        # Write configuration file
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path, "saved"

    except Exception as e:
        formatter.error(f"Failed to save configuration: {e}")
        return None, "error"
