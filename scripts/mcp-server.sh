#!/usr/bin/env bash
# ChunkHound MCP Server - Easy startup script for IDEs like Zed
# Usage: ./scripts/mcp-server.sh [--db path/to/db] [--verbose]

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Default database path
DB_PATH="${CHUNKHOUND_DB_PATH:-$HOME/.cache/chunkhound/chunks.duckdb}"

# Parse arguments
ARGS=()
VERBOSE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --db)
            DB_PATH="$2"
            shift 2
            ;;
        --verbose|-v)
            ARGS+=("--verbose")
            VERBOSE=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--db path/to/db] [--verbose]"
            echo ""
            echo "Environment variables:"
            echo "  CHUNKHOUND_DB_PATH           - Default database path"
            echo "  CHUNKHOUND_EMBEDDING__API_KEY - Required for semantic search"
            echo ""
            echo "Examples:"
            echo "  $0                              # Use default database"
            echo "  $0 --db ./my-project.duckdb    # Use specific database"
            echo "  $0 --verbose                   # Enable verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure database directory exists
mkdir -p "$(dirname "$DB_PATH")"

# Check if database exists, suggest indexing if not (only in verbose mode)
if [[ ! -f "$DB_PATH" && $VERBOSE -eq 1 ]]; then
    echo "Warning: Database not found at $DB_PATH" >&2
    echo "Run 'uv run chunkhound index .' to index the current directory first" >&2
    echo "" >&2
fi

# Export environment variables for the MCP server
export CHUNKHOUND_DB_PATH="$DB_PATH"

# Start MCP server using uv (suppress startup messages unless verbose)
if [[ $VERBOSE -eq 1 ]]; then
    echo "Starting ChunkHound MCP server..." >&2
    echo "Database: $DB_PATH" >&2
    echo "Use Ctrl+C to stop" >&2
    echo "" >&2
fi

exec uv run chunkhound mcp --db "$DB_PATH" "${ARGS[@]}"
