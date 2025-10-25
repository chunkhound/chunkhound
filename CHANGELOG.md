# Changelog

All notable changes to ChunkHound will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Per-file timeout controls: `indexing.per_file_timeout_seconds`, `indexing.per_file_timeout_min_size_kb`, CLI flags and env overrides.
- Change-detection with mandatory checksum verification for unchanged files (now always-on when size+mtime match).
- DuckDB schema: `files.content_hash` column (idempotent migration via `ALTER TABLE IF NOT EXISTS`).
- LanceDB schema: `files.content_hash` column for checksum support.
- Progress improvements: split "Parsing files" vs "Handling files" with live cumulative info.
- Post-run prompt to add timed-out files to `indexing.exclude` in `.chunkhound.json` when interactive.
- Env override for DB executor timeout: `CHUNKHOUND_DB_EXECUTE_TIMEOUT`.

### Fixed
- DuckDB `get_file_by_path(as_model=True)` now returns epoch-float `mtime` and correct `size_bytes` for accurate skip checks.
- Skipped counts broken out into `Unchanged` and `Filtered` buckets for clarity.

### Security
- Removed embedded API key from `.chunkhound.json`. Use environment variables instead (e.g., `CHUNKHOUND_EMBEDDING__API_KEY`).

### Changed
- **BREAKING**: Checksum algorithm switched from SHA-256 to xxHash3-64 for faster file change detection. All files will be reindexed on first run after upgrade to compute new checksums.
- **BREAKING**: Checksum verification now mandatory for all providers when size+mtime match. Removed `indexing.verify_checksum_when_mtime_equal` config option and `--verify-checksum` CLI flag.
- **BREAKING**: LanceDB provider now requires `content_hash` column in files schema.
- Default per-file timeout is now enabled by default: `indexing.per_file_timeout_seconds=3.0` (previously `0`, disabled). Set to `0` to disable.
- When timeouts are enabled and no explicit concurrency is set, parser workers auto‑scale to `cpu_count` (capped at 32). Override via `indexing.max_concurrent`, `--max-concurrent`, or `CHUNKHOUND_INDEXING__MAX_CONCURRENT`.
- PHP language support with comprehensive tree-sitter parsing for classes, interfaces, traits, functions, methods, namespaces, and PHPDoc comments
- Vue.js Single File Component (SFC) support with specialized parsing for template, script, and style sections
- Cross-reference tracking between Vue template elements and script definitions for enhanced semantic understanding
- Vue template directive and event handler parsing with full metadata extraction

### Enhanced
- Embedding performance settings (`batch_size`, `timeout`, `max_concurrent_batches`) are now documented as user-configurable via `.chunkhound.json` and environment variables for custom endpoints like Ollama

## [3.3.1] - 2025-09-25

### Enhanced
- Dependency updates to latest stable versions for improved stability and performance
- Test infrastructure reliability with better provider detection and error handling

### Fixed
- Tree-sitter 0.25.x API compatibility ensuring parsing works with latest language parsers
- Code formatting and import organization for cleaner, more maintainable codebase

## [3.3.0] - 2025-09-21

### Added
- Official Windows support with full CI testing across Windows, macOS, and Ubuntu
- Command-line search functionality (`chunkhound search`) for semantic and regex queries without starting MCP
- CONTRIBUTING.md guidelines
- Setup wizard when `.chunkhound.json` isn't found in the directory

### Fixed
- File exclude patterns (**/tmp/**) on Linux systems
- Regex search path resolution across platforms

## [3.2.0] - 2025-08-24

### Enhanced
- Semantic search upgraded from two-hop to dynamic multi-hop expansion with intelligent stopping criteria, delivering more comprehensive and contextually relevant results while avoiding search explosion

## [3.1.0] - 2025-08-21

### Added
- PDF document parsing and indexing with full text extraction using PyMuPDF integration

### Enhanced
- Language support expanded to 22 languages with comprehensive documentation breakdown

### Fixed
- JSON file parsing now extracts specific node content instead of entire file content, improving search precision and reducing noise

## [3.0.1] - 2025-08-21

### Enhanced
- Documentation site improved with cross-linking between pages and hero image for better navigation
- OpenAI-compatible endpoint flexibility increased by making API keys optional for local deployments
- Test infrastructure reliability improved with comprehensive CI fixes and timeout handling

### Fixed
- JSON file parsing now handles empty chunks correctly, eliminating indexing failures on common JSON patterns
- Test suite stability enhanced with proper background task cleanup and configuration isolation
- GitHub Actions workflow simplified and made more reliable by removing redundant processes

## [3.0.0] - 2025-08-20

### Added
- VoyageAI embedding provider with advanced two-hop semantic search and reranking capabilities
- GitHub Pages documentation site with interactive examples and improved navigation
- Intelligent file exclusion system with .gitignore support and JSON size filtering
- Advanced makefile parsing with dependency analysis for better code comprehension
- Comprehensive test suite for database consistency and integration testing
- Real-time filesystem indexing with MCP integration for live code monitoring

### Enhanced
- Parsing system completely rebuilt with cAST (Code AST) algorithm for universal language support
- Configuration system dramatically simplified with fewer user-facing options for easier setup
- OpenAI provider unified to handle both standard and custom OpenAI-compatible endpoints
- MCP server reliability improved with proper initialization sequencing and watchdog coordination
- Test infrastructure enhanced with Ollama compatibility and extended timeouts
- Directory indexing consolidated between CLI and MCP with shared service architecture

### Fixed
- MCP server initialization blocking resolved - no more startup deadlocks during directory scanning
- Custom OpenAI endpoint configuration now properly recognized and applied
- Real-time indexing now generates missing embeddings for unchanged code chunks
- SSL verification disabled for custom OpenAI-compatible endpoints to support local deployments
- Watchdog filesystem monitoring no longer blocks MCP server startup process
- MCP server properly respects target directory path arguments across all operations

### Removed
- TEI (Text Embeddings Inference) provider support - simplified provider ecosystem
- BGE provider support - consolidated to core providers for better maintenance
- Legacy parsing system replaced with modern cAST algorithm
- Obsolete configuration documentation and setup files cleaned up

## [2.8.1] - 2025-07-20

### Enhanced
- Architecture documentation significantly improved for better LLM comprehension and AI-assisted development workflows

### Fixed
- Type annotation syntax errors that could cause import failures in Python 3.10+ environments
- Enhanced smoke tests now detect forward reference type annotation issues early

## [2.8.0] - 2025-07-20

### Added
- MCP HTTP transport support alongside stdio transport for flexible deployment options

### Enhanced
- Configuration system unified across CLI and MCP components for consistent behavior
- File change processing reliability improved in MCP servers with better debouncing and coordination
- Database portability enhanced with relative path storage

### Fixed
- MCP server initialization deadlocks and startup crashes resolved with proper async coordination
- File deletion handling improved using IndexingCoordinator for better reliability
- MCP server tool discovery enhanced with fallback logic for better error recovery
- File path resolution improved in DuckDB provider for cross-platform consistency

## [2.7.0] - 2025-07-12

### Fixed
- MCP server now uses configured embedding model instead of hardcoded text-embedding-3-small default, ensuring semantic search works with any configured model
- MCP test environment improvements with comprehensive test data and configuration files

## [2.6.3] - 2025-07-10

### Fixed
- Configuration merge precedence now correctly preserves environment variables over JSON config values
- MCP server semantic search now works properly when running from different directories

### Removed
- Removed obsolete Ubuntu 20 Dockerfile as issue was resolved in configuration system

## [2.6.2] - 2025-07-10

### Fixed
- MCP server now properly loads embedding provider configuration from target directory

## [2.6.1] - 2025-07-10

### Fixed
- MCP server now properly respects CLI-provided project root directory for configuration loading
- Configuration files (.chunkhound.json) are now correctly loaded when running MCP server from different directories

## [2.6.0] - 2025-07-10

### Fixed
- MCP server crashes on Ubuntu and Linux systems when running from different directories by fixing database path resolution and process coordination
- Enhanced TaskGroup error reporting to show underlying causes instead of generic wrapper errors
- Configuration file loading in MCP server now properly respects .chunkhound.json files in target directories
- Database lock conflicts between multiple MCP instances resolved with proper process detection

### Enhanced
- Docker test infrastructure for MCP server validation to prevent future regressions
- Improved error messages for debugging MCP server issues with detailed analysis

## [2.5.4] - 2025-07-10

### Fixed
- MCP server reliability on Ubuntu and other Linux distributions when running from different directories
- Database path resolution consistency across all MCP server components

## [2.5.3] - 2025-07-10

### Fixed
- MCP server communication reliability improved by removing debug logging that interfered with JSON-RPC protocol

## [2.5.2] - 2025-07-10

### Added
- Automatic database optimization during embedding generation to maintain performance with large datasets (every 1000 batches, configurable via `CHUNKHOUND_EMBEDDING_OPTIMIZATION_BATCH_FREQUENCY`)

### Fixed
- MCP server compatibility on Ubuntu and other strict platforms by preserving virtual environment context in subprocesses
- OpenAI embedding provider crash on Ubuntu due to async resource creation outside event loop context

## [2.5.1] - 2025-01-09

### Fixed
- Project detection now properly respects CHUNKHOUND_PROJECT_ROOT environment variable, ensuring MCP command works correctly when launched from any directory
- Removed duplicate MCP parser function that could cause confusion

## [2.5.0] - 2025-01-09

### Enhanced
- MCP positional path argument now controls complete project scope - database location, config file search, and watch paths are all set to the specified directory instead of just watch paths

### Fixed
- MCP launcher import path resolution when running from different directories, eliminating TaskGroup errors on Ubuntu and other strict platforms

## [2.4.4] - 2025-01-09

### Fixed
- Ubuntu TaskGroup crash fixed by removing problematic directory change in MCP launcher

## [2.4.3] - 2025-01-09

### Fixed
- MCP server now works correctly when launched from any directory, not just the project root
- Fixed path resolution inconsistencies that caused TaskGroup errors on Ubuntu deployments

## [2.4.2] - 2025-01-09

### Added
- MCP command now accepts optional path argument to specify directory for indexing and watching (defaults to current directory)

### Fixed
- Parser architecture inconsistencies resolved across C, Bash, and Makefile parsers for consistent search functionality
- MCP server database duplication eliminated through proper async task isolation
- LanceDB storage growth controlled with automatic optimization during quiet periods
- MCP server reliability improved with corrected import structure and dependency resolution
- Python parser behavior now consistent between CLI and MCP modes
- Search operation freezes after file deletion resolved with proper thread safety

## [2.4.1] - 2025-01-09

### Fixed
- Package structure consolidated under chunkhound/ directory for improved import reliability and Python packaging best practices

## [2.4.0] - 2025-01-09

### Fixed
- LanceDB storage growth issue resolved with automatic database optimization during quiet periods
- Configuration system project root detection for .chunkhound.json files improved

### Changed
- Enhanced database provider architecture with capability detection and activity tracking
- Modernized configuration system by removing legacy registry config building

## [2.3.1] - 2025-07-09

### Fixed
- MCP server communication reliability improved by preventing stderr output from corrupting JSON-RPC messages
- Enhanced configuration documentation with automatic .chunkhound.json detection examples

## [2.3.0] - 2025-07-08

### Changed
- **BREAKING**: Configuration system completely refactored with centralized management and clear precedence hierarchy
- **BREAKING**: Automatic configuration file loading removed - config files now only load with explicit `--config` flag
- **BREAKING**: Environment variables standardized to `CHUNKHOUND_*` prefix with `__` delimiters (e.g., `CHUNKHOUND_EMBEDDING__API_KEY`)
- **BREAKING**: Legacy `OPENAI_API_KEY` and `OPENAI_BASE_URL` environment variables no longer supported

### Added
- Complete CLI argument coverage for all configuration options
- Centralized configuration precedence: CLI args → Config file → Environment variables → Defaults
- Comprehensive migration guide for updating existing configurations
- Database file gitignore pattern for Lance database files

### Fixed
- MCP server database duplication caused by shared transaction state across async tasks
- Parser architecture inconsistencies for C, Bash, and Makefile language parsers
- Configuration auto-detection issues that caused deployment complexity

## [2.2.0] - 2025-01-07

### Fixed
- Database freezing during concurrent file operations through proper async/sync boundary handling
- Thread safety issues in DuckDB provider with synchronized WAL cleanup and operation timeouts
- LanceDB duplicate file entries through atomic merge operations and path normalization
- File deletion operations now properly handle async contexts without blocking the event loop

### Changed
- Aligned LanceDB provider with serial executor pattern for consistency with DuckDB
- Improved path normalization to handle symlinks and different path representations
- Enhanced database operation reliability with proper thread isolation

### Added
- Support for complete configuration storage including API keys in .chunkhound.json files
- Consolidated embedding provider creation system for consistent behavior across CLI and config files

## [2.1.4] - 2025-07-03

### Fixed
- CLI argument defaults no longer override config file values
- Updated dependencies via uv.lock

## [2.1.3] - 2025-07-03

### Changed
- Consolidated embedding provider creation to use single factory pattern for consistency
- Reduced embedding provider log verbosity for cleaner output

## [2.1.2] - 2025-07-03

### Fixed
- API key configuration loading from .chunkhound.json files
- Configuration precedence documentation to match actual behavior

### Added
- Complete configuration examples with API key and security guidance

## [2.1.1] - 2025-07-03

### Added
- Centralized version management system for consistent versioning across all components

### Changed
- Simplified version updates through automated scripts
- Enhanced installation and development documentation
- Code formatting improvements and linting cleanup

### Fixed
- Version consistency across CLI, MCP server, and package initialization
- Import statement in package `__init__.py` for better module exposure

## [2.1.0] - 2025-07-02

### Fixed
- Database duplication in MCP server by implementing single-threaded executor pattern
- WAL corruption handling during DuckDB catalog replay
- Parser architecture inconsistencies for C, Bash, and Makefile parsers
- DuckDB foreign key constraint transaction limitations
- Python parser CLI/MCP divergence through unified factory pattern
- Connection management architectural violations

### Changed
- Consolidated database operations through DuckDBProvider executor pattern
- Simplified ConnectionManager to handle only connection lifecycle
- Updated file discovery patterns to include all 16 supported languages
- Removed deprecated connection methods and schema fields
- Enhanced transaction handling with contextvars for task isolation

### Added
- Automatic database migration system for schema updates
- Enhanced parser functionality for C pointer functions and Bash function bodies
- Task-local transaction state management
- Comprehensive executor methods for database operations

## [2.0.0] - 2025-06-26

### Added
- 10 new language parsers: Rust, Go, C++, C, Kotlin, Groovy, Bash, TOML, Makefile, Matlab
- Search pagination with response size limits
- Registry-based parser architecture
- MCP search task coordinator
- Test coverage for file modification tracking
- Comment and docstring indexing for all language parsers
- Background periodic indexing for better performance
- Path filtering support for targeted searches
- HNSW index WAL recovery with enhanced checkpoints
- Embedding cache optimization with CRC32-based content tracking

### Changed
- **BREAKING**: 'run' command renamed to 'index' with current directory default
- **BREAKING**: Parser system refactored to registry pattern
- Centralized language support in Language enum
- Optimized embedding performance with token-aware batching
- Enhanced PyInstaller compatibility
- Improved cross-platform build support (Windows, Ubuntu Docker)
- Enhanced MCP server JSON-RPC communication with logging suppression

### Fixed
- Parser error handling and registry integration
- OpenAI token limit handling
- PyInstaller module path resolution
- Database WAL corruption issues on server exit
- File watcher cancellation responsiveness
- Signal handler safety by removing unsafe database operations
- Windows PyInstaller and MATLAB dependency issues
- Build workflow reliability across platforms

## [1.2.3] - 2025-06-23

### Changed
- Default database location changed to current directory for better persistence

### Fixed
- OpenAI token limit exceeded error with dynamic batching for large embedding requests
- Empty chunk filtering to reduce noise in search results
- Python parser validation for empty symbol names
- Windows build support with comprehensive GitHub Actions workflow
- macOS Intel build issues with UV package manager installation
- Cross-platform build workflow reliability

### Added
- Windows build support with automated testing
- Enhanced debugging for build processes across platforms

## [1.2.2] - 2024-12-15

### Added
- File watching CLI for real-time code monitoring

### Changed
- Unified JavaScript and TypeScript parsers
- Default database location to current directory

### Fixed
- Empty symbol validation in Python parser

## [1.2.1] - 2024-11-28

### Added
- Ubuntu 20.04 build support
- Token limit management for MCP search

### Fixed
- Duplicate chunks after file edits
- File modification detection race conditions

## [1.2.0] - 2024-11-15

### Added
- C# language support
- JSON, YAML, and plain text file support
- File watching with real-time indexing

### Fixed
- File deletion handling
- Database connection issues

## [1.1.0] - 2025-06-12

### Added
- Multi-language support: TypeScript, JavaScript, C#, Java, and Markdown
- Comprehensive CLI interface
- Binary distribution with faster startup

### Changed
- Improved CLI startup performance (90% faster)
- Binary startup performance (16x faster)

### Fixed
- Version display consistency
- Cross-platform build issues

## [1.0.1] - 2025-06-11

### Added
- Python 3.10+ compatibility
- PyPI publishing
- Standalone executable support
- MCP server integration

### Fixed
- Dependency conflicts
- OpenAI model parameter handling
- Binary compilation issues

## [1.0.0] - 2025-06-10

### Added
- Initial release of ChunkHound
- Python parsing with tree-sitter
- DuckDB backend for storage and search
- OpenAI embeddings for semantic search
- CLI interface for indexing and searching
- MCP server for AI assistant integration
- File watching for real-time indexing
- Regex search capabilities

For more information, visit: https://github.com/chunkhound/chunkhound

[Unreleased]: https://github.com/chunkhound/chunkhound/compare/v3.3.1...HEAD
[3.3.1]: https://github.com/chunkhound/chunkhound/compare/v3.3.0...v3.3.1
[3.3.0]: https://github.com/chunkhound/chunkhound/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/chunkhound/chunkhound/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/chunkhound/chunkhound/compare/v3.0.1...v3.1.0
[3.0.1]: https://github.com/chunkhound/chunkhound/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/chunkhound/chunkhound/compare/v2.8.1...v3.0.0
[2.8.1]: https://github.com/chunkhound/chunkhound/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/chunkhound/chunkhound/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/chunkhound/chunkhound/compare/v2.6.3...v2.7.0
[2.6.3]: https://github.com/chunkhound/chunkhound/compare/v2.6.2...v2.6.3
[2.6.2]: https://github.com/chunkhound/chunkhound/compare/v2.6.1...v2.6.2
[2.6.1]: https://github.com/chunkhound/chunkhound/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/chunkhound/chunkhound/compare/v2.5.4...v2.6.0
[2.5.4]: https://github.com/chunkhound/chunkhound/compare/v2.5.3...v2.5.4
[2.5.3]: https://github.com/chunkhound/chunkhound/compare/v2.5.2...v2.5.3
[2.5.2]: https://github.com/chunkhound/chunkhound/compare/v2.5.1...v2.5.2
[2.5.1]: https://github.com/chunkhound/chunkhound/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/chunkhound/chunkhound/compare/v2.4.4...v2.5.0
[2.4.4]: https://github.com/chunkhound/chunkhound/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/chunkhound/chunkhound/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/chunkhound/chunkhound/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/chunkhound/chunkhound/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/chunkhound/chunkhound/compare/v2.3.1...v2.4.0
[2.3.1]: https://github.com/chunkhound/chunkhound/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/chunkhound/chunkhound/compare/v2.2.0...v2.3.0
[2.2.0]: https://github.com/chunkhound/chunkhound/compare/v2.1.4...v2.2.0
[2.1.4]: https://github.com/chunkhound/chunkhound/compare/v2.1.3...v2.1.4
[2.1.3]: https://github.com/chunkhound/chunkhound/compare/v2.1.2...v2.1.3
[2.1.2]: https://github.com/chunkhound/chunkhound/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/chunkhound/chunkhound/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/chunkhound/chunkhound/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/chunkhound/chunkhound/compare/v1.2.3...v2.0.0
[1.2.3]: https://github.com/chunkhound/chunkhound/compare/v1.2.2...v1.2.3
[1.2.2]: https://github.com/chunkhound/chunkhound/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/chunkhound/chunkhound/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/chunkhound/chunkhound/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/chunkhound/chunkhound/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/chunkhound/chunkhound/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/chunkhound/chunkhound/releases/tag/v1.0.0
### Changed
- Default per-file timeout is now enabled by default: `indexing.per_file_timeout_seconds=3.0` (previously `0`, disabled). Set to `0` to disable.
### MCP & Performance
- Fix: Disable interactive prompts in MCP mode to preserve JSON-RPC protocol.
  - CLI now checks `CHUNKHOUND_MCP_MODE=1` and never calls `input()` in that mode.
  - MCP stdio/http servers set `CHUNKHOUND_MCP_MODE=1` at startup.
- Perf: Single‑read checksum verification.
  - When `indexing.verify_checksum_when_mtime_equal=true`, we compute the sample hash once during change detection and carry it through storage, eliminating the post‑parse re‑read.
  - Reduces file I/O in the verify‑equal path and removes a small race window.
- Safety: Cap concurrent timeout children per worker.
  - Worker processes use a semaphore to limit timeout child processes.
  - The cap auto‑scales to `min(num_workers*2, 32)`; in restricted CI sandboxes without SemLock, the limiter falls back gracefully.
