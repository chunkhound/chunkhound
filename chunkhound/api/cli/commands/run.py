"""Run command module - handles directory indexing operations."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from chunkhound.core.config.config import Config
from chunkhound.registry import configure_registry, create_indexing_coordinator
from chunkhound.services.directory_indexing_service import DirectoryIndexingService
from chunkhound.version import __version__

from ..parsers.run_parser import process_batch_arguments
from ..utils.rich_output import RichOutputFormatter
from ..utils.validation import (
    ensure_database_directory,
    validate_file_patterns,
    validate_path,
    validate_provider_args,
)


async def run_command(args: argparse.Namespace, config: Config) -> None:
    """Execute the run command using the service layer.

    Args:
        args: Parsed command-line arguments
        config: Pre-validated configuration instance
    """
    # Ignore decision check (formerly top-level 'diagnose')
    if getattr(args, "check_ignores", False):
        # Ensure this mode doesn't require embeddings either
        setattr(args, "no_embeddings", True)
        await _check_ignores(args, config)
        return

    # Simulate mode
    if getattr(args, "simulate", False):
        # Ensure simulate doesn't require embeddings
        setattr(args, "no_embeddings", True)
        await _simulate_index(args, config)
        return

    # Initialize Rich output formatter
    formatter = RichOutputFormatter(verbose=args.verbose)

    # Check if local config was found (for logging purposes)
    project_dir = Path(args.path) if hasattr(args, "path") else Path.cwd()
    local_config_path = project_dir / ".chunkhound.json"
    if local_config_path.exists():
        formatter.info(f"Found local config: {local_config_path}")

    # Use database path from config
    db_path = Path(config.database.path)

    # Display modern startup information
    formatter.startup_info(
        version=__version__,
        directory=str(args.path),
        database=str(db_path),
        config=config.__dict__ if hasattr(config, "__dict__") else {},
    )

    # Process and validate batch arguments (includes deprecation warnings)
    process_batch_arguments(args)

    # Validate arguments - update args.db to use config value for validation
    args.db = db_path
    if not _validate_run_arguments(args, formatter, config):
        sys.exit(1)

    try:
        # Configure registry with the Config object
        configure_registry(config)

        formatter.success(f"Service layer initialized: {args.db}")

        # Create progress manager for modern UI
        with formatter.create_progress_display() as progress_manager:
            # Get the underlying Progress instance for service layers
            progress_instance = progress_manager.get_progress_instance()

            # Create indexing coordinator with Progress instance
            indexing_coordinator = create_indexing_coordinator()
            # Pass progress to the coordinator after creation
            if hasattr(indexing_coordinator, "progress"):
                indexing_coordinator.progress = progress_instance
            # Pass startup profiling flag
            if getattr(args, "profile_startup", False):
                try:
                    setattr(indexing_coordinator, "profile_startup", True)
                except Exception:
                    pass

            # Get initial stats
            initial_stats = await indexing_coordinator.get_stats()
            formatter.initial_stats_panel(initial_stats)

            # Simple progress callback for verbose output
            def progress_callback(message: str):
                if args.verbose:
                    formatter.verbose_info(message)

            # Create indexing service with Progress instance
            indexing_service = DirectoryIndexingService(
                indexing_coordinator=indexing_coordinator,
                config=config,
                progress_callback=progress_callback,
                progress=progress_instance,
            )

            # Process directory - service layers will add subtasks to progress_instance
            stats = await indexing_service.process_directory(
                Path(args.path),
                no_embeddings=args.no_embeddings,
                thin_missing_embeddings=getattr(args, "thin_missing_embeddings", False),
            )

        # Display results
        _print_completion_summary(stats, formatter)

        # Emit startup profiling timings if requested
        if getattr(args, "profile_startup", False):
            try:
                import json as _json
                prof = getattr(indexing_coordinator, "_startup_profile", None) or {}
                if isinstance(prof, dict):
                    prof = dict(prof)
                # Attach resolved discovery backend + reasons if available
                rb = getattr(indexing_coordinator, "_resolved_discovery_backend", None)
                rr = getattr(indexing_coordinator, "_resolved_discovery_reasons", None)
                if rb and isinstance(prof, dict):
                    prof["resolved_backend"] = rb
                if rr and isinstance(prof, dict):
                    prof["resolved_reasons"] = rr
                # Attach git enumerator counters if available
                grt = getattr(indexing_coordinator, "_git_rows_tracked", None)
                gro = getattr(indexing_coordinator, "_git_rows_others", None)
                grtot = getattr(indexing_coordinator, "_git_rows_total", None)
                gps = getattr(indexing_coordinator, "_git_pathspecs", None)
                gpc = getattr(indexing_coordinator, "_git_pathspecs_capped", None)
                if grtot is not None or grt is not None or gro is not None:
                    if grt is not None:
                        prof["git_rows_tracked"] = int(grt)
                    if gro is not None:
                        prof["git_rows_others"] = int(gro)
                    if grtot is not None:
                        prof["git_rows_total"] = int(grtot)
                    if gps is not None:
                        prof["git_pathspecs"] = int(gps)
                    if gpc is not None:
                        prof["git_pathspecs_capped"] = bool(gpc)
                if prof:
                    print(_json.dumps({"startup_profile": prof}, indent=2), file=sys.stderr)
            except Exception:
                pass

        # Offer to add timed-out files to exclusion list in local config
        try:
            skipped_timeouts = []
            if hasattr(stats, "skipped_due_to_timeout"):
                skipped_timeouts = stats.skipped_due_to_timeout or []

            # Never prompt in MCP mode (stdio must not emit prompts/output)
            if skipped_timeouts and os.environ.get("CHUNKHOUND_MCP_MODE") == "1":
                formatter.info(
                    f"{len(skipped_timeouts)} files timed out. "
                    "Prompts are disabled in MCP mode. To exclude them, add to .chunkhound.json under indexing.exclude."
                )
                return

            # Respect explicit no-prompts
            if skipped_timeouts and os.environ.get("CHUNKHOUND_NO_PROMPTS") == "1":
                formatter.info(
                    f"{len(skipped_timeouts)} files timed out (prompts disabled)."
                )
                return

            # Only prompt in interactive TTY and when there are timeouts
            if skipped_timeouts and sys.stdin.isatty():
                base_dir = Path(args.path).resolve() if hasattr(args, "path") else Path.cwd().resolve()

                # Convert to unique relative paths within the project
                rel_paths: list[str] = []
                seen: set[str] = set()
                for p in skipped_timeouts:
                    try:
                        rel = Path(p).resolve().relative_to(base_dir).as_posix()
                    except Exception:
                        # If not under base_dir, keep as-is (rare)
                        rel = Path(p).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        rel_paths.append(rel)

                formatter.info(
                    f"{len(rel_paths)} timed-out files can be excluded from future runs."
                )
                reply = input("Add these to indexing.exclude in .chunkhound.json? [y/N]: ").strip().lower()
                if reply in ("y", "yes"):
                    local_config_path = base_dir / ".chunkhound.json"
                    # Load or initialize config data
                    data = {}
                    if local_config_path.exists():
                        import json

                        try:
                            data = json.loads(local_config_path.read_text())
                        except Exception:
                            data = {}

                    # Ensure structure exists
                    indexing = data.get("indexing") or {}
                    exclude_list = list(indexing.get("exclude") or [])

                    # Merge unique entries
                    existing = set(exclude_list)
                    added = 0
                    for rel in rel_paths:
                        if rel not in existing:
                            exclude_list.append(rel)
                            existing.add(rel)
                            added += 1

                    if added > 0:
                        indexing["exclude"] = exclude_list
                        data["indexing"] = indexing
                        import json

                        local_config_path.write_text(
                            json.dumps(data, indent=2, sort_keys=False) + "\n"
                        )
                        formatter.success(
                            f"Added {added} file(s) to indexing.exclude in {local_config_path}"
                        )
                    else:
                        formatter.info("All timed-out files already excluded.")
        except Exception as e:
            formatter.warning(f"Failed to offer exclusion prompt: {e}")

        formatter.success("Run command completed successfully")

    except KeyboardInterrupt:
        formatter.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        formatter.error(f"Run command failed: {e}")
        logger.exception("Run command error details")
        sys.exit(1)
    finally:
        pass


def _print_completion_summary(stats, formatter: RichOutputFormatter) -> None:
    """Print completion summary from IndexingStats using Rich formatting."""
    # Convert stats object to dictionary for Rich display
    if hasattr(stats, "__dict__"):
        stats_dict = stats.__dict__
    else:
        stats_dict = stats if isinstance(stats, dict) else {}
    formatter.completion_summary(stats_dict, stats.processing_time)


def _validate_run_arguments(
    args: argparse.Namespace, formatter: RichOutputFormatter, config: Any = None
) -> bool:
    """Validate run command arguments.

    Args:
        args: Parsed arguments
        formatter: Output formatter
        config: Configuration (optional)

    Returns:
        True if valid, False otherwise
    """
    # Validate path
    if not validate_path(args.path, must_exist=True, must_be_dir=True):
        return False

    # Ensure database directory exists
    if not ensure_database_directory(args.db):
        return False

    # Validate provider arguments
    if not args.no_embeddings:
        # Use unified config values if available, fall back to CLI args
        if config and config.embedding:
            provider = config.embedding.provider
            api_key = (
                config.embedding.api_key.get_secret_value()
                if config.embedding.api_key
                else None
            )
            base_url = config.embedding.base_url
            model = config.embedding.model
        else:
            # Check if CLI args have provider info
            provider = getattr(args, "provider", None)
            api_key = getattr(args, "api_key", None)
            base_url = getattr(args, "base_url", None)
            model = getattr(args, "model", None)

            # If no provider info found, provide helpful error
            if not provider:
                formatter.error("No embedding provider configured.")
                formatter.info("To fix this, you can:")
                formatter.info(
                    "  1. Create .chunkhound.json config file with embeddings"
                )
                formatter.info("  2. Use --no-embeddings to skip embeddings")
                return False
        if not validate_provider_args(provider, api_key, base_url, model):
            return False

    # Validate file patterns
    if not validate_file_patterns(args.include, args.exclude):
        return False

    return True


__all__ = ["run_command"]


async def _simulate_index(args: argparse.Namespace, config: Config) -> None:
    """Dry-run discovery and print list of relative files.

    Minimal implementation: perform discovery via the coordinator and print
    the discovered files sorted. Later we may reflect change-detection.
    """
    base_dir = Path(args.path).resolve() if hasattr(args, "path") else Path.cwd().resolve()

    # Optional debug output about ignore configuration (stderr to avoid breaking JSON piping)
    try:
        if getattr(args, "debug_ignores", False):
            from chunkhound.core.config.indexing_config import IndexingConfig as _IdxCfg

            sources = []
            try:
                sources = list(config.indexing.resolve_ignore_sources())
            except Exception:
                # Best-effort; keep empty on error
                sources = []

            defaults = []
            try:
                defaults = list(_IdxCfg._default_excludes())
            except Exception:
                defaults = []

            print(f"[debug-ignores] CH root: {base_dir}", file=sys.stderr)
            print(f"[debug-ignores] Active sources: {sources}", file=sys.stderr)
            # Show first 10 normalized default excludes to quickly confirm runtime defaults
            first_n = defaults[:10]
            print("[debug-ignores] Default excludes (first 10):", file=sys.stderr)
            for pat in first_n:
                print(f"  - {pat}", file=sys.stderr)
    except BrokenPipeError:
        # If stderr is piped and consumer exits early (e.g., `| head`), exit quietly
        try:
            sys.stderr.close()
        except Exception:
            pass

    # Configure registry and create services like real run
    # For simulate, avoid touching on-disk DBs. Use in-memory DuckDB when possible.
    try:
        # Prefer an in-memory DB to keep simulate side-effect free
        # Only override when path is unset or points to a non-existent parent.
        db_path = Path(getattr(config.database, "path", Path(":memory:")) or Path(":memory:"))
        if str(db_path) != ":memory":  # typos
            if str(db_path) != ":memory:" and not db_path.parent.exists():
                # If parent path doesn't exist, switch to in-memory
                try:
                    config.database.path = Path(":memory:")  # type: ignore[attr-defined]
                except Exception:
                    pass
        # If still not in-memory, ensure parent exists
        dbp = getattr(config.database, "path", None)
        if dbp and str(dbp) != ":memory:":
            Path(dbp).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # As a last resort, in-memory
        try:
            config.database.path = Path(":memory:")  # type: ignore[attr-defined]
        except Exception:
            pass

    configure_registry(config)
    indexing_coordinator = create_indexing_coordinator()
    # Ensure coordinator has access to the effective Config for discovery
    try:
        setattr(indexing_coordinator, "config", config)
    except Exception:
        pass

    # Resolve patterns using the DirectoryIndexingService helper to keep logic aligned
    from chunkhound.services.directory_indexing_service import DirectoryIndexingService

    svc = DirectoryIndexingService(indexing_coordinator=indexing_coordinator, config=config)
    include_patterns, exclude_patterns = svc._resolve_file_patterns()

    # Normalize include patterns and call internal discovery
    from chunkhound.utils.file_patterns import normalize_include_patterns

    processed_patterns = normalize_include_patterns(include_patterns)

    # Optional profiling for discovery during simulate
    files = []
    if getattr(args, "profile_startup", False):
        import time as _t
        _t0 = _t.perf_counter()
        files = await indexing_coordinator._discover_files(  # type: ignore[attr-defined]
            base_dir, processed_patterns, exclude_patterns
        )
        _t1 = _t.perf_counter()
        try:
            import json as _json
            prof = {
                "discovery_ms": round((_t1 - _t0) * 1000.0, 3),
                "files_discovered": len(files),
                "backend": (getattr(config.indexing, "discovery_backend", "python") if getattr(config, "indexing", None) else "python"),
            }
            rb = getattr(indexing_coordinator, "_resolved_discovery_backend", None)
            if rb:
                prof["resolved_backend"] = rb
            rr = getattr(indexing_coordinator, "_resolved_discovery_reasons", None)
            if rr:
                prof["resolved_reasons"] = rr
            # Attach git enumerator counters if available
            grt = getattr(indexing_coordinator, "_git_rows_tracked", None)
            gro = getattr(indexing_coordinator, "_git_rows_others", None)
            grtot = getattr(indexing_coordinator, "_git_rows_total", None)
            gps = getattr(indexing_coordinator, "_git_pathspecs", None)
            gpc = getattr(indexing_coordinator, "_git_pathspecs_capped", None)
            if grtot is not None or grt is not None or gro is not None:
                if grt is not None:
                    prof["git_rows_tracked"] = int(grt)
                if gro is not None:
                    prof["git_rows_others"] = int(gro)
                if grtot is not None:
                    prof["git_rows_total"] = int(grtot)
                if gps is not None:
                    prof["git_pathspecs"] = int(gps)
                if gpc is not None:
                    prof["git_pathspecs_capped"] = bool(gpc)
            print(_json.dumps(prof), file=sys.stderr)
        except Exception:
            pass
    else:
        files = await indexing_coordinator._discover_files(  # type: ignore[attr-defined]
            base_dir, processed_patterns, exclude_patterns
        )

    # Defensive: ensure simulate exactly mirrors real-flow ignore decisions.
    # If no git repos are present and gitignore source is active, build a
    # workspace-root engine that collects .gitignore rules from the tree
    # (root + nested) and apply it once. This mirrors the sequential walker’s
    # behavior for non-repo overlays and fixes edge cases where discovery took
    # a path without the engine wired.
    try:
        sources = config.indexing.resolve_ignore_sources()
        if "gitignore" in (sources or []):
            from chunkhound.utils.ignore_engine import (
                detect_repo_roots as _detect_roots,
                build_ignore_engine as _build_root_engine,
            )
            roots = _detect_roots(base_dir, config.indexing.get_effective_config_excludes())
            if not roots:
                eng = _build_root_engine(
                    root=base_dir,
                    sources=["gitignore"],
                    chignore_file=getattr(config.indexing, "chignore_file", ".chignore"),
                    config_exclude=config.indexing.get_effective_config_excludes(),
                )
                files = [p for p in files if not eng.matches(p, is_dir=False)]
    except Exception:
        # Soft-fail; simulate is best-effort
        pass

    # Respect config_file_size_threshold_kb for structured config languages to mirror real indexing
    try:
        from chunkhound.core.types.common import Language as _Lang
        if getattr(config, "indexing", None) is not None:
            _thr = getattr(config.indexing, "config_file_size_threshold_kb", 20)
            try:
                threshold_kb = int(_thr) if _thr is not None else 20
            except Exception:
                threshold_kb = 20
        else:
            threshold_kb = 20
    except Exception:
        threshold_kb = 20
    if threshold_kb is not None and threshold_kb > 0:
        filtered: list[Path] = []
        for p in files:
            try:
                lang = _Lang.from_file_extension(p)
                if getattr(lang, "is_structured_config_language", False):
                    try:
                        if (p.stat().st_size / 1024.0) > float(threshold_kb):
                            # Skip oversized structured config/data files
                            continue
                    except Exception:
                        pass
            except Exception:
                pass
            filtered.append(p)
        files = filtered

    # Gather sizes and relative paths
    items: list[tuple[str, int]] = []
    for p in files:
        try:
            st = p.stat()
            size = int(st.st_size)
        except Exception:
            size = 0
        rel = p.resolve().relative_to(base_dir).as_posix()
        items.append((rel, size))

    # Sort
    sort_mode = getattr(args, "sort", "path") or "path"
    if sort_mode == "size":
        items.sort(key=lambda x: (x[1], x[0]))
    elif sort_mode == "size_desc":
        items.sort(key=lambda x: (-x[1], x[0]))
    else:
        items.sort(key=lambda x: x[0])

    import json as _json
    if getattr(args, "json", False):
        try:
            print(
                _json.dumps(
                    {"files": [{"path": rel, "size_bytes": size} for rel, size in items]},
                    indent=2,
                )
            )
        except BrokenPipeError:
            # Common when piping to `head`; exit without noisy stacktrace
            try:
                sys.stdout.close()
            except Exception:
                pass
            return
    else:
        show_sizes = bool(getattr(args, "show_sizes", False))
        try:
            if show_sizes:
                for rel, size in items:
                    print(f"{size:>10}  {rel}")
            else:
                for rel, _ in items:
                    print(rel)
        except BrokenPipeError:
            # Allow piping to tools that close early without stacktrace
            try:
                sys.stdout.close()
            except Exception:
                pass
            return


async def _check_ignores(args: argparse.Namespace, config: Config) -> None:
    """Compare ChunkHound ignore decisions with a sentinel (currently: Git)."""
    base_dir = Path(args.path).resolve() if hasattr(args, "path") else Path.cwd().resolve()

    vs = getattr(args, "vs", "git") or "git"
    if vs != "git":
        print(f"Unsupported --vs value: {vs}", file=sys.stderr)
        sys.exit(2)

    # Helper functions (inlined to keep this feature local to index command)
    def _nearest_repo_root(path: Path, stop: Path) -> Path | None:
        p = path.resolve()
        stop = stop.resolve()
        while True:
            if (p / ".git").exists():
                return p
            if p == stop or p.parent == p:
                return None
            p = p.parent

    def _git_ignored(repo_root: Path, rel_path: str) -> bool:
        try:
            from chunkhound.utils.git_safe import run_git
            proc = run_git(["check-ignore", "-q", "--no-index", rel_path], cwd=repo_root, timeout_s=5.0)
            return proc.returncode == 0
        except Exception:
            return False

    def _ch_ignored(root: Path, file_path: Path) -> bool:
        try:
            from chunkhound.utils.ignore_engine import build_repo_aware_ignore_engine

            sources = config.indexing.resolve_ignore_sources()
            cfg_ex = config.indexing.get_effective_config_excludes()
            chf = config.indexing.chignore_file  # deprecated; ignored
            eng = build_repo_aware_ignore_engine(root, sources, chf, cfg_ex)
            return bool(eng.matches(file_path, is_dir=False))
        except Exception:
            return False

    # Collect candidate files (intentionally unpruned; we care about diffs)
    candidates: list[Path] = []
    for p in base_dir.rglob("*"):
        if p.is_file():
            candidates.append(p)

    mismatches: list[dict[str, Any]] = []
    for fp in candidates:
        repo = _nearest_repo_root(fp.parent, base_dir) or base_dir
        try:
            rel = fp.resolve().relative_to(repo if repo else base_dir).as_posix()
        except Exception:
            rel = fp.name
        git_decision = _git_ignored(repo, rel) if repo else False
        ch_decision = _ch_ignored(base_dir, fp)
        if git_decision != ch_decision:
            mismatches.append({
                "path": fp.resolve().relative_to(base_dir).as_posix(),
                "git": git_decision,
                "ch": ch_decision,
            })

    import json as _json
    report = {"mismatches": mismatches, "total": len(candidates), "base": base_dir.as_posix()}
    if getattr(args, "json", False):
        try:
            print(_json.dumps(report, indent=2))
        except BrokenPipeError:
            try:
                sys.stdout.close()
            except Exception:
                pass
            return
    else:
        try:
            print(f"Base: {report['base']}")
            print(f"Paths scanned: {report['total']}")
            print(f"Mismatches: {len(mismatches)}")
            for m in mismatches[:20]:
                print(f" - {m['path']}: CH={m['ch']} Git={m['git']}")
        except BrokenPipeError:
            # Allow piping to tools that close early without stacktrace
            try:
                sys.stdout.close()
            except Exception:
                pass
            return
