"""Batch file processor for parallel processing across CPU cores.

# FILE_CONTEXT: Worker function for ProcessPoolExecutor to parse files in parallel
# ROLE: Performs CPU-bound read→parse→chunk pipeline independently per batch
# CRITICAL: Must be picklable (top-level function, serializable arguments)
"""

import os
import multiprocessing
from multiprocessing.connection import Connection
from dataclasses import dataclass
from pathlib import Path

from chunkhound.core.detection import detect_language
from chunkhound.core.types.common import FileId, Language
from time import perf_counter

def _dbg_log(msg: str, operation: str | None = None, duration_ms: float | None = None) -> None:
    """Log debug/performance information.

    Uses loguru if available (for performance logging), otherwise falls back to file logging.
    """
    import os
    if os.environ.get("CHUNKHOUND_SUPPRESS_CONSOLE_LOGGING"):
        return

    try:
        from loguru import logger

        # Check if we have a performance logger configured
        if operation and duration_ms is not None:
            # This is a performance timing log
            logger.bind(operation=operation, duration_ms=duration_ms).info(msg)
        else:
            # Regular debug log
            logger.debug(msg)
    except ImportError:
        # Fallback to original file logging for compatibility
        try:
            import os, datetime
            path = os.getenv("CHUNKHOUND_DEBUG_FILE")
            if not path:
                return
            ts = datetime.datetime.now().isoformat()
            pid = os.getpid()
            with open(path, "a", encoding="utf-8", errors="ignore") as f:
                f.write(f"[{ts}][PID {pid}] {msg}\n")
        except Exception:
            pass
from chunkhound.parsers.parser_factory import create_parser_for_language


@dataclass
class ParsedFileResult:
    """Result from processing a single file in a batch."""

    file_path: Path
    chunks: list[dict]
    language: Language
    file_size: int
    file_mtime: float
    status: str
    error: str | None = None
    content_hash: str | None = None


def _parse_file_worker(file_path_str: str, language_value: str, conn: Connection) -> None:
    """Child-process worker to parse a single file and send results via pipe.

    Using a dedicated process lets us enforce a strict wall-clock timeout by
    terminating the child when exceeded, without risking stuck threads.
    """
    try:
        # Local imports to keep worker picklable and light
        from pathlib import Path as _Path
        from chunkhound.core.types.common import Language as _Language, FileId as _FileId
        from chunkhound.parsers.parser_factory import create_parser_for_language as _create

        language = _Language.from_string(language_value)
        parser = _create(language)
        if not parser:
            conn.send(("error", f"No parser available for {language}"))
            return

        chunks = parser.parse_file(_Path(file_path_str), _FileId(0))
        chunks_data = [chunk.to_dict() for chunk in chunks]
        conn.send(("ok", chunks_data))
    except Exception as e:  # pragma: no cover - safety net
        try:
            conn.send(("error", str(e)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _parse_file_with_timeout(
    file_path: Path, language: Language, timeout_s: float
) -> tuple[str, list[dict] | str | None]:
    """Parse a file in a child process with a wall-clock timeout.

    Returns a tuple of (status, payload):
    - ("success", list_of_chunk_dicts)
    - ("error", error_message)
    - ("timeout", None)
    """
    # Use spawn context for safety (works on all platforms)
    # Use spawn for safety; within worker processes this is still safe
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(
        target=_parse_file_worker,
        args=(str(file_path), language.value, child_conn),
        daemon=True,
    )
    _dbg_log(f"TIMEOUT-SPAWN start: {file_path}")
    p.start()
    # Close our reference to the child side in the parent
    try:
        child_conn.close()
    except Exception:
        pass

    try:
        if parent_conn.poll(timeout_s):
            status, payload = parent_conn.recv()
            # Ensure process exits
            p.join(timeout=0.5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.5)
            if status == "ok":
                _dbg_log(f"TIMEOUT-DONE success: {file_path}")
                return ("success", payload)
            else:
                _dbg_log(f"TIMEOUT-DONE error: {file_path} -> {payload}")
                return ("error", payload)
        else:
            # Timeout - terminate the child process cleanly
            p.terminate()
            p.join(timeout=0.5)
            _dbg_log(f"TIMEOUT-KILL: {file_path} after {timeout_s}s")
            return ("timeout", None)
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass


# Semaphore to cap concurrent timeout children per worker process
_timeout_semaphore: object | None = None


def _init_timeout_semaphore(max_concurrent: int) -> None:
    """Initialize a shared semaphore limiting concurrent timeout children in a worker."""
    global _timeout_semaphore
    if max_concurrent and _timeout_semaphore is None:
        try:
            ctx = multiprocessing.get_context("spawn")
            _timeout_semaphore = ctx.Semaphore(int(max_concurrent))
        except Exception:
            # CI sandboxes may deny SemLock; fall back to unlimited
            _timeout_semaphore = None


def process_file_batch(
    file_info_list: list[Path] | list[tuple[Path, str | None]],
    config_dict: dict,
) -> list[ParsedFileResult]:
    """Process a batch of files in a worker process.

    This function runs in a separate process via ProcessPoolExecutor.
    Performs the complete read→parse→chunk pipeline for all files in the batch.

    Args:
        file_paths: List of file paths to process in this batch
        config_dict: Serialized configuration dictionary for parser initialization

    Returns:
        List of ParsedFileResult objects with parsed chunks and metadata
    """
    results = []

    # Set up performance logging in worker process if configured
    if "performance_log_path" in config_dict:
        try:
            from loguru import logger
            perf_path = config_dict["performance_log_path"]
            logger.add(
                perf_path,
                level="INFO",
                rotation="50 MB",
                retention="1 month",
                format="{time:YYYY-MM-DD HH:mm:ss} | {extra[operation]: <20} | {extra[duration_ms]: >8.1f}ms | {message}",
                filter=lambda record: record["extra"].get("operation") is not None,
                encoding="utf-8",
            )
        except Exception:
            # If performance logging setup fails, continue without it
            pass

    # Read timeout config once
    timeout_s = float(config_dict.get("per_file_timeout_seconds", 0.0) or 0.0)
    # Respect explicit 0 so users can apply timeout to all file sizes.
    # Do not coerce 0 to the default.
    try:
        timeout_min_kb = int(config_dict.get("per_file_timeout_min_size_kb", 128))
    except Exception:
        timeout_min_kb = 128
    try:
        max_concurrent_timeouts = int(config_dict.get("max_concurrent_timeouts", 32))
    except Exception:
        max_concurrent_timeouts = 32
    _init_timeout_semaphore(max_concurrent_timeouts)

    # Normalize inputs to list of (Path, hash)
    normalized: list[tuple[Path, str | None]] = []
    for item in file_info_list:
        if isinstance(item, tuple):
            normalized.append(item)
        else:
            normalized.append((item, None))

    for file_path, precomputed_hash in normalized:
        try:
            # Get file metadata
            file_stat = os.stat(file_path)

            # Detect language (content-aware for ambiguous extensions)
            language = detect_language(file_path)
            _dbg_log(
                f"START file={file_path} size_kb={file_stat.st_size/1024:.1f} lang={language.value} "
                f"tmo_s={timeout_s} min_kb={timeout_min_kb} threshold_kb={config_dict.get('config_file_size_threshold_kb')}"
            )
            t0 = perf_counter()
            if language == Language.UNKNOWN:
                results.append(
                    ParsedFileResult(
                        file_path=file_path,
                        chunks=[],
                        language=language,
                        file_size=file_stat.st_size,
                        file_mtime=file_stat.st_mtime,
                        status="skipped",
                        error="Unknown file type",
                    )
                )
                duration_ms = (perf_counter() - t0) * 1000
                _dbg_log(f"END   file={file_path} status=skipped reason=unknown_type", "file_parsing", duration_ms)
                continue

            # Skip large config/data files (config files are typically < 20KB)
            if language.is_structured_config_language:
                file_size_kb = file_stat.st_size / 1024
                threshold_kb = config_dict.get("config_file_size_threshold_kb", 20)
                # Treat <=0 as disabled (per documentation)
                if isinstance(threshold_kb, (int, float)) and threshold_kb <= 0:
                    threshold_kb = None
                if threshold_kb is not None and file_size_kb > threshold_kb:
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            status="skipped",
                            error="large_config_file",
                        )
                    )
                    duration_ms = (perf_counter() - t0) * 1000
                    _dbg_log(f"END   file={file_path} status=skipped reason=large_config_file", "file_parsing", duration_ms)
                    continue

            # Parse file and generate chunks (with optional per-file timeout)
            if timeout_s > 0 and ((file_stat.st_size / 1024) >= timeout_min_kb):
                if _timeout_semaphore is not None:
                    with _timeout_semaphore:
                        status, payload = _parse_file_with_timeout(
                            file_path, language, timeout_s
                        )
                else:
                    status, payload = _parse_file_with_timeout(
                        file_path, language, timeout_s
                    )
                if status == "timeout":
                    # Defer user notification to final summary; avoid live console noise
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            content_hash=precomputed_hash,
                            status="skipped",
                            error="timeout",
                        )
                    )
                    duration_ms = (perf_counter() - t0) * 1000
                    _dbg_log(f"END   file={file_path} status=skipped reason=timeout", "file_parsing", duration_ms)
                    continue
                elif status == "error":
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            content_hash=precomputed_hash,
                            status="error",
                            error=str(payload),
                        )
                    )
                    duration_ms = (perf_counter() - t0) * 1000
                    _dbg_log(f"END   file={file_path} status=error reason={payload}", "file_parsing", duration_ms)
                    continue
                else:
                    chunks_data = payload if isinstance(payload, list) else []
            else:
                # No timeout path (original behavior)
                parser = create_parser_for_language(language)
                if not parser:
                    results.append(
                        ParsedFileResult(
                            file_path=file_path,
                            chunks=[],
                            language=language,
                            file_size=file_stat.st_size,
                            file_mtime=file_stat.st_mtime,
                            status="error",
                            error=f"No parser available for {language}",
                        )
                    )
                    continue

                # Note: FileId(0) is placeholder - actual ID assigned during storage
                chunks = parser.parse_file(file_path, FileId(0))
                chunks_data = [chunk.to_dict() for chunk in chunks]

            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=chunks_data,
                    language=language,
                    file_size=file_stat.st_size,
                    file_mtime=file_stat.st_mtime,
                    content_hash=precomputed_hash,
                    status="success",
                )
            )
            duration_ms = (perf_counter() - t0) * 1000
            _dbg_log(f"END   file={file_path} status=success chunks={len(chunks_data)}", "file_parsing", duration_ms)

        except Exception as e:
            # Capture errors but continue processing other files in batch
            results.append(
                ParsedFileResult(
                    file_path=file_path,
                    chunks=[],
                    language=Language.UNKNOWN,
                    file_size=0,
                    file_mtime=0.0,
                    status="error",
                    error=str(e),
                )
            )
            _dbg_log(f"END   file={file_path} status=error-unhandled reason={e}")

    return results
