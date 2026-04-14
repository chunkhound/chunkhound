from __future__ import annotations

import asyncio
import gc
import threading
import time
from collections.abc import Awaitable
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from chunkhound.core.utils.path_utils import normalize_realtime_path
from chunkhound.providers.database.duckdb_provider import DuckDBTransactionConflictError
from chunkhound.services.realtime_path_filter import RealtimePathFilter
from chunkhound.utils.windows_constants import IS_WINDOWS

from .events import RealtimeMutation, SimpleEventHandler, normalize_file_path

class RealtimePipelineMixin:
    def _emit_status_update(self) -> None:
        try:
            if self._status_callback:
                self._status_callback(self._build_health_snapshot())
        except Exception:
            # Status plumbing must never affect runtime behavior.
            pass
    def _record_source_event(self, event_type: str, file_path: Path | str) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        self._last_source_event_at = self._utc_now()
        self._last_source_event_type = event_type
        self._last_source_event_path = normalized_path
        self._track_event_pressure(
            normalized_path,
            event_type=event_type,
            count_event=True,
        )
    def _record_accepted_event(self, event_type: str, file_path: Path | str) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        source_generation = self._advance_source_generation(normalized_path)
        self._refresh_pending_change_generation(normalized_path, source_generation)
        self._last_accepted_event_at = self._utc_now()
        self._last_accepted_event_type = event_type
        self._last_accepted_event_path = normalized_path
        self._track_event_pressure(
            normalized_path,
            event_type=event_type,
            scope="included",
        )
        self._emit_status_update()
    def _record_filtered_event(self, event_type: str, file_path: Path | str) -> None:
        self._filtered_event_count += 1
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="excluded",
        )
        self._emit_status_update()
    def _record_translation_error(self) -> None:
        self._translation_error_count += 1
    def _record_duplicate_suppression(
        self, event_type: str, file_path: Path | str
    ) -> None:
        self._suppressed_duplicate_count += 1
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="included",
        )
        self._emit_status_update()
    def _record_processing_started(self, file_path: Path | str) -> None:
        self._active_processing_count += 1
        self._last_processing_started_at = self._utc_now()
        self._last_processing_started_path = str(file_path)
        self._emit_status_update()
    def _record_processing_finished(
        self, file_path: Path | str, *, completed: bool
    ) -> None:
        if completed:
            self._last_processing_completed_at = self._utc_now()
            self._last_processing_completed_path = str(file_path)
        if self._active_processing_count > 0:
            self._active_processing_count -= 1
        self._emit_status_update()
    def _record_processing_error(self) -> None:
        self._processing_error_count += 1
    def _start_transient_task(
        self, awaitable: Awaitable[Any]
    ) -> asyncio.Task[Any]:
        return self._transient_tasks.create_task(awaitable)
    def _normalize_mutation_path(self, file_path: Path | str) -> Path:
        path_obj = Path(file_path)
        base_dir = self.watch_path or getattr(self.config, "target_dir", None)
        return normalize_realtime_path(
            path_obj, base_dir if isinstance(base_dir, Path) else None
        )
    @classmethod
    def _mutation_priority(cls, operation: str) -> int:
        return cls._MUTATION_PRIORITIES.get(
            operation, cls._MUTATION_PRIORITIES["change"]
        )
    @classmethod
    def _normalize_add_priority(cls, priority: str) -> tuple[str, bool]:
        if priority == "change":
            return "change", True
        if priority in {"priority", "scan"}:
            return "scan", False
        if priority == "embed":
            return "embed", False
        if priority in cls._MUTATION_PRIORITIES:
            return priority, False
        return "change", False
    @staticmethod
    def _status_operation(operation: str) -> str:
        if operation == "scan":
            return "change"
        return operation
    def _build_mutation(
        self,
        operation: str,
        file_path: Path | str,
        retry_count: int = 0,
        source_generation: int | None = None,
        first_queued_at: str | None = None,
    ) -> RealtimeMutation:
        self._next_mutation_id += 1
        return RealtimeMutation(
            mutation_id=self._next_mutation_id,
            operation=operation,
            path=self._normalize_mutation_path(file_path),
            first_queued_at=first_queued_at or self._utc_now(),
            retry_count=retry_count,
            source_generation=source_generation,
        )
    def _advance_source_generation(self, file_path: Path | str) -> int:
        normalized_path = str(self._normalize_mutation_path(file_path))
        self._next_source_generation += 1
        self._latest_source_generation_by_path[normalized_path] = (
            self._next_source_generation
        )
        return self._next_source_generation
    def _current_source_generation(self, file_path: Path | str) -> int | None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        return self._latest_source_generation_by_path.get(normalized_path)
    def _refresh_pending_change_generation(
        self, file_path: Path | str, source_generation: int | None = None
    ) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        current_generation = source_generation
        if current_generation is None:
            current_generation = self._latest_source_generation_by_path.get(
                normalized_path
            )
        if current_generation is None:
            return

        key = ("change", normalized_path)
        existing = self._pending_mutations.get(key)
        if existing is None or existing.source_generation == current_generation:
            return
        self._pending_mutations[key] = replace(
            existing,
            source_generation=current_generation,
        )
    def _mark_coalesced_change(
        self, file_path: Path | str, event_type: str = "modified"
    ) -> None:
        self._refresh_pending_change_generation(file_path)
        self._track_event_pressure(
            file_path,
            event_type=event_type,
            scope="included",
            count_coalesced=True,
        )
    def _reserve_change_follow_up(
        self, file_path: Path | str, source_generation: int | None = None
    ) -> None:
        normalized_path = str(self._normalize_mutation_path(file_path))
        current_generation = source_generation
        if current_generation is None:
            current_generation = self._current_source_generation(normalized_path)
        if current_generation is None:
            return

        existing_generation = self._reserved_follow_up_change_generations.get(
            normalized_path
        )
        if existing_generation is None or current_generation > existing_generation:
            self._reserved_follow_up_change_generations[normalized_path] = (
                current_generation
            )
    def _mutation_for_processing(self, mutation: RealtimeMutation) -> RealtimeMutation:
        if mutation.operation not in {"change", "scan"}:
            return mutation
        current_generation = self._current_source_generation(mutation.path)
        if (
            current_generation is None
            or mutation.source_generation == current_generation
        ):
            return mutation
        return replace(mutation, source_generation=current_generation)
    def _delete_mutation_is_stale(self, mutation: RealtimeMutation) -> bool:
        if mutation.source_generation is None:
            return False

        current_generation = self._current_source_generation(mutation.path)
        return (
            current_generation is not None
            and current_generation > mutation.source_generation
        )
    @staticmethod
    def _pending_mutation_key(mutation: RealtimeMutation) -> tuple[str, str]:
        return (mutation.operation, str(mutation.path))
    def _owns_pending_mutation(self, mutation: RealtimeMutation) -> bool:
        current = self._pending_mutations.get(self._pending_mutation_key(mutation))
        return current is not None and current.mutation_id == mutation.mutation_id
    def _delete_mutation_supersedes_existing(
        self, mutation: RealtimeMutation, existing: RealtimeMutation
    ) -> bool:
        if mutation.operation != "delete" or existing.operation != "delete":
            return False

        incoming_generation = mutation.source_generation
        existing_generation = existing.source_generation

        if incoming_generation is not None and existing_generation is None:
            return True
        if incoming_generation is None:
            return False
        if existing_generation is None:
            return True
        if incoming_generation > existing_generation:
            return True
        if (
            incoming_generation == existing_generation
            and mutation.retry_count < existing.retry_count
        ):
            return True
        return False
    def _register_pending_mutation(self, mutation: RealtimeMutation) -> bool:
        key = self._pending_mutation_key(mutation)
        existing = self._pending_mutations.get(key)
        if existing is not None:
            if self._delete_mutation_supersedes_existing(mutation, existing):
                self._pending_mutations[key] = mutation
                self._debug(
                    "replaced pending delete ownership "
                    f"path={mutation.path} old_generation="
                    f"{existing.source_generation} new_generation="
                    f"{mutation.source_generation}"
                )
                self._emit_status_update()
                return True
            return False

        self._pending_mutations[key] = mutation
        path_key = str(mutation.path)
        self._pending_path_counts[path_key] = (
            self._pending_path_counts.get(path_key, 0) + 1
        )
        self.pending_files.add(mutation.path)
        return True
    def _release_pending_mutation(self, mutation: RealtimeMutation) -> None:
        key = self._pending_mutation_key(mutation)
        current = self._pending_mutations.get(key)
        if current is None or current.mutation_id != mutation.mutation_id:
            return
        self._pending_mutations.pop(key, None)

        path_key = str(mutation.path)
        remaining = self._pending_path_counts.get(path_key, 0) - 1
        if remaining <= 0:
            self._pending_path_counts.pop(path_key, None)
            self.pending_files.discard(mutation.path)
            return
        self._pending_path_counts[path_key] = remaining
    async def _enqueue_mutation(
        self, mutation: RealtimeMutation, *, register: bool = True
    ) -> bool:
        if register and not self._register_pending_mutation(mutation):
            return False
        if not register and not self._owns_pending_mutation(mutation):
            return False

        self._queue_sequence += 1
        await self.file_queue.put(
            (
                self._mutation_priority(mutation.operation),
                self._queue_sequence,
                mutation,
            )
        )
        self._debug(
            "queued "
            f"{mutation.path} operation={mutation.operation} "
            f"retry={mutation.retry_count}"
        )
        self._emit_status_update()
        return True
    async def _retry_mutation_after_delay(
        self, mutation: RealtimeMutation, delay_seconds: float
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            if not self._owns_pending_mutation(mutation):
                self._debug(
                    "dropped superseded delete retry "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                return
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "dropped stale delete retry "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                self._release_pending_mutation(mutation)
                self._emit_status_update()
                return
            await self._enqueue_mutation(mutation, register=False)
        except asyncio.CancelledError:
            self._release_pending_mutation(mutation)
            raise
        except Exception:
            self._release_pending_mutation(mutation)
            raise
    def _schedule_delete_retry(self, mutation: RealtimeMutation) -> bool:
        if self._delete_mutation_is_stale(mutation):
            self._debug(
                "skipped stale delete retry scheduling "
                f"path={mutation.path} source_generation={mutation.source_generation}"
            )
            return True

        if mutation.retry_count >= self._DELETE_CONFLICT_MAX_RETRIES:
            return False

        retry_mutation = self._build_mutation(
            "delete",
            mutation.path,
            retry_count=mutation.retry_count + 1,
            source_generation=mutation.source_generation,
            first_queued_at=mutation.first_queued_at,
        )
        if not self._register_pending_mutation(retry_mutation):
            return True

        delay_seconds = self._DELETE_CONFLICT_BASE_RETRY_DELAY_SECONDS * (
            2**mutation.retry_count
        )
        self._start_transient_task(
            self._retry_mutation_after_delay(retry_mutation, delay_seconds)
        )
        self._debug(
            "scheduled delete retry "
            f"{retry_mutation.retry_count} for {retry_mutation.path} "
            f"after {delay_seconds:.2f}s"
        )
        self._emit_status_update()
        return True
    def _prepare_dequeued_mutation(
        self, mutation: RealtimeMutation
    ) -> tuple[RealtimeMutation, bool]:
        mutation = self._mutation_for_processing(mutation)
        owned_when_dequeued = self._owns_pending_mutation(mutation)
        self._release_pending_mutation(mutation)
        return mutation, owned_when_dequeued
    def _collect_delete_batch(
        self,
        mutation: RealtimeMutation,
        *,
        owned_when_dequeued: bool,
    ) -> tuple[
        list[tuple[RealtimeMutation, bool]],
        tuple[RealtimeMutation, bool] | None,
    ]:
        batch = [(mutation, owned_when_dequeued)]
        buffered_mutation: tuple[RealtimeMutation, bool] | None = None

        while len(batch) < self._DELETE_BATCH_SIZE:
            try:
                _, _, queued_mutation = self.file_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            prepared_mutation, prepared_owned = self._prepare_dequeued_mutation(
                queued_mutation
            )
            if prepared_mutation.operation != "delete":
                buffered_mutation = (prepared_mutation, prepared_owned)
                break

            batch.append((prepared_mutation, prepared_owned))

        return batch, buffered_mutation
    async def _queue_follow_up_change(
        self,
        file_path: Path | str,
        *,
        source_generation: int,
        first_queued_at: str | None = None,
    ) -> bool:
        follow_up = self._build_mutation(
            "change",
            file_path,
            source_generation=source_generation,
            first_queued_at=first_queued_at,
        )
        if not self._register_pending_mutation(follow_up):
            self._refresh_pending_change_generation(file_path, source_generation)
            return False
        return await self._enqueue_mutation(follow_up, register=False)
    async def _drain_reserved_follow_up_change(
        self,
        file_path: Path | str,
        *,
        normalized_path: str | None,
        processing_error: Exception | None = None,
    ) -> None:
        if normalized_path is None:
            return

        follow_up_generation = self._reserved_follow_up_change_generations.pop(
            normalized_path,
            None,
        )
        if follow_up_generation is None:
            return

        try:
            await self._queue_follow_up_change(
                file_path,
                source_generation=follow_up_generation,
            )
        except Exception as error:
            self._reserve_change_follow_up(file_path, follow_up_generation)
            if processing_error is not None:
                logger.error(
                    "Failed to requeue reserved follow-up change for "
                    f"{file_path} after processing error {processing_error}: {error}"
                )
                return
            raise
    @staticmethod
    def _overflow_drop_label(drop_count: int) -> str:
        return "event" if drop_count == 1 else "events"
    def _record_event_queue_overflow(
        self, event_type: str, file_path: Path, *, timestamp: str
    ) -> None:
        if self._event_queue_overflow_state == "idle":
            self._event_queue_overflow_state = "reconciling"
            self._event_queue_overflow_burst_count += 1
            self._event_queue_overflow_current_burst_dropped = 1
            self._event_queue_overflow_sample_event_type = event_type
            self._event_queue_overflow_sample_file_path = str(file_path)
            self._event_queue_overflow_last_started_at = timestamp
            message = (
                "Realtime event queue overflow detected; entering reconciliation mode."
            )
            logger.warning(message)
            self._set_warning(message)
            self._start_transient_task(
                self.request_resync(
                    "event_queue_overflow",
                    {
                        "event_type": event_type,
                        "file_path": str(file_path),
                        "drop_reason": "queue_full",
                        "overflow_burst": self._event_queue_overflow_burst_count,
                        "dropped_events": (
                            self._event_queue_overflow_current_burst_dropped
                        ),
                    },
                )
            )
            return

        self._event_queue_overflow_current_burst_dropped += 1
    def _complete_event_queue_overflow_burst(self, *, success: bool) -> None:
        if self._event_queue_overflow_state == "idle":
            return

        drop_count = max(self._event_queue_overflow_current_burst_dropped, 1)
        self._event_queue_overflow_last_burst_dropped = drop_count

        if success:
            self._event_queue_overflow_state = "idle"
            self._event_queue_overflow_current_burst_dropped = 0
            self._event_queue_overflow_last_cleared_at = self._utc_now()
            message = (
                "Realtime event queue overflow recovered after dropping "
                f"{drop_count} {self._overflow_drop_label(drop_count)}."
            )
            logger.info(message)
            self._set_warning(message)
            return

        self._event_queue_overflow_state = "failed"
        message = (
            "Realtime event queue overflow reconciliation failed after dropping "
            f"{drop_count} {self._overflow_drop_label(drop_count)}."
        )
        logger.warning(message)
        self._set_warning(message)
    def _handle_queue_result(
        self, event_type: str, file_path: Path, accepted: bool, reason: str | None
    ) -> None:
        timestamp = self._utc_now()
        self._event_queue_last_event_type = event_type
        self._event_queue_last_file_path = str(file_path)

        if accepted:
            self._event_queue_accepted += 1
            self._event_queue_last_enqueued_at = timestamp
            self._record_accepted_event(event_type, file_path)
        else:
            self._event_queue_dropped += 1
            self._event_queue_last_reason = reason
            self._event_queue_last_dropped_at = timestamp
            self._track_event_pressure(
                file_path,
                event_type=event_type,
                scope="included",
            )
            if reason == "queue_full":
                self._record_event_queue_overflow(
                    event_type,
                    file_path,
                    timestamp=timestamp,
                )
            else:
                self._set_warning(
                    f"realtime event dropped ({reason or 'unknown_reason'})"
                )

        self._emit_status_update()
    async def request_resync(
        self, reason: str, details: dict[str, Any] | None = None
    ) -> bool:
        """Request a debounced backend-neutral reconciliation scan."""
        self._needs_resync = True
        self._resync_request_count += 1
        self._last_resync_reason = reason
        self._last_resync_details = details
        self._last_resync_requested_at = self._utc_now()
        self._last_resync_request_monotonic = time.monotonic()
        self._emit_status_update()

        if self._resync_dispatch_task and not self._resync_dispatch_task.done():
            return False

        self._resync_dispatch_task = self._start_transient_task(
            self._dispatch_resync()
        )
        return True
    async def _dispatch_resync(self) -> None:
        """Coalesce resync requests and run the callback on the trailing edge."""
        try:
            while True:
                requested_at = self._last_resync_request_monotonic
                await asyncio.sleep(self._RESYNC_DEBOUNCE_SECONDS)
                if requested_at == self._last_resync_request_monotonic:
                    break

            reason = self._last_resync_reason or "unspecified"
            details = self._last_resync_details
            callback = self._resync_callback
            if callback is None:
                self._last_resync_error = "No resync callback configured"
                self._complete_event_queue_overflow_burst(success=False)
                self._set_error(self._last_resync_error)
                return

            while True:
                started_request_at = self._last_resync_request_monotonic
                self._resync_in_progress = True
                self._last_resync_started_at = self._utc_now()
                self._last_resync_error = None
                self._emit_status_update()

                try:
                    result = await callback(reason, details)
                    callback_error = self._resync_callback_error(result)
                    if callback_error is not None:
                        raise RuntimeError(callback_error)
                    self._needs_resync = False
                    self._resync_performed_count += 1
                    self._last_resync_completed_at = self._utc_now()
                    self._complete_event_queue_overflow_burst(success=True)
                    if self._service_state not in {"stopping", "stopped"}:
                        self._clear_resync_error_state()
                except Exception as e:
                    self._last_resync_error = str(e)
                    self._complete_event_queue_overflow_burst(success=False)
                    self._set_error(f"Realtime resync failed: {e}")
                    break
                finally:
                    self._resync_in_progress = False
                    self._emit_status_update()

                if started_request_at == self._last_resync_request_monotonic:
                    break
                reason = self._last_resync_reason or reason
                details = self._last_resync_details
        finally:
            self._resync_dispatch_task = None
    def _polling_snapshot(
        self, watch_path: Path
    ) -> tuple[dict[Path, tuple[int, int]], int, bool]:
        """Collect a filesystem snapshot off the event loop for polling mode."""
        current_files: dict[Path, tuple[int, int]] = {}
        files_checked = 0
        truncated = False
        simple_handler = SimpleEventHandler(
            None, self.config, None, root_path=watch_path
        )

        rglob_gen = watch_path.rglob("*")
        try:
            for file_path in rglob_gen:
                try:
                    if not file_path.is_file():
                        continue

                    files_checked += 1
                    if simple_handler._should_index(file_path):
                        try:
                            stat_result = file_path.stat()
                        except OSError:
                            continue

                        current_files[file_path] = (
                            stat_result.st_mtime_ns,
                            stat_result.st_size,
                        )

                    if files_checked >= 5000:
                        truncated = True
                        break
                except (OSError, PermissionError):
                    continue
        finally:
            rglob_gen.close()

        return current_files, files_checked, truncated
    def _collect_supported_files(self, dir_path: Path) -> list[Path]:
        """Collect supported files in a directory off the event loop."""
        simple_handler = SimpleEventHandler(
            None,
            self.config,
            None,
            root_path=self._path_filter_root(dir_path),
        )
        supported_files: list[Path] = []

        for file_path in dir_path.rglob("*"):
            try:
                if file_path.is_file() and simple_handler._should_index(file_path):
                    supported_files.append(file_path)
            except (OSError, PermissionError):
                continue

        return supported_files
    def _path_filter_root(self, fallback_path: Path | None = None) -> Path:
        """Return the logical workspace root for realtime scope decisions."""
        if self.watch_path is not None:
            return self.watch_path

        target_dir = getattr(self.config, "target_dir", None)
        if isinstance(target_dir, Path):
            return target_dir

        if fallback_path is not None:
            return fallback_path

        return Path.cwd()
    async def _polling_monitor(self, watch_path: Path) -> None:
        """Simple polling monitor for large directories."""
        logger.debug(f"Starting polling monitor for {watch_path}")
        self._debug(f"polling monitor active for {watch_path}")
        # Track both mtime and size so Windows polling catches overwrites that
        # fail to advance mtime reliably on CI filesystems.
        known_files: dict[Path, tuple[int, int]] = {}

        # Use a shorter interval during the first few seconds to ensure
        # freshly created files are detected quickly after startup/fallback.
        polling_start = time.time()

        try:
            while True:
                try:
                    current_files, files_checked, truncated = await asyncio.to_thread(
                        self._polling_snapshot, watch_path
                    )
                    self._last_poll_snapshot_at = self._utc_now()
                    self._last_poll_files_checked = files_checked
                    self._last_poll_snapshot_truncated = truncated
                    if truncated:
                        self._set_warning(
                            "Polling snapshot truncated after 5000 files to avoid "
                            "event-loop starvation"
                        )

                    for file_path, current_fingerprint in current_files.items():
                        if file_path not in known_files:
                            logger.debug(f"Polling detected new file: {file_path}")
                            self._debug(f"polling detected new file: {file_path}")
                            self._record_source_event("created", file_path)
                            accepted = await self.add_file(file_path, priority="change")
                            if accepted:
                                self._record_accepted_event("created", file_path)
                            else:
                                source_generation = self._advance_source_generation(
                                    file_path
                                )
                                self._refresh_pending_change_generation(
                                    file_path,
                                    source_generation,
                                )
                                self._reserve_change_follow_up(
                                    file_path,
                                    source_generation,
                                )
                        elif known_files[file_path] != current_fingerprint:
                            logger.debug(f"Polling detected modified file: {file_path}")
                            self._debug(f"polling detected modified file: {file_path}")
                            self._record_source_event("modified", file_path)
                            accepted = await self.add_file(file_path, priority="change")
                            if accepted:
                                self._record_accepted_event("modified", file_path)
                            else:
                                source_generation = self._advance_source_generation(
                                    file_path
                                )
                                self._refresh_pending_change_generation(
                                    file_path,
                                    source_generation,
                                )
                                self._reserve_change_follow_up(
                                    file_path,
                                    source_generation,
                                )

                    # Check for deleted files.
                    deleted = set(known_files.keys()) - set(current_files.keys())
                    for file_path in deleted:
                        logger.debug(f"Polling detected deleted file: {file_path}")
                        self._record_source_event("deleted", file_path)
                        self._record_accepted_event("deleted", file_path)
                        delete_source_generation: int | None = (
                            self._current_source_generation(file_path)
                        )
                        await self._enqueue_mutation(
                            self._build_mutation(
                                "delete",
                                file_path,
                                source_generation=delete_source_generation,
                            )
                        )
                        self._debug(f"polling detected deleted file: {file_path}")

                    known_files = current_files

                    # Adaptive poll interval: 0.5s for the first 60s, then 3s
                    # Extended fast polling window ensures reliable detection during
                    # multi-file test sequences on Windows CI where setup + indexing
                    # can consume the initial fast-polling budget
                    elapsed = time.time() - polling_start
                    interval = 0.5 if elapsed < 60.0 else 3.0
                    self._emit_status_update()
                    await asyncio.sleep(interval)

                except Exception as e:
                    logger.error(f"Polling monitor error: {e}")
                    self._set_error(f"Polling monitor error: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.debug("Polling monitor cancelled")
            raise
        finally:
            # Force cleanup of any lingering file handles on Windows
            gc.collect()
            logger.debug("Polling monitor stopped")
    async def add_file(self, file_path: Path, priority: str = "change") -> bool:
        """Add file to the realtime pipeline and report whether work was admitted."""
        operation, debounced = self._normalize_add_priority(priority)
        source_generation = (
            self._current_source_generation(file_path)
            if operation == "change"
            else None
        )
        mutation = self._build_mutation(
            operation,
            file_path,
            source_generation=source_generation,
        )
        file_str = str(mutation.path)
        if (
            operation == "change"
            and file_str in self._active_change_generations
            and ("change", file_str) not in self._pending_mutations
        ):
            self._reserve_change_follow_up(mutation.path, source_generation)
            self._mark_coalesced_change(mutation.path)
            self._emit_status_update()
            return False
        if debounced:
            if file_str in self._pending_debounce:
                # Keep the already-pending debounce horizon fresh.
                self._pending_debounce[file_str] = time.monotonic()
                self._mark_coalesced_change(mutation.path)
                self._emit_status_update()
                return False

        if not self._register_pending_mutation(mutation):
            if debounced:
                if file_str in self._pending_debounce:
                    # Keep the already-pending debounce horizon fresh.
                    self._pending_debounce[file_str] = time.monotonic()
                self._mark_coalesced_change(mutation.path)
                self._emit_status_update()
            return False

        # Simple debouncing for change events
        if debounced:
            file_str = str(mutation.path)
            self._pending_debounce[file_str] = time.monotonic()
            self._start_transient_task(self._debounced_add_file(mutation))
            self._debug(f"queued (debounced) {mutation.path} operation={operation}")
            self._emit_status_update()
            return True

        # Immediate mutations bypass debouncing.
        return await self._enqueue_mutation(mutation, register=False)
    async def _debounced_add_file(
        self, file_or_mutation: Path | RealtimeMutation, priority: str = "change"
    ) -> None:
        """Process file after debounce delay.

        Loops until the debounce window has been quiet for at least
        ``_debounce_delay``. If debounce state is cleared externally, release the
        registered mutation ownership so pending counts do not leak.
        """
        if isinstance(file_or_mutation, RealtimeMutation):
            mutation = file_or_mutation
        else:
            operation, _ = self._normalize_add_priority(priority)
            mutation = self._build_mutation(operation, file_or_mutation)
        file_str = str(mutation.path)
        remaining_delay = self._debounce_delay

        try:
            while True:
                await asyncio.sleep(remaining_delay)

                if file_str not in self._pending_debounce:
                    self._release_pending_mutation(mutation)
                    return

                last_update = self._pending_debounce[file_str]
                remaining_delay = self._debounce_delay - (
                    time.monotonic() - last_update
                )

                # Windows timer granularity can wake slightly before the debounce
                # horizon; retry instead of leaving the file stuck in pending state.
                if remaining_delay > 0:
                    continue
                break
        except asyncio.CancelledError:
            self._pending_debounce.pop(file_str, None)
            self._release_pending_mutation(mutation)
            raise

        del self._pending_debounce[file_str]
        current_mutation = self._pending_mutations.get(
            self._pending_mutation_key(mutation),
            mutation,
        )
        if not await self._enqueue_mutation(current_mutation, register=False):
            self._release_pending_mutation(current_mutation)
            return
        logger.debug(f"Processing debounced file: {current_mutation.path}")
        self._debug(f"processing debounced file: {current_mutation.path}")
        return
    async def _consume_events(self) -> None:
        """Simple event consumer - pure asyncio queue."""
        while True:
            try:
                # Get event from async queue with timeout
                try:
                    event_type, file_path = await asyncio.wait_for(
                        self.event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Normal timeout, continue to check if task should stop
                    continue

                # Layer 3: Event deduplication to prevent redundant processing.
                file_key = str(file_path)
                current_time = time.time()

                if file_key in self._recent_file_events:
                    last_event_type, last_event_time = self._recent_file_events[
                        file_key
                    ]
                    if (
                        last_event_type == event_type
                        and (current_time - last_event_time)
                        < self._EVENT_DEDUP_WINDOW_SECONDS
                    ):
                        logger.debug(
                            "Suppressing duplicate "
                            f"{event_type} event for {file_path} "
                            f"(within {self._EVENT_DEDUP_WINDOW_SECONDS}s window)"
                        )
                        self._debug(f"suppressed duplicate {event_type}: {file_path}")
                        self._record_duplicate_suppression(event_type, file_path)
                        self.event_queue.task_done()
                        continue

                # Record this event
                self._recent_file_events[file_key] = (event_type, current_time)

                # Cleanup old entries to keep dict bounded (max 1000 files)
                if len(self._recent_file_events) > 1000:
                    cutoff = current_time - self._EVENT_HISTORY_RETENTION_SECONDS
                    self._recent_file_events = {
                        k: v
                        for k, v in self._recent_file_events.items()
                        if v[1] > cutoff
                    }

                if event_type in ("created", "modified"):
                    # Use existing add_file method for deduplication and priority
                    await self.add_file(file_path, priority="change")
                    self._debug(f"event {event_type}: {file_path}")
                elif event_type == "deleted":
                    source_generation = self._current_source_generation(file_path)
                    await self._enqueue_mutation(
                        self._build_mutation(
                            "delete",
                            file_path,
                            source_generation=source_generation,
                        )
                    )
                    self._debug(f"event deleted: {file_path}")
                elif event_type == "dir_created":
                    await self._enqueue_mutation(
                        self._build_mutation("dir_index", file_path)
                    )
                    self._debug(f"event dir_created: {file_path}")
                elif event_type == "dir_deleted":
                    source_generation = self._current_source_generation(file_path)
                    await self._enqueue_mutation(
                        self._build_mutation(
                            "dir_delete",
                            file_path,
                            source_generation=source_generation,
                        )
                    )
                    self._debug(f"event dir_deleted: {file_path}")

                self.event_queue.task_done()

            except Exception as e:
                logger.error(f"Error consuming event: {e}")
                self._set_error(f"Error consuming realtime event: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
    async def remove_file(self, file_path: Path) -> None:
        """Remove file from database."""
        logger.debug(f"Removing file from database: {file_path}")
        await self.services.provider.delete_file_completely_async(str(file_path))
        self._debug(f"removed file from database: {file_path}")
        normalized = normalize_file_path(file_path)
        async with self._file_condition:
            self._removed_files.add(normalized)
            self._file_condition.notify_all()
    async def _add_directory_watch(self, dir_path: str) -> None:
        """Add a new directory to monitoring with recursive watching."""
        async with self.watch_lock:
            if dir_path not in self.watched_directories:
                if self.observer and self.event_handler:
                    self.observer.schedule(
                        self.event_handler,
                        dir_path,
                        recursive=True,  # Keep new directories recursively covered.
                    )
                    self.watched_directories.add(dir_path)
                    logger.debug(f"Added recursive watch for new directory: {dir_path}")
    async def _remove_directory_watch(self, dir_path: str) -> None:
        """Remove directory from monitoring and clean up database."""
        async with self.watch_lock:
            if dir_path in self.watched_directories:
                # Note: Watchdog auto-removes watches for deleted dirs
                self.watched_directories.discard(dir_path)

                # Clean up database entries for files in deleted directory
                await self._cleanup_deleted_directory(
                    dir_path,
                    source_generation=self._current_source_generation(dir_path),
                )
                logger.debug(f"Removed watch for deleted directory: {dir_path}")
    async def _cleanup_deleted_directory(
        self, dir_path: str | Path, *, source_generation: int | None = None
    ) -> int:
        """Queue cleanup work for files that were under a deleted directory.

        Enumerates rows directly from the ``files`` table by path prefix so
        chunkless rows (binary, empty, or unparseable files) are still cleaned
        up alongside chunked rows.
        """
        normalized_dir = self._normalize_mutation_path(dir_path)
        absolute_dir = str(normalized_dir)

        base_root = self._path_filter_root(normalized_dir)
        try:
            relative_dir = normalized_dir.relative_to(base_root).as_posix()
        except ValueError:
            relative_dir = absolute_dir

        provider = self.services.provider
        list_paths = getattr(provider, "list_file_paths_under_directory", None)
        if callable(list_paths):
            file_paths = await asyncio.to_thread(list_paths, relative_dir)
        else:
            search_results, _ = await provider.search_regex_async(
                pattern=f"^{absolute_dir}/.*",
                page_size=1000,
            )
            file_paths = [
                result.get("file_path") or result.get("path", "")
                for result in search_results
            ]

        queued_files = 0
        for file_path in file_paths:
            if not file_path:
                continue
            accepted = await self._enqueue_mutation(
                self._build_mutation(
                    "delete",
                    file_path,
                    source_generation=source_generation,
                )
            )
            if accepted:
                queued_files += 1

        logger.info(
            "Queued cleanup for "
            f"{queued_files} files from deleted directory: {absolute_dir}"
        )
        self._debug(
            f"queued deleted directory cleanup {absolute_dir} files={queued_files}"
        )
        return queued_files
    async def _process_delete_mutation(
        self, mutation: RealtimeMutation, *, owned_when_dequeued: bool
    ) -> None:
        """Apply one queued delete with bounded retry for transaction conflicts."""
        if not owned_when_dequeued and not self._delete_mutation_is_stale(mutation):
            self._debug(
                "skipped superseded delete "
                f"path={mutation.path} source_generation={mutation.source_generation}"
            )
            return

        self._record_processing_started(mutation.path)
        completed = False
        try:
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "skipped stale delete "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                completed = True
                return

            await self.remove_file(mutation.path)
            completed = True
        except DuckDBTransactionConflictError as error:
            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "ignored stale delete conflict "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                completed = True
                return

            if self._schedule_delete_retry(mutation):
                if self._delete_mutation_is_stale(mutation):
                    completed = True
                    return
                logger.info(
                    "Retrying realtime delete for "
                    f"{mutation.path} after transaction conflict "
                    f"(attempt {mutation.retry_count + 1}/"
                    f"{self._DELETE_CONFLICT_MAX_RETRIES})"
                )
                self._debug(
                    "retrying delete after transaction conflict "
                    f"path={mutation.path} attempt={mutation.retry_count + 1}"
                )
                return

            logger.error(f"Error removing file {mutation.path}: {error}")
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(f"Error removing file {mutation.path}: {error}")
        except Exception as error:
            logger.error(f"Error removing file {mutation.path}: {error}")
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(f"Error removing file {mutation.path}: {error}")
        finally:
            self._record_processing_finished(mutation.path, completed=completed)
    async def _process_delete_batch(
        self, mutations: list[tuple[RealtimeMutation, bool]]
    ) -> None:
        """Apply a bounded batch of queued deletes with per-path retry handling."""
        active_mutations: list[RealtimeMutation] = []
        finished_mutation_ids: set[int] = set()

        for mutation, owned_when_dequeued in mutations:
            if not owned_when_dequeued and not self._delete_mutation_is_stale(mutation):
                self._debug(
                    "skipped superseded delete "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                continue

            self._record_processing_started(mutation.path)
            active_mutations.append(mutation)

            if self._delete_mutation_is_stale(mutation):
                self._debug(
                    "skipped stale delete "
                    f"path={mutation.path} source_generation="
                    f"{mutation.source_generation}"
                )
                self._record_processing_finished(mutation.path, completed=True)
                finished_mutation_ids.add(mutation.mutation_id)

        executable_mutations = [
            mutation
            for mutation in active_mutations
            if mutation.mutation_id not in finished_mutation_ids
        ]
        if not executable_mutations:
            return

        sample_paths = ", ".join(
            str(mutation.path) for mutation in executable_mutations[:3]
        )
        try:
            await self.services.provider.delete_files_batch_async(
                [str(mutation.path) for mutation in executable_mutations]
            )
            for mutation in executable_mutations:
                self._record_processing_finished(mutation.path, completed=True)
                finished_mutation_ids.add(mutation.mutation_id)
        except DuckDBTransactionConflictError as error:
            surviving_mutations: list[RealtimeMutation] = []
            exhausted_mutations: list[RealtimeMutation] = []

            for mutation in executable_mutations:
                if self._delete_mutation_is_stale(mutation):
                    self._debug(
                        "ignored stale delete conflict "
                        f"path={mutation.path} source_generation="
                        f"{mutation.source_generation}"
                    )
                    self._record_processing_finished(mutation.path, completed=True)
                    finished_mutation_ids.add(mutation.mutation_id)
                    continue

                if self._schedule_delete_retry(mutation):
                    surviving_mutations.append(mutation)
                    continue

                exhausted_mutations.append(mutation)

            if surviving_mutations:
                logger.info(
                    "Retrying realtime delete batch for "
                    f"{len(surviving_mutations)} files after transaction conflict"
                )
                self._debug(
                    "retrying delete batch after transaction conflict "
                    f"sample_paths={sample_paths}"
                )

            if exhausted_mutations:
                exhausted_paths = ", ".join(
                    str(mutation.path) for mutation in exhausted_mutations[:3]
                )
                logger.error(f"Error removing files {exhausted_paths}: {error}")
                for mutation in exhausted_mutations:
                    self.failed_files.add(str(mutation.path))
                self._record_processing_error()
                self._set_error(f"Error removing files {exhausted_paths}: {error}")
        except Exception as error:
            logger.error(f"Error removing files {sample_paths}: {error}")
            for mutation in executable_mutations:
                self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(f"Error removing files {sample_paths}: {error}")
        finally:
            for mutation in active_mutations:
                if mutation.mutation_id in finished_mutation_ids:
                    continue
                self._record_processing_finished(mutation.path, completed=False)
    async def _process_deleted_directory_mutation(
        self, mutation: RealtimeMutation
    ) -> None:
        self._record_processing_started(mutation.path)
        completed = False
        try:
            await self._cleanup_deleted_directory(
                mutation.path,
                source_generation=mutation.source_generation,
            )
            completed = True
        except Exception as error:
            logger.error(
                f"Error cleaning up deleted directory {mutation.path}: {error}"
            )
            self.failed_files.add(str(mutation.path))
            self._record_processing_error()
            self._set_error(
                f"Error cleaning up deleted directory {mutation.path}: {error}"
            )
        finally:
            self._record_processing_finished(mutation.path, completed=completed)
    async def _index_directory(self, dir_path: Path) -> None:
        """Index files in a newly created directory."""
        self._record_processing_started(dir_path)
        completed = False
        try:
            supported_files = await asyncio.to_thread(
                self._collect_supported_files, dir_path
            )

            # Add files to processing queue
            for file_path in supported_files:
                await self.add_file(file_path, priority="change")

            logger.debug(
                f"Queued {len(supported_files)} files from new directory: {dir_path}"
            )
            self._debug(
                f"queued {len(supported_files)} files from new directory: {dir_path}"
            )
            completed = True

        except Exception as e:
            logger.error(f"Error indexing new directory {dir_path}: {e}")
            self._record_processing_error()
            self._set_error(f"Error indexing new directory {dir_path}: {e}")
        finally:
            self._record_processing_finished(dir_path, completed=completed)
    async def _process_loop(self) -> None:
        """Main processing loop - simple and robust."""
        logger.debug("Starting processing loop")
        buffered_mutation: tuple[RealtimeMutation, bool] | None = None

        while True:
            try:
                if buffered_mutation is None:
                    _, _, queued_mutation = await self.file_queue.get()
                    mutation, owned_when_dequeued = self._prepare_dequeued_mutation(
                        queued_mutation
                    )
                else:
                    mutation, owned_when_dequeued = buffered_mutation
                    buffered_mutation = None

                active_change_path: str | None = None
                if mutation.operation in {"change", "scan"}:
                    active_change_path = str(mutation.path)
                    self._active_change_generations[active_change_path] = (
                        mutation.source_generation
                    )

                # Fast path for embedding generation without re-parsing the file.
                if mutation.operation == "embed":
                    completed = False
                    try:
                        self._record_processing_started(mutation.path)
                        indexing_coordinator = self.services.indexing_coordinator
                        await indexing_coordinator.generate_missing_embeddings()
                        completed = True
                    except Exception as error:
                        logger.warning(
                            "Embedding generation failed in realtime "
                            f"(embed pass): {error}"
                        )
                        self._record_processing_error()
                        self._set_warning(
                            "Embedding generation failed in realtime "
                            f"embed pass: {error}"
                        )
                    finally:
                        self._record_processing_finished(
                            mutation.path, completed=completed
                        )
                    continue

                if mutation.operation == "delete":
                    delete_batch, buffered_mutation = self._collect_delete_batch(
                        mutation,
                        owned_when_dequeued=owned_when_dequeued,
                    )
                    if len(delete_batch) == 1:
                        await self._process_delete_mutation(
                            mutation,
                            owned_when_dequeued=owned_when_dequeued,
                        )
                    else:
                        await self._process_delete_batch(delete_batch)
                    continue

                if mutation.operation == "dir_delete":
                    await self._process_deleted_directory_mutation(mutation)
                    continue

                if mutation.operation == "dir_index":
                    await self._index_directory(mutation.path)
                    continue

                file_path = mutation.path

                # Check if file still exists (prevent race condition with deletion)
                if not file_path.exists():
                    if active_change_path is not None:
                        self._active_change_generations.pop(active_change_path, None)
                    logger.debug(f"Skipping {file_path} - file no longer exists")
                    async with self._file_condition:
                        self.failed_files.add(normalize_file_path(file_path))
                        self._file_condition.notify_all()
                    continue

                # Process the file
                logger.debug(
                    f"Processing {file_path} (operation: {mutation.operation})"
                )

                # Skip embeddings for initial and change events to keep loop responsive.
                # An explicit 'embed' follow-up event will generate embeddings.
                skip_embeddings = True

                # Use existing indexing coordinator
                self._record_processing_started(file_path)
                completed = False
                processing_error: Exception | None = None
                try:
                    result = await self.services.indexing_coordinator.process_file(
                        file_path, skip_embeddings=skip_embeddings
                    )

                    # Ensure database transaction is flushed for immediate visibility
                    if hasattr(self.services.provider, "flush"):
                        await self.services.provider.flush()

                    normalized = normalize_file_path(file_path)
                    async with self._file_condition:
                        self._indexed_files.add(normalized)
                        self._file_condition.notify_all()

                    # Clear event dedup entry so future modifications aren't suppressed
                    self._recent_file_events.pop(str(file_path), None)

                    # If we skipped embeddings, queue for embedding generation
                    if skip_embeddings:
                        await self.add_file(file_path, priority="embed")

                    # Record processing summary into MCP debug log
                    try:
                        chunks = (
                            result.get("chunks", None)
                            if isinstance(result, dict)
                            else None
                        )
                        embeds = (
                            result.get("embeddings", None)
                            if isinstance(result, dict)
                            else None
                        )
                        self._debug(
                            f"processed {file_path} operation={mutation.operation} "
                            f"skip_embeddings={skip_embeddings} "
                            f"chunks={chunks} embeddings={embeds}"
                        )
                    except Exception:
                        pass
                    completed = True
                except Exception as error:
                    processing_error = error
                finally:
                    self._record_processing_finished(file_path, completed=completed)
                    if active_change_path is not None:
                        self._active_change_generations.pop(active_change_path, None)
                        await self._drain_reserved_follow_up_change(
                            file_path,
                            normalized_path=active_change_path,
                            processing_error=processing_error,
                        )
                if processing_error is not None:
                    raise processing_error

            except asyncio.CancelledError:
                logger.debug("Processing loop cancelled")
                raise
            except Exception as error:
                mutation_path = (
                    mutation.path if "mutation" in locals() else Path("<unknown>")
                )
                logger.error(f"Error processing {mutation_path}: {error}")
                # Track failed files for debugging and monitoring
                self._record_processing_error()
                self._set_error(f"Error processing {mutation_path}: {error}")
                async with self._file_condition:
                    self.failed_files.add(normalize_file_path(mutation_path))
                    self._file_condition.notify_all()
                # Continue processing other files
