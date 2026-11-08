"""Detached lifecycle diagnostics and single-flight provider readiness."""

import asyncio
import contextvars
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, wait
from contextlib import contextmanager
from dataclasses import dataclass

from chunkhound.providers.database.ownership import (
    DatabaseOwnership,
    DatabaseReaderCloseError,
)
from chunkhound.providers.database.serial_executor import (
    DatabaseOperationTimeoutError,
    DatabaseTemporarilyUnavailableError,
)
from chunkhound.utils.async_worker import await_worker_completion
from chunkhound.utils.text_sanitization import sanitize_error_text


@dataclass(frozen=True)
class Failure:
    kind: type[BaseException]
    message: str
    cause: tuple["Failure", ...] = ()

    @classmethod
    def capture(
        cls, error: BaseException, seen: frozenset[int] = frozenset()
    ) -> "Failure":
        if id(error) in seen:
            return cls(type(error), "exception cause cycle")
        seen = seen | {id(error)}
        # Never keep exception objects, args, frames, futures, or backend handles.
        text = re.sub(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+", "[redacted URL]", str(error))
        text = sanitize_error_text(text)
        notes = getattr(error, "__notes__", ())
        if notes:
            note_text = re.sub(
                r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s]+",
                "[redacted URL]",
                "; ".join(notes),
            )
            text += " [" + sanitize_error_text(note_text) + "]"
        links = []
        for linked in (error.__cause__, error.__context__):
            if linked is not None and id(linked) not in seen:
                links.append(cls.capture(linked, seen))
                seen = seen | {id(linked)}
        return cls(type(error), text, tuple(links))

    def exception(self) -> BaseException:
        # Common Python/native exceptions accept a message. Structured provider
        # exceptions need their specific constructor; do not preserve raw args.
        from chunkhound.core.exceptions import DatabaseError

        result: BaseException
        if issubclass(self.kind, DatabaseError):
            result = DatabaseError(operation="database readiness", reason=self.message)
        elif self.kind is KeyboardInterrupt or self.kind is SystemExit:
            # Exact built-ins only: preserve control flow without invoking unknown
            # constructors. SystemExit.code is sanitized text, never a raw payload.
            result = self.kind(self.message)
        elif any(
            self.kind.__init__ is plain.__init__
            and self.kind.__new__ is plain.__new__
            for plain in (
                BaseException,
                Exception,
                RuntimeError,
                ValueError,
                TypeError,
                OSError,
                TimeoutError,
                LookupError,
                AssertionError,
            )
        ):
            # Only inherited, message-based constructors are safe here. Unknown
            # structured exceptions may silently accept a string as a resource.
            try:
                result = self.kind(self.message)
            except Exception:
                result = RuntimeError(f"{self.kind.__name__}: {self.message}")
        else:
            result = RuntimeError(f"{self.kind.__name__}: {self.message}")
        if self.cause:
            result.__cause__ = self.cause[0].exception()
        return result


@dataclass(frozen=True)
class LifecycleFailure:
    stage: str
    original: Failure | None
    cleanup: Failure
    restart_required: bool = False

    def exception(self) -> "DatabaseRecoveryRequiredError":
        return DatabaseRecoveryRequiredError(self)


class DatabaseRecoveryRequiredError(DatabaseTemporarilyUnavailableError):
    def __init__(self, failure: LifecycleFailure) -> None:
        self.failure = failure
        advice = (
            "Native cleanup is not acknowledged; restart the process and use "
            "external recovery tooling if needed."
            if failure.restart_required
            else "The next database-backed action will retry cleanup/reconnect."
        )
        original = (
            f" Original {failure.original.kind.__name__}: {failure.original.message}."
            if failure.original is not None
            else ""
        )
        super().__init__(
            f"Database recovery required at {failure.stage}.{original} "
            f"{failure.cleanup.kind.__name__}: {failure.cleanup.message}. {advice}"
        )
        self.__cause__ = failure.cleanup.exception()
        if failure.original is not None:
            self.__cause__.__cause__ = failure.original.exception()


Outcome = Failure | LifecycleFailure | None
_owner: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "database_readiness_owner", default=None
)


@contextmanager
def readiness_owner(provider: object) -> Iterator[None]:
    token = _owner.set(provider)
    try:
        yield
    finally:
        _owner.reset(token)


def owns_readiness(provider: object) -> bool:
    return _owner.get() is provider


@dataclass(frozen=True)
class ReaderRecovery:
    """Intentional public retry capability, never part of detached diagnostics."""

    ownership: DatabaseOwnership
    release_ownership: bool

    def exception(self, failure: Failure) -> DatabaseReaderCloseError:
        error = DatabaseReaderCloseError(self.ownership)
        error._release_ownership = self.release_ownership
        if failure.cause:
            error.__cause__ = failure.cause[0].exception()
        return error


@dataclass
class ReadinessAttempt:
    completion: Future[Outcome]
    # Retained by this attempt's callers, not looked up on the mutable provider.
    # No caught exception/frame is cached, and Readiness.failure stays detached.
    reader_recovery: ReaderRecovery | None = None


class Readiness:
    """One result per admitted attempt; a later invocation may retry failure."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inflight: ReadinessAttempt | None = None
        self.ready = False
        self.failure: Outcome = None

    def claim(self, force: bool) -> tuple[ReadinessAttempt, bool] | None:
        with self.lock:
            if self.inflight is not None:
                return self.inflight, False
            if self.ready and not force:
                return None
            self.inflight = ReadinessAttempt(Future())
            return self.inflight, True

    def run(self, attempt: ReadinessAttempt, transition: Callable[[], None]) -> None:
        outcome: Outcome = None
        try:
            transition()
        except DatabaseReaderCloseError as error:
            attempt.reader_recovery = ReaderRecovery(
                error.ownership, error._release_ownership
            )
            outcome = Failure.capture(error)
        except DatabaseRecoveryRequiredError as error:
            outcome = error.failure
        except BaseException as error:
            outcome = Failure.capture(error)
        with self.lock:
            self.failure = outcome
            self.ready = outcome is None
            # Publish under the admission lock. Joined callers retain this exact
            # future even if a later invocation begins another attempt.
            attempt.completion.set_result(outcome)
            self.inflight = None

    @staticmethod
    def deliver(attempt: ReadinessAttempt, outcome: Outcome) -> None:
        if outcome is not None:
            error = (
                attempt.reader_recovery.exception(outcome)
                if attempt.reader_recovery is not None and isinstance(outcome, Failure)
                else outcome.exception()
            )
            raise error from error.__cause__

    def wait(
        self,
        claim: tuple[ReadinessAttempt, bool],
        transition: Callable[[], None],
        *,
        timeout: float,
    ) -> None:
        attempt, owner = claim
        if owner:
            context = contextvars.copy_context()
            threading.Thread(
                target=context.run,
                args=(self.run, attempt, transition),
                name="database-readiness",
            ).start()
        # The deadline belongs only to this caller. The admitted transition
        # continues to completion, and async joiners drain that same attempt.
        done, _ = wait((attempt.completion,), timeout=timeout)
        if attempt.completion not in done:
            raise DatabaseOperationTimeoutError("Operation 'connect' timed out")
        self.deliver(attempt, attempt.completion.result())

    async def wait_async(
        self, claim: tuple[ReadinessAttempt, bool], transition: Callable[[], None]
    ) -> None:
        attempt, owner = claim
        if owner:
            context = contextvars.copy_context()
            asyncio.get_running_loop().run_in_executor(
                None, context.run, self.run, attempt, transition
            )
        outcome = await await_worker_completion(asyncio.wrap_future(attempt.completion))
        self.deliver(attempt, outcome)
