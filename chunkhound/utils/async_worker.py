"""Drain private completion futures before relinquishing worker resources."""
import asyncio
from collections.abc import Callable
from typing import TypeVar
from loguru import logger

T = TypeVar("T")

class WorkerDrain:
    def __init__(self, on_cancel: Callable[[], None] | None = None) -> None:
        self.cancellation: asyncio.CancelledError | None = None
        self.on_cancel = on_cancel

    async def wait(self, future: asyncio.Future[T]) -> T:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError as error:
                if self.cancellation is None:
                    self.cancellation = error
                if self.on_cancel is not None:
                    try:
                        self.on_cancel()
                    except BaseException as callback_error:
                        # Stop signalling is best effort, drainage is mandatory.
                        # In particular a broken callback must not release the
                        # caller while the private completion future still runs.
                        logger.error("Worker stop callback failed: {}", type(callback_error).__name__)
            except BaseException:
                break
        return future.result()

    def finish(self, error: BaseException | None = None) -> None:
        if self.cancellation is not None:
            if error is not None and not isinstance(error, asyncio.CancelledError):
                logger.error("Worker failed during cancellation: {}", type(error).__name__)
            raise self.cancellation
        if error is not None:
            raise error

async def await_worker_completion(future: asyncio.Future[T]) -> T:
    drain = WorkerDrain()
    try:
        result = await drain.wait(future)
    except BaseException as error:
        drain.finish(error)
        raise  # finish always raises in this branch
    drain.finish()
    return result
