"""Do not release worker-owned resources just because an await was cancelled."""

import asyncio
from typing import TypeVar

T = TypeVar("T")


async def await_worker_completion(future: asyncio.Future[T]) -> T:
    """Drain retained worker; cancellation wins after retrieving its result/error.

    Pass an executor completion future, not an independently cancellable Task
    (e.g. to_thread): a cancelled Task need not mean its thread has stopped.
    Keep the future private; all awaiting owners must use this shielded drain.
    """
    cancellation: asyncio.CancelledError | None = None
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            cancellation = error
        except BaseException:
            # Future holds the original error; retrieve it exactly once below.
            break
    try:
        result = future.result()
    except BaseException:
        if cancellation is not None:
            raise cancellation from None
        raise
    if cancellation is not None:
        raise cancellation
    return result
