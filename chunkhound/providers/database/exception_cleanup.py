"""Release completed backend frames without discarding exception diagnostics."""

import traceback


def clear_exception_frames(error: BaseException) -> None:
    """Clear retained locals throughout both branches of an exception chain.

    Cause and context may differ (even when context is suppressed), and either
    may contain cycles. Keep the exceptions, links and traceback locations for
    diagnostics; only completed frames can release their backend references.
    """
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        traceback.clear_frames(current.__traceback__)
