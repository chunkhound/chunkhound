"""Shared JSON-RPC 2.0 envelope helpers and exceptions for test clients.

Factored out as the common base of ``SubprocessJsonRpcClient`` and
``HttpMcpClient`` — both build the same request/notification envelopes and
unwrap the same result/error shape, even though they differ in how they
physically send bytes (stdin/stdout pipe vs. HTTP+SSE). Exceptions live here
too (rather than in ``subprocess_jsonrpc.py``) so both client modules can
import from this single module without a circular import.
"""

from __future__ import annotations

from typing import Any


class SubprocessJsonRpcError(Exception):
    """Base exception for JSON-RPC test-client communication errors."""

    pass


class SubprocessCrashError(SubprocessJsonRpcError):
    """Raised when the subprocess terminates unexpectedly."""

    pass


class JsonRpcTimeoutError(SubprocessJsonRpcError):
    """Raised when a JSON-RPC request times out."""

    pass


class JsonRpcResponseError(SubprocessJsonRpcError):
    """Raised when a JSON-RPC response contains an error."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"JSON-RPC error {code}: {message}")


def build_request(
    method: str, params: dict[str, Any] | None, request_id: int
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request envelope."""
    request: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        request["params"] = params
    return request


def build_notification(
    method: str, params: dict[str, Any] | None
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 notification envelope (no ``id`` field)."""
    notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        notification["params"] = params
    return notification


def unwrap_result(response: dict[str, Any], method: str) -> dict[str, Any]:
    """Return ``response["result"]``, raising on a JSON-RPC error or malformed body."""
    if "error" in response:
        error = response["error"]
        raise JsonRpcResponseError(
            code=error.get("code", -1),
            message=error.get("message", "Unknown error"),
            data=error.get("data"),
        )
    if "result" not in response:
        raise SubprocessJsonRpcError(
            f"Response to {method!r} missing 'result' field: {response}"
        )
    return response["result"]
