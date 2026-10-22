"""In-process outbound boundary; native code and child processes need OS isolation."""

import ipaddress
import socket
from collections.abc import Callable
from functools import wraps
from typing import Any

import pytest
from pytest_socket import SocketConnectBlockedError

_PATCHES: pytest.MonkeyPatch | None = None
_ALLOWED = ["loopback", "localhost", "Unix sockets"]


def _check_host(host: Any) -> None:
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="replace")
    if isinstance(host, str):
        if host.lower().rstrip(".") == "localhost":
            return
        try:
            address = ipaddress.ip_address(host)
            if address.is_loopback:
                return
        except ValueError:
            pass
    raise SocketConnectBlockedError(_ALLOWED, str(host))


def _check_address(connection: socket.socket, address: Any) -> None:
    if connection.family == getattr(socket, "AF_UNIX", None):
        return
    _check_host(address[0] if isinstance(address, tuple) else address)


def _guard_socket(method: Callable[..., Any], name: str) -> Callable[..., Any]:
    @wraps(method)
    def guarded(connection: socket.socket, *args: Any, **kwargs: Any) -> Any:
        if name == "sendmsg":
            address = args[3] if len(args) > 3 else kwargs.get("address")
        else:
            address = args[-1] if name == "sendto" else args[0]
        if address is not None:
            _check_address(connection, address)
        return method(connection, *args, **kwargs)

    return guarded


def _guard_lookup(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def guarded(host: Any = None, *args: Any, **kwargs: Any) -> Any:
        # A null getaddrinfo host selects local bind addresses without DNS.
        if host is not None or method.__name__ != "getaddrinfo":
            _check_host(host[0] if method.__name__ == "getnameinfo" else host)
        return method(host, *args, **kwargs)

    return guarded


def restrict_network() -> None:
    global _PATCHES
    if _PATCHES is not None:
        return
    _PATCHES = pytest.MonkeyPatch()
    for name in ("connect", "connect_ex", "sendto", "sendmsg"):
        if hasattr(socket.socket, name):
            method = getattr(socket.socket, name)
            _PATCHES.setattr(socket.socket, name, _guard_socket(method, name))
    for name in (
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
        "gethostbyaddr",
        "getnameinfo",
    ):
        _PATCHES.setattr(socket, name, _guard_lookup(getattr(socket, name)))


def restore_network() -> None:
    global _PATCHES
    if _PATCHES is not None:
        _PATCHES.undo()
        _PATCHES = None
