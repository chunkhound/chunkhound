"""Length-prefixed frame codec for IPC.

Tries to use msgpack for efficient binary serialization; falls back to JSON
if msgpack is not installed.  Both proxy and daemon share the same venv, so
the codec choice is always consistent — no protocol negotiation needed.

Frame format: [4-byte big-endian uint32 length][payload bytes]
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

try:
    import msgpack as _msgpack
    _USE_MSGPACK = True
except ImportError:
    _msgpack = None  # type: ignore[assignment]
    _USE_MSGPACK = False

_HEADER = struct.Struct(">I")  # big-endian uint32


def encode(obj: Any) -> bytes:
    """Serialize *obj* to bytes (msgpack or JSON)."""
    if _USE_MSGPACK:
        return _msgpack.packb(obj, use_bin_type=True)  # type: ignore[union-attr]
    return json.dumps(obj).encode()


def decode(data: bytes) -> Any:
    """Deserialize *data* from bytes (msgpack or JSON)."""
    if _USE_MSGPACK:
        return _msgpack.unpackb(data, raw=False)  # type: ignore[union-attr]
    return json.loads(data)


def write_frame(writer: asyncio.StreamWriter, obj: Any) -> None:
    """Encode *obj* and write a length-prefixed frame to *writer*."""
    payload = encode(obj)
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)


async def read_frame(reader: asyncio.StreamReader) -> Any:
    """Read a length-prefixed frame from *reader* and decode it."""
    header = await reader.readexactly(_HEADER.size)
    (length,) = _HEADER.unpack(header)
    payload = await reader.readexactly(length)
    return decode(payload)
