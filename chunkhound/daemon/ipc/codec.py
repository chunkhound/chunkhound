"""Length-prefixed frame codec for IPC.

Uses msgpack for efficient binary serialization (hard dependency).

Frame format: [4-byte big-endian uint32 length][payload bytes]
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any

import msgpack as _msgpack  # type: ignore[import-untyped]

_HEADER = struct.Struct(">I")  # big-endian uint32
MAX_FRAME_BYTES = 100 * 1024 * 1024  # 100 MB


def encode(obj: Any) -> bytes:
    """Serialize *obj* to bytes using msgpack."""
    return _msgpack.packb(obj, use_bin_type=True)  # type: ignore[no-any-return]


def decode(data: bytes) -> Any:
    """Deserialize *data* from bytes using msgpack."""
    return _msgpack.unpackb(data, raw=False)


def write_frame(writer: asyncio.StreamWriter, obj: Any) -> None:
    """Encode *obj* and write a length-prefixed frame to *writer*."""
    payload = encode(obj)
    writer.write(_HEADER.pack(len(payload)))
    writer.write(payload)


async def read_frame(reader: asyncio.StreamReader) -> Any:
    """Read a length-prefixed frame from *reader* and decode it."""
    header = await reader.readexactly(_HEADER.size)
    (length,) = _HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise ValueError(f"IPC frame too large: {length} bytes (max {MAX_FRAME_BYTES})")
    payload = await reader.readexactly(length)
    return decode(payload)
