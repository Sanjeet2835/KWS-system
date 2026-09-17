"""Node → server bytes. Keep this identical if the node moves to an ESP32."""

from __future__ import annotations

import struct

MAGIC = b"KWS1"
VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBHI")  # magic, version, preroll_ms, sample_rate
FRAME_LEN_STRUCT = struct.Struct("<H")


def pack_header(preroll_ms: int, sample_rate: int) -> bytes:
    return HEADER_STRUCT.pack(MAGIC, VERSION, preroll_ms, sample_rate)


def unpack_header(buf: bytes) -> tuple[int, int]:
    magic, version, preroll_ms, sample_rate = HEADER_STRUCT.unpack(buf)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported protocol version {version}")
    return preroll_ms, sample_rate


def pack_frame(pcm_le: bytes) -> bytes:
    if len(pcm_le) > 65535:
        raise ValueError("frame too large")
    return FRAME_LEN_STRUCT.pack(len(pcm_le)) + pcm_le


def pack_end() -> bytes:
    return FRAME_LEN_STRUCT.pack(0)


HEADER_BYTES = HEADER_STRUCT.size
