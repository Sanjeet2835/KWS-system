"""Open a socket on wake and push 20 ms int16 frames."""

from __future__ import annotations

import socket
import time

import numpy as np

from node.capture import float_to_int16
from shared.config import ASR_HOST, ASR_PORT, PREROLL_MS
from shared.feature_spec import SPEC
from shared.protocol import pack_end, pack_frame, pack_header


class AsrClient:
    def __init__(self, host: str = ASR_HOST, port: int = ASR_PORT) -> None:
        self.host = host
        self.port = port

    def session(self, preroll: np.ndarray):
        return StreamSession(self.host, self.port, preroll)


class StreamSession:
    def __init__(self, host: str, port: int, preroll: np.ndarray) -> None:
        self.sock = socket.create_connection((host, port), timeout=2.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.t_connect = time.monotonic()
        self.sock.sendall(pack_header(PREROLL_MS, SPEC.sample_rate))
        self.send_audio(preroll)
        self.t_first_byte = time.monotonic()

    def send_audio(self, audio: np.ndarray) -> None:
        pcm = float_to_int16(audio).tobytes()
        # Send as 20 ms chunks so the server can measure arrival pacing.
        step = 640  # 320 samples * 2 bytes
        for i in range(0, len(pcm), step):
            self.sock.sendall(pack_frame(pcm[i : i + step]))

    def close(self) -> None:
        try:
            self.sock.sendall(pack_end())
        finally:
            self.sock.close()
