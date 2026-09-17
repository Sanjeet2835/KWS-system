"""Tiny shared board for the UI thread. No locks needed for these scalars."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Board:
    status: str = "LISTENING"
    keyword: str = ""
    confidence: float = 0.0
    transcript: str = "—"
    last_latency_ms: float | None = None
    bytes_in: int = 0
    streams: int = 0
    backend: str = "—"
    messages: list[str] = field(default_factory=list)
    # ESP32-S3 listen node (scored device). Updated by UDP TEL1.
    s3_state: str = "offline"
    s3_cpu: float = 0.0
    s3_idle_cpu: float = 0.0
    s3_heap_kb: int = 0
    s3_rssi: int = 0
    s3_rms: float = 0.0
    s3_seen: float = 0.0
    # On-device INT8 pipeline cost, straight from the node.
    s3_mfcc_ms: float = 0.0
    s3_invoke_ms: float = 0.0
    s3_arena_kb: float = 0.0
    s3_wakes: int = 0
    s3_lat_ms: float = 0.0
    transcript_partial: bool = False
    latencies: list[float] = field(default_factory=list)

    def note_latency(self, ms: float) -> None:
        self.last_latency_ms = ms
        self.latencies.append(ms)
        self.latencies = self.latencies[-8:]

    def log(self, line: str) -> None:
        self.messages.append(line)
        self.messages = self.messages[-6:]
