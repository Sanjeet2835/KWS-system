"""Gate F: keyword END → first audio byte arriving at the ASR.

One clock (this process) times both ends, so there is no NTP skew to argue about:

  1. Play a WAV whose keyword ends at a known offset.
  2. The node hears it through its own mic, wakes, and opens KWS1 to this port.
  3. We stamp the first payload byte and subtract the keyword-end instant.

The reported number therefore includes everything the problem statement cares
about: detection lag, chime/refractory handling, TCP connect, and preroll flush.

  python -m server.latency_harness --wav data/sahayak/keyword/kw_000.wav --trials 12

With --keyword-end-s omitted the end of the file is treated as the keyword end,
which is right for the single-word clips the dataset generator writes.
Results land in results/latency.json.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import ASR_PORT, WAKE_SCORE_THRESHOLD  # noqa: E402
from shared.protocol import FRAME_LEN_STRUCT, HEADER_BYTES, unpack_header  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=ASR_PORT)
    p.add_argument("--wav", type=Path, required=True, help="keyword clip to play")
    p.add_argument("--keyword-end-s", type=float, default=None, help="seconds from playback start to keyword end")
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--note", default="", help="playback level / distance, for the record")
    p.add_argument("--out", type=Path, default=RESULTS / "latency.json")
    return p.parse_args()


def play_wav(path: Path) -> None:
    import numpy as np
    import sounddevice as sd
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
        sw = wf.getsampwidth()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    data = np.frombuffer(raw, dtype=dtype)
    if nch > 1:
        data = data.reshape(-1, nch)
    sd.play(data, sr, blocking=True)


def wait_first_byte(sock: socket.socket, timeout: float) -> float | None:
    sock.settimeout(timeout)
    try:
        conn, _ = sock.accept()
    except TimeoutError:
        return None
    conn.settimeout(2.0)
    try:
        header = _recv(conn, HEADER_BYTES)
        unpack_header(header)
        (nbytes,) = FRAME_LEN_STRUCT.unpack(_recv(conn, 2))
        if nbytes:
            _recv(conn, nbytes)
        return time.monotonic()
    finally:
        conn.close()


def _recv(conn, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("closed")
        buf.extend(chunk)
    return bytes(buf)


def main() -> None:
    args = parse_args()
    import wave

    with wave.open(str(args.wav), "rb") as wf:
        duration = wf.getnframes() / float(wf.getframerate())
    keyword_end = args.keyword_end_s if args.keyword_end_s is not None else duration

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.listen(1)
    print(f"harness on :{args.port}  keyword_end={keyword_end:.3f}s  file={args.wav}")
    print("Start the node pointing at this port, then press Enter.")
    input()

    deltas = []
    misses = 0
    for i in range(args.trials):
        t0 = time.monotonic()
        play_wav(args.wav)
        t_end = t0 + keyword_end
        t_recv = wait_first_byte(sock, timeout=4.0)
        if t_recv is None:
            misses += 1
            print(f"trial {i + 1}: no stream received (node did not wake)")
            continue
        ms = (t_recv - t_end) * 1000.0
        deltas.append(ms)
        print(f"trial {i + 1}: {ms:.1f} ms")
        # Longer than REFRACTORY_S so the next trial starts from a clean state.
        time.sleep(1.8)

    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "wav": str(args.wav),
        "keyword_end_s": round(keyword_end, 3),
        "threshold": WAKE_SCORE_THRESHOLD,
        "trials": args.trials,
        "woke": len(deltas),
        "no_wake": misses,
        "samples_ms": [round(d, 1) for d in deltas],
    }
    if deltas:
        deltas.sort()
        n = len(deltas)
        result |= {
            "median_ms": round(deltas[n // 2], 1),
            "p95_ms": round(deltas[min(n - 1, int(0.95 * (n - 1) + 0.5))], 1),
            "min_ms": round(deltas[0], 1),
            "max_ms": round(deltas[-1], 1),
            "mean_ms": round(sum(deltas) / n, 1),
        }
        print(
            f"\nn={n}  median={result['median_ms']} ms  p95={result['p95_ms']} ms  "
            f"min={result['min_ms']}  max={result['max_ms']}  no-wake={misses}"
        )
    else:
        print("\nno successful trials — check the mic, threshold, and that the node reaches this host")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
