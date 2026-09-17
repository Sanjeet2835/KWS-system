"""Hardware gate: is the INMP441 actually wired and producing audio?

Everything measured downstream (TPR, false accepts, latency) is meaningless
until this passes, so it gets its own tool with a pass/fail verdict.

The firmware prints `MIC rms=... peak=... dc=...` at 4 Hz once mic monitoring is
on; this sends the `MIC` toggle command and judges the numbers.

  python -m tools.mic_check
  python -m tools.mic_check --seconds 20

Wiring (ESP32-S3-DevKitC, INMP441 is 3.3 V only):
  VDD -> 3V3   GND -> GND   L/R -> GND
  SCK -> GPIO 15   WS -> GPIO 16   SD -> GPIO 17
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.s3_link import DEFAULT_BAUD, S3Link, default_port  # noqa: E402

MIC_RE = re.compile(r"MIC rms=([\d.eE+-]+) peak=([\d.eE+-]+) dc=([\d.eE+-]+)")

# A live INMP441 idles around 1e-4..3e-3 RMS. Stuck-at-zero means no clock or no
# data line; a pinned value means the SD pin is floating or L/R is wrong.
SILENT_RMS = 2e-5
# After 8× digital gain, a quiet room sits around 0.02–0.04 RMS.
QUIET_ROOM_MAX = 0.08


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--seconds", type=float, default=15.0)
    return p.parse_args()


def bar(rms: float, width: int = 34) -> str:
    filled = int(min(1.0, rms / 0.15) * width)
    return "#" * filled + "." * (width - filled)


def verdict(samples: list[tuple[float, float, float]]) -> int:
    if not samples:
        print("\nFAIL: no MIC lines. Flash the current firmware and check the port.")
        return 1
    rms = [s[0] for s in samples]
    peak = [s[1] for s in samples]
    dc = [s[2] for s in samples]
    lo, hi = min(rms), max(rms)
    print("\n--- verdict ---")
    print(f"rms   min={lo:.6f} max={hi:.6f}")
    print(f"peak  max={max(peak):.4f}")
    print(f"dc    min={min(dc):.1f} max={max(dc):.1f} (raw 24-bit mean)")

    if hi < SILENT_RMS:
        print("FAIL: signal is flat. SD/SCK/WS not connected, or VDD is on 5 V.")
        return 1
    if hi - lo < SILENT_RMS:
        print("FAIL: constant value, no variation. SD pin is likely floating.")
        return 1
    if lo > QUIET_ROOM_MAX:
        print("WARN: never gets quiet. Check L/R is tied to GND and the mic is not clipping.")
        return 1
    if hi < 0.01:
        print("WARN: mic responds but nothing loud was seen. Speak into it and rerun.")
        return 1
    print("PASS: mic is live and tracks speech.")
    print("Next: python -m tools.record_s3 --label keyword --speaker <you> --count 30")
    return 0


def main() -> int:
    args = parse_args()
    port = args.port or default_port()
    print(f"listening on {port} for {args.seconds:.0f}s")
    print("Say the wake word a few times, then tap the mic body.\n")

    samples: list[tuple[float, float, float]] = []

    with S3Link(port, args.baud, reset=False) as link:
        ready = False
        # Opening UART-USB often pulses reset; wait until the app is up
        # before sending MIC or we only catch the boot banner.
        for ev in link.read(20.0):
            print(f"  {ev.raw}")
            if ev.kind == "tel" or "boot: done" in ev.raw:
                ready = True
                break
        if not ready:
            print("FAIL: board never reached listen. Unplug/replug USB-C and retry.")
            return 1
        link.send("MIC")
        try:
            for ev in link.read(args.seconds):
                if ev.kind == "mic":
                    rms = float(ev.fields["rms"])
                    peak = float(ev.fields["peak"])
                    dc = float(ev.fields["dc"])
                    samples.append((rms, peak, dc))
                    print(f"  {ev.raw}")
                    print(f"  rms={rms:.5f} peak={peak:.3f} {bar(rms)}")
                elif ev.raw:
                    print(f"  {ev.raw}")
        except KeyboardInterrupt:
            pass
        link.send("MIC")
        time.sleep(0.2)

    return verdict(samples)


if __name__ == "__main__":
    raise SystemExit(main())
