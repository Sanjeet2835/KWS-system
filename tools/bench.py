"""On-device MFCC / TFLM timings and idle-listen CPU, written to results/bench.json.

BENCH forces 20 inferences on a quiet buffer so the numbers do not depend on a
live microphone. Idle CPU is sampled from TEL1 while the energy gate is skipping
the network — that is the quota the problem statement scores.

  python -m tools.bench
  python -m tools.bench --no-reset --idle-seconds 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.s3_link import DEFAULT_BAUD, S3Link  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"

BENCH_RE = re.compile(
    r"BENCH reps=(?P<reps>\d+) "
    r"mfcc_us=(?P<mfcc_min>\d+)/(?P<mfcc_avg>\d+)/(?P<mfcc_max>\d+) "
    r"inv_us=(?P<inv_min>\d+)/(?P<inv_avg>\d+)/(?P<inv_max>\d+) "
    r"total_us=(?P<total>\d+) period_us=(?P<period>\d+) "
    r"speech_cpu=(?P<speech_cpu>[\d.]+)% "
    r"arena=(?P<arena_used>\d+)/(?P<arena_bytes>\d+)"
)
WARM_RE = re.compile(r"warm-up ok mfcc_us=(?P<mfcc>\d+) inv_us=(?P<inv>\d+) pkw=(?P<pkw>[\d.]+)")
TFLM_RE = re.compile(
    r"tflm ok model=(?P<model>\d+) bytes arena=(?P<arena_used>\d+)/(?P<arena_bytes>\d+) bytes"
)
CPU_RE = re.compile(r"cpu=(?P<mhz>\d+) MHz")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--no-reset", action="store_true")
    p.add_argument("--idle-seconds", type=float, default=15.0)
    p.add_argument("--out", type=Path, default=RESULTS / "bench.json")
    return p.parse_args()


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    ordered = sorted(xs)
    return {
        "n": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(ordered[len(ordered) // 2], 3),
        "max": round(ordered[-1], 3),
    }


def main() -> int:
    args = parse_args()
    bench = None
    warm = None
    tflm = None
    mhz = None
    status: dict = {}
    cpu_samples: list[float] = []
    rms_samples: list[float] = []

    with S3Link(args.port, args.baud, reset=not args.no_reset) as link:
        # Catch the boot banner (warm-up / tflm / cpu MHz) if we just reset.
        for ev in link.read(8.0):
            m = BENCH_RE.search(ev.raw)
            if m:
                bench = {k: float(v) if "." in v else int(v) for k, v in m.groupdict().items()}
            m = WARM_RE.search(ev.raw)
            if m:
                warm = {k: float(v) if k == "pkw" else int(v) for k, v in m.groupdict().items()}
            m = TFLM_RE.search(ev.raw)
            if m:
                tflm = {k: int(v) for k, v in m.groupdict().items()}
            m = CPU_RE.search(ev.raw)
            if m:
                mhz = int(m.group("mhz"))
            if ev.kind == "status":
                status = ev.fields
            print(ev.raw)

        if bench is None:
            link.send("BENCH")
            for ev in link.read(12.0):
                print(ev.raw)
                m = BENCH_RE.search(ev.raw)
                if m:
                    bench = {k: float(v) if "." in v else int(v) for k, v in m.groupdict().items()}
                    break

        status = link.status() or status
        print(f"STATUS {status}")

        print(f"sampling idle TEL1 for {args.idle_seconds:.0f}s...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.idle_seconds:
            for ev in link.read(2.0):
                if ev.kind == "tel":
                    if ev.fields.get("state") == "listen" and "cpu" in ev.fields:
                        cpu_samples.append(float(ev.fields["cpu"]))
                    if "rms" in ev.fields:
                        rms_samples.append(float(ev.fields["rms"]))

    if bench is None:
        print("FAIL: no BENCH line. Flash the current firmware.")
        return 1

    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu_mhz": mhz,
        "tflm": tflm,
        "warm_up": warm,
        "bench": bench,
        "status": status,
        "idle_cpu_pct": _stats(cpu_samples),
        "idle_rms": _stats(rms_samples),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(
        f"\nsteady-state  mfcc={bench['mfcc_avg']/1000:.1f} ms  "
        f"invoke={bench['inv_avg']/1000:.1f} ms  "
        f"speech CPU={bench['speech_cpu']}% of one core / {bench['period']/1000:.0f} ms"
    )
    if result["idle_cpu_pct"]:
        c = result["idle_cpu_pct"]
        print(f"idle listen CPU  mean={c['mean']}%  max={c['max']}%  (quota 10%)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
