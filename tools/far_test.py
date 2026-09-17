"""Gate F: false accepts per hour, and idle CPU while it happens.

Leave the node listening to material it must ignore — TV, a podcast, Hindi and
English conversation — and count how many times it wakes. This is the number that
separates a wake word from a demo that fires whenever anyone talks.

  python -m tools.far_test --minutes 60 --note "Hindi news, 2 m, 60 dBA"

Play the audio from any speaker in the room; the node is not connected to it, so
there is nothing to synchronise. Results land in results/far.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import WAKE_SCORE_THRESHOLD  # noqa: E402
from tools.s3_link import DEFAULT_BAUD, S3Link  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--note", default="", help="what was playing, how loud, how far")
    p.add_argument("--out", type=Path, default=RESULTS / "far.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seconds = args.minutes * 60.0

    with S3Link(args.port, args.baud) as link:
        status = link.status()
        print(f"node status: {status or 'no reply'}")
        print(f"\nlistening for {args.minutes:.0f} min. Play the background material now.")
        print("Do NOT say the wake word. Ctrl-C stops early and still writes results.\n")

        wakes = []
        cpu_samples = []
        rms_samples = []
        invoke_us = []
        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 < seconds:
                for ev in link.read(5.0):
                    if ev.kind == "wake":
                        w = ev.wake
                        elapsed = w.host_time - t0
                        wakes.append(
                            {
                                "at_s": round(elapsed, 2),
                                "score": w.score,
                                "p_keyword": w.p_keyword,
                                "invoke_us": w.invoke_us,
                            }
                        )
                        print(f"  [{elapsed / 60:6.2f} min] FALSE ACCEPT score={w.score:.3f}")
                    elif ev.kind == "tel":
                        f = ev.fields
                        if f.get("state") == "listen" and "cpu" in f:
                            cpu_samples.append(float(f["cpu"]))
                        if "rms" in f:
                            rms_samples.append(float(f["rms"]))
                        if f.get("inv_us"):
                            invoke_us.append(int(f["inv_us"]))
                mins = (time.monotonic() - t0) / 60.0
                rate = len(wakes) / max(mins / 60.0, 1e-9)
                print(
                    f"\r  {mins:6.2f} / {args.minutes:.0f} min   "
                    f"false accepts={len(wakes)}  rate={rate:.2f}/hr",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            print("\nstopped early")

    elapsed_min = (time.monotonic() - t0) / 60.0
    hours = elapsed_min / 60.0
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": args.note,
        "threshold": WAKE_SCORE_THRESHOLD,
        "duration_min": round(elapsed_min, 2),
        "false_accepts": len(wakes),
        "false_accepts_per_hour": round(len(wakes) / hours, 3) if hours > 0 else None,
        "wakes": wakes,
        "idle_cpu_pct": _stats(cpu_samples),
        "rms": _stats(rms_samples),
        "invoke_us": _stats([float(v) for v in invoke_us]),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"\n\nduration {elapsed_min:.1f} min, {len(wakes)} false accepts")
    print(f"=> {result['false_accepts_per_hour']} false accepts/hour at threshold {WAKE_SCORE_THRESHOLD}")
    if result["idle_cpu_pct"]:
        c = result["idle_cpu_pct"]
        print(f"idle CPU: mean {c['mean']}% max {c['max']}% (quota 10%)")
    print(f"wrote {args.out}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
