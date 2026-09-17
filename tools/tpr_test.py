"""Gate F: true positive rate from a scripted live trial.

Accuracy on held-out WAVs is not the number judges care about; the number is
"how often does it wake when I say the word, standing where I actually stand".
So this prompts a human, waits for the node to wake, and records hit or miss per
distance.

  python -m tools.tpr_test --speaker mayank --per-distance 10
  python -m tools.tpr_test --distances 1 3 5 --per-distance 10

Results land in results/tpr.json, keyed by speaker and distance.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import KEYWORD, WAKE_SCORE_THRESHOLD  # noqa: E402
from tools.s3_link import DEFAULT_BAUD, S3Link  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--speaker", default="anon")
    p.add_argument("--distances", type=float, nargs="+", default=[1.0, 3.0])
    p.add_argument("--per-distance", type=int, default=10)
    p.add_argument("--window", type=float, default=3.0, help="seconds allowed for a wake")
    p.add_argument("--out", type=Path, default=RESULTS / "tpr.json")
    return p.parse_args()


def drain(link: S3Link) -> None:
    for _ in link.read(0.4):
        pass


def one_trial(link: S3Link, window: float) -> dict:
    """Wait for a wake after the prompt. The refractory period is 1.5 s, so one
    utterance can only ever produce one WAKE line."""
    t0 = time.monotonic()
    for ev in link.read(window):
        if ev.kind == "wake":
            w = ev.wake
            return {
                "hit": True,
                "latency_s": round(w.host_time - t0, 3),
                "score": w.score,
                "p_keyword": w.p_keyword,
                "invoke_us": w.invoke_us,
                "mfcc_us": w.mfcc_us,
            }
    return {"hit": False}


def main() -> int:
    args = parse_args()

    with S3Link(args.port, args.baud) as link:
        status = link.status()
        print(f"node status: {status or 'no reply'}")
        print(f"\nkeyword={args.keyword!r} speaker={args.speaker} threshold={WAKE_SCORE_THRESHOLD}")
        print("Enter = I am about to say it, s = skip, q = quit this distance\n")

        by_distance = {}
        for dist in args.distances:
            print(f"--- stand {dist:g} m from the mic ---")
            trials = []
            while len(trials) < args.per_distance:
                cmd = input(f"[{len(trials) + 1}/{args.per_distance}] {dist:g} m > ").strip().lower()
                if cmd == "q":
                    break
                if cmd == "s":
                    continue
                drain(link)
                print(f"    say '{args.keyword}' now...")
                r = one_trial(link, args.window)
                trials.append(r)
                if r["hit"]:
                    print(f"    HIT  score={r['score']:.3f} after {r['latency_s']:.2f}s")
                else:
                    print("    MISS")
                # Clear the refractory window before the next prompt.
                time.sleep(1.7)

            hits = sum(1 for t in trials if t["hit"])
            n = len(trials)
            tpr = hits / n if n else None
            lat = sorted(t["latency_s"] for t in trials if t["hit"])
            by_distance[f"{dist:g}m"] = {
                "n": n,
                "hits": hits,
                "tpr": round(tpr, 4) if tpr is not None else None,
                "median_wake_latency_s": lat[len(lat) // 2] if lat else None,
                "trials": trials,
            }
            if n:
                print(f"    {dist:g} m: TPR {hits}/{n} = {tpr * 100:.0f}%\n")

    total_n = sum(v["n"] for v in by_distance.values())
    total_hits = sum(v["hits"] for v in by_distance.values())
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "keyword": args.keyword,
        "speaker": args.speaker,
        "threshold": WAKE_SCORE_THRESHOLD,
        "overall": {
            "n": total_n,
            "hits": total_hits,
            "tpr": round(total_hits / total_n, 4) if total_n else None,
        },
        "by_distance": by_distance,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep earlier speakers so several team members accumulate into one file.
    existing = json.loads(args.out.read_text()) if args.out.exists() else {"runs": []}
    if "runs" not in existing:
        existing = {"runs": [existing]}
    existing["runs"].append(result)
    args.out.write_text(json.dumps(existing, indent=2))

    if total_n:
        print(f"overall TPR {total_hits}/{total_n} = {total_hits / total_n * 100:.1f}%")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
