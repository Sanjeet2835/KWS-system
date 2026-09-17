"""Record training clips on the Mac mic into data/<keyword>/*_real/.

The S3 INMP441 is still the gold-standard source (tools.record_s3). These Mac
clips close the accent / room-language gap so the next retrain hears *you*
before the live demo. Named the same way train.py already parses:

    <label>_<speaker>_<idx>.wav

Guided session (recommended):
  python -m tools.record_mac --speaker mayank --session

Or one label at a time:
  python -m tools.record_mac --label keyword --speaker mayank --count 40
  python -m tools.record_mac --label unknown --speaker mayank --count 40
  python -m tools.record_mac --label silence --speaker mayank --count 20
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import DATA_DIR, KEYWORD  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, SPEC  # noqa: E402

LABEL_DIRS = {
    "keyword": "keyword_real",
    "unknown": "unknown_real",
    "lookalike": "lookalike_real",
    "silence": "silence_real",
    "noise": "noise_real",
}

UNKNOWN_PROMPTS = (
    "yes",
    "no",
    "stop",
    "go",
    "left",
    "right",
    "up",
    "down",
    "on",
    "off",
    "house",
    "bird",
    "cat",
    "dog",
    "happy",
    "wow",
    "tree",
    "three",
    "four",
    "sheila",
    "sahayak",
    "sahayata",
    "hello",
    "okay",
    "zero",
    "one",
    "two",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "marvin wait",
    "not marvin",
    "maybe",
    "please",
    "thanks",
    "india",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--label", choices=tuple(LABEL_DIRS), default="keyword")
    p.add_argument("--speaker", default="mac", help="one token, no underscores")
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--clip-ms", type=int, default=1000)
    p.add_argument("--lead-ms", type=int, default=200)
    p.add_argument("--device", type=int, default=None, help="sounddevice input index")
    p.add_argument("--session", action="store_true", help="40 keyword + 40 unknown + 20 silence")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _open_stream(device: int | None):
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "sounddevice is not installed. In the project venv: pip install sounddevice"
        ) from exc
    return sd


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SPEC.sample_rate)
        wf.writeframes(pcm.astype("<i2").tobytes())


def level(pcm: np.ndarray) -> tuple[float, float]:
    f = pcm.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(f * f))), float(np.max(np.abs(f)))


def record_clip(sd, clip_ms: int, lead_ms: int, device: int | None) -> np.ndarray:
    n = int(SPEC.sample_rate * (clip_ms + lead_ms) / 1000)
    rec = sd.rec(n, samplerate=SPEC.sample_rate, channels=1, dtype="int16", device=device)
    sd.wait()
    audio = rec.reshape(-1)
    lead = int(SPEC.sample_rate * lead_ms / 1000)
    body = audio[lead:]
    if len(body) >= CLIP_SAMPLES:
        return body[:CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=np.int16)
    out[: len(body)] = body
    return out


def prompt_for(label: str, keyword: str, i: int) -> str:
    if label == "keyword":
        return keyword
    if label == "silence":
        return "(stay quiet — room tone)"
    if label == "noise":
        return "(talk / TV / anything except the keyword)"
    return UNKNOWN_PROMPTS[i % len(UNKNOWN_PROMPTS)]


def guided(args: argparse.Namespace, label: str, count: int, out_dir: Path) -> int:
    sd = _open_stream(args.device)
    existing = len(list(out_dir.glob(f"{label}_{args.speaker}_*.wav")))
    print(f"\n{count} clips  label={label}  speaker={args.speaker}")
    print(f"writing {out_dir}  (index starts at {existing})")
    print("Enter = record, s = skip, q = quit\n")
    saved = 0
    i = existing
    while saved < count:
        say = prompt_for(label, args.keyword, i)
        cmd = input(f"[{saved + 1}/{count}] say {say!r} > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        print("    listening…")
        pcm = record_clip(sd, args.clip_ms, args.lead_ms, args.device)
        rms, peak = level(pcm)
        if label != "silence" and peak >= 0.99:
            print(f"    clipped (peak {peak:.2f}) — redo farther from the mic")
            continue
        if label != "silence" and rms < 0.004:
            print(f"    too quiet (rms {rms:.4f}) — redo")
            continue
        path = out_dir / f"{label}_{args.speaker}_{i:03d}.wav"
        write_wav(path, pcm)
        print(f"    saved {path.name}  rms={rms:.4f} peak={peak:.2f}")
        saved += 1
        i += 1
    print(f"{saved} clips → {out_dir}")
    return saved


def main() -> None:
    args = parse_args()
    if "_" in args.speaker:
        raise SystemExit("--speaker must not contain underscores")
    jobs = (
        (("keyword", 40), ("unknown", 40), ("silence", 20))
        if args.session
        else ((args.label, args.count),)
    )
    total = 0
    try:
        for label, count in jobs:
            dest = args.out or (DATA_DIR / args.keyword / LABEL_DIRS[label])
            total += guided(args, label, count, dest)
    except KeyboardInterrupt:
        print("\nstopped")
    print(f"\ndone. {total} new clips. Retrain with: python -m training.train --keyword {args.keyword}")


if __name__ == "__main__":
    main()
