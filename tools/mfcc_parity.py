"""Prove the C MFCC equals the Python MFCC, on the host, before flashing.

The whole design rests on the ESP32 computing the same features the model was
trained on. If they drift, accuracy quietly collapses and the cause is invisible
from the outside — so this compiles firmware/src/mfcc.cpp natively with clang,
runs both implementations on the same clips, and reports the worst deviation.

  python -m tools.mfcc_parity
  python -m tools.mfcc_parity --wav data/sahayak/keyword/kw_000.wav
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import float_to_int16, load_wav_mono_16k  # noqa: E402
from node.features import features_from_clip, mfcc_frame  # noqa: E402
from shared.config import DATA_DIR, KEYWORD  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, SPEC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FW = ROOT / "firmware"

HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "mfcc.h"

int main(int argc, char **argv) {
  static int16_t clip[%(clip)d];
  static float out[%(frames)d * %(mfcc)d];
  FILE *f = fopen(argv[1], "rb");
  if (!f) return 2;
  if (fread(clip, sizeof(int16_t), %(clip)d, f) != %(clip)d) return 3;
  fclose(f);
  mfcc_from_clip(clip, out);
  FILE *g = fopen(argv[2], "wb");
  fwrite(out, sizeof(float), %(frames)d * %(mfcc)d, g);
  fclose(g);
  return 0;
}
"""


def build_harness(workdir: Path) -> Path:
    src = workdir / "harness.cpp"
    src.write_text(
        HARNESS % {"clip": CLIP_SAMPLES, "frames": SPEC.n_frames, "mfcc": SPEC.n_mfcc}
    )
    exe = workdir / "harness"
    cmd = [
        "clang++",
        "-O2",
        "-std=c++17",
        f"-I{FW / 'include'}",
        str(src),
        str(FW / "src" / "mfcc.cpp"),
        "-lm",
        "-o",
        str(exe),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit("failed to compile the firmware MFCC on the host")
    return exe


def run_c(exe: Path, workdir: Path, clip_i16: np.ndarray) -> np.ndarray:
    raw = workdir / "in.pcm"
    out = workdir / "out.f32"
    raw.write_bytes(clip_i16.astype("<i2").tobytes())
    proc = subprocess.run([str(exe), str(raw), str(out)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"harness failed ({proc.returncode}): {proc.stderr}")
    return np.frombuffer(out.read_bytes(), dtype="<f4").reshape(SPEC.n_frames, SPEC.n_mfcc)


def python_features_from_int16(clip_i16: np.ndarray) -> np.ndarray:
    """The training path: int16 -> float -> highpass -> MFCC."""
    from node.capture import highpass, int16_to_float

    audio = highpass(int16_to_float(clip_i16))
    return features_from_clip(audio)[:, :, 0]


def collect_clips(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    if args.wav:
        audio = load_wav_mono_16k(args.wav)
        pcm = float_to_int16(audio[:CLIP_SAMPLES])
        if len(pcm) < CLIP_SAMPLES:
            pcm = np.pad(pcm, (0, CLIP_SAMPLES - len(pcm)))
        return [(args.wav.name, pcm)]

    rng = np.random.default_rng(0)
    clips: list[tuple[str, np.ndarray]] = [
        ("silence", np.zeros(CLIP_SAMPLES, dtype=np.int16)),
        ("white", float_to_int16(rng.normal(0, 0.1, CLIP_SAMPLES).astype(np.float32))),
        (
            "tone440",
            float_to_int16(
                0.4 * np.sin(2 * np.pi * 440 * np.arange(CLIP_SAMPLES) / SPEC.sample_rate).astype(np.float32)
            ),
        ),
        ("loud", float_to_int16(rng.uniform(-0.95, 0.95, CLIP_SAMPLES).astype(np.float32))),
    ]
    folder = DATA_DIR / KEYWORD / "keyword"
    for wav in sorted(folder.glob("*.wav"))[: args.n_real]:
        audio = load_wav_mono_16k(wav)[:CLIP_SAMPLES]
        pcm = float_to_int16(audio)
        if len(pcm) < CLIP_SAMPLES:
            pcm = np.pad(pcm, (0, CLIP_SAMPLES - len(pcm)))
        clips.append((wav.name, pcm))
    return clips


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--n-real", type=int, default=6, help="dataset clips to include")
    # Both sides are float32 but sum in different orders, so bit-exactness is not
    # achievable. What matters is staying far below one INT8 quantisation step of
    # the model input, which 1e-3 relative comfortably does.
    p.add_argument("--tol", type=float, default=1e-3, help="max allowed relative deviation")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not (FW / "include" / "mfcc_tables.h").exists():
        raise SystemExit("no mfcc_tables.h; run python -m training.export_firmware --tables-only")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        exe = build_harness(workdir)
        print(f"compiled {FW / 'src' / 'mfcc.cpp'} natively\n")

        worst = 0.0
        worst_name = ""
        rows = []
        for name, pcm in collect_clips(args):
            c = run_c(exe, workdir, pcm)
            p = python_features_from_int16(pcm)
            dev = float(np.max(np.abs(c - p)))
            scale = float(np.max(np.abs(p))) or 1.0
            rel = dev / scale
            rows.append((name, dev, rel))
            if rel > worst:
                worst, worst_name = rel, name

    print(f"{'clip':<28} {'max|C-Py|':>12} {'relative':>10}")
    for name, dev, rel in rows:
        flag = "" if rel <= args.tol else "  <-- OVER TOL"
        print(f"{name:<28} {dev:>12.6f} {rel:>10.2e}{flag}")

    print(f"\nworst relative deviation: {worst:.2e} on {worst_name} (tolerance {args.tol:.0e})")
    if worst > args.tol:
        print("FAIL: firmware and training features disagree. Fix before flashing.")
        return 1
    print("PASS: firmware MFCC matches node/features.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
