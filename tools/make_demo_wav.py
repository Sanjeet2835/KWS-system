"""Write a short synthetic clip so the laptop pipeline can be tested without a mic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import float_to_int16  # noqa: E402
from shared.feature_spec import SPEC  # noqa: E402


def speech_like(seconds: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SPEC.sample_rate)
    t = np.arange(n) / SPEC.sample_rate
    # Silence, then a formant-ish burst, then silence — enough for the stub.
    env = np.zeros(n, dtype=np.float32)
    start, end = int(0.4 * SPEC.sample_rate), int(1.3 * SPEC.sample_rate)
    env[start:end] = 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, end - start)))
    tone = (
        0.35 * np.sin(2 * np.pi * 180 * t)
        + 0.25 * np.sin(2 * np.pi * 340 * t)
        + 0.12 * np.sin(2 * np.pi * 510 * t)
    )
    noise = 0.04 * rng.normal(size=n)
    return np.clip((tone + noise) * env, -1, 1).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/demo_speech.wav"))
    p.add_argument("--seconds", type=float, default=2.4)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    audio = speech_like(args.seconds)
    import wave

    pcm = float_to_int16(audio)
    with wave.open(str(args.out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SPEC.sample_rate)
        wf.writeframes(pcm.tobytes())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
