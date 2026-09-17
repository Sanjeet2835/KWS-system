"""Phase A: plot a capture and print Gate A checks.

  python -m tools.plot_waveform /tmp/mic_test.wav
  python -m tools.plot_waveform /tmp/mic_test.wav --save /tmp/gate_a.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import load_wav_mono_16k  # noqa: E402
from shared.feature_spec import SPEC  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("wav", type=Path)
    p.add_argument("--save", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio = load_wav_mono_16k(args.wav)
    peak = float(np.max(np.abs(audio)))
    duration = len(audio) / SPEC.sample_rate
    rms = float(np.sqrt(np.mean(audio**2)))
    dc = float(np.mean(audio))
    print(f"samples={len(audio)}  duration={duration:.3f}s  peak={peak:.3f}  rms={rms:.4f}  dc={dc:.5f}")
    print("gate A:")
    print(f"  duration ok (not wildly off): {0.2 < duration < 60}")
    print(f"  no clip (peak < 0.99):        {peak < 0.99}")
    print(f"  has energy (rms > 0.005):     {rms > 0.005}")
    print(f"  dc near 0 (|dc| < 0.02):      {abs(dc) < 0.02}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    t = np.arange(len(audio)) / SPEC.sample_rate
    ax0.plot(t, audio, color="#1f6feb", linewidth=0.6)
    ax0.set_title("waveform 16 kHz mono")
    ax0.set_ylabel("amp")
    ax0.set_ylim(-1.05, 1.05)
    ax1.specgram(audio, NFFT=512, Fs=SPEC.sample_rate, noverlap=256, cmap="magma")
    ax1.set_title("spectrogram")
    ax1.set_ylabel("Hz")
    ax1.set_xlabel("s")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=120)
        print("wrote", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
