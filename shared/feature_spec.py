"""Single MFCC recipe. Pi Python and future ESP32 C must match this exactly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    sample_rate: int = 16000
    window_ms: int = 30
    hop_ms: int = 20
    n_mels: int = 40
    n_mfcc: int = 10
    n_frames: int = 49
    fmin_hz: float = 20.0
    fmax_hz: float = 4000.0
    preemphasis: float = 0.97
    # The 480-sample window is zero-padded to this power of two so the ESP32 can
    # use an esp-dsp radix-2 real FFT instead of an O(N^2) DFT.
    n_fft: int = 512
    # DC/rumble removal. Same 2nd-order Butterworth runs in Python and in C.
    highpass_hz: float = 70.0
    # Pi I2S overlay captures at this rate; we decimate to sample_rate.
    capture_rate: int = 48000


SPEC = FeatureSpec()

WINDOW_SAMPLES = int(SPEC.sample_rate * SPEC.window_ms / 1000)  # 480
HOP_SAMPLES = int(SPEC.sample_rate * SPEC.hop_ms / 1000)  # 320
CLIP_SAMPLES = WINDOW_SAMPLES + (SPEC.n_frames - 1) * HOP_SAMPLES  # 15840 ≈ 0.99 s
FFT_BINS = SPEC.n_fft // 2 + 1  # 257
FEATURE_SHAPE = (SPEC.n_frames, SPEC.n_mfcc, 1)
INT16_MAX = 32768.0
