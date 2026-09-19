"""Single MFCC recipe. Pi Python and future ESP32 C must match this exactly."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:

    sample_rate: int = 16000

    window_ms: int = 20
    hop_ms: int = 20

    n_mels: int = 20
    n_frames: int = 49

    fmin_hz: float = 20.0
    fmax_hz: float = 4000.0

    preemphasis: float = 0.97

    n_fft: int = 512

    highpass_hz: float = 70.0

    capture_rate: int = 48000


SPEC = FeatureSpec()

WINDOW_SAMPLES = int(
    SPEC.sample_rate * SPEC.window_ms / 1000
)  # 320

HOP_SAMPLES = int(
    SPEC.sample_rate * SPEC.hop_ms / 1000
)  # 320

CLIP_SAMPLES = (
    WINDOW_SAMPLES
    + (SPEC.n_frames - 1) * HOP_SAMPLES
)  # 15680 ≈ 0.98 s

FFT_BINS = SPEC.n_fft // 2 + 1  # 257

FEATURE_SHAPE = (
    SPEC.n_frames,
    SPEC.n_mels,
    1,
)  # (49, 20, 1)

INT16_MAX = 32768.0
