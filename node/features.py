"""Reference MFCC. Firmware later must match these numbers frame-for-frame."""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft
from scipy.signal.windows import hann

from shared.feature_spec import (
    CLIP_SAMPLES,
    FEATURE_SHAPE,
    FFT_BINS,
    HOP_SAMPLES,
    SPEC,
    WINDOW_SAMPLES,
)


def _mel_filterbank() -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    n_fft_bins = FFT_BINS
    mels = np.linspace(hz_to_mel(SPEC.fmin_hz), hz_to_mel(SPEC.fmax_hz), SPEC.n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((SPEC.n_fft + 1) * hz / SPEC.sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft_bins - 1)
    fb = np.zeros((SPEC.n_mels, n_fft_bins), dtype=np.float32)
    for i in range(SPEC.n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center = min(left + 1, n_fft_bins - 1)
        if right == center:
            right = min(center + 1, n_fft_bins - 1)
        for j in range(left, center):
            fb[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            fb[i, j] = (right - j) / (right - center)
    return fb


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    mat = np.cos(np.pi * (n + 0.5) * k / n_mels).astype(np.float32)
    mat *= np.sqrt(2.0 / n_mels).astype(np.float32)
    mat[0] *= np.sqrt(0.5)
    return mat


_WINDOW = hann(WINDOW_SAMPLES, sym=False).astype(np.float32)
_FBANK = _mel_filterbank()
_DCT = _dct_matrix(SPEC.n_mfcc, SPEC.n_mels)


def mfcc_frame(window: np.ndarray) -> np.ndarray:
    """One 10-D MFCC vector from a 30 ms window of float32 audio."""
    if window.shape[0] != WINDOW_SAMPLES:
        raise ValueError(f"expected {WINDOW_SAMPLES} samples, got {window.shape[0]}")
    x = window.astype(np.float32)
    x[1:] = x[1:] - SPEC.preemphasis * x[:-1]
    # float32 end to end so the trained model sees the same numbers the ESP32
    # produces; tools/mfcc_parity.py holds the two implementations together.
    freq = rfft(x * _WINDOW, n=SPEC.n_fft)
    spec = (freq.real.astype(np.float32) ** 2 + freq.imag.astype(np.float32) ** 2).astype(np.float32)
    mel = np.maximum(_FBANK @ spec, np.float32(1e-10))
    log_mel = np.log(mel, dtype=np.float32)
    return (_DCT @ log_mel).astype(np.float32)


class FeatureRing:
    """Keeps the last n_frames MFCC rows. Preallocated. No growth in the hot path."""

    def __init__(self) -> None:
        self._buf = np.zeros(FEATURE_SHAPE, dtype=np.float32)
        self._audio = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._filled = 0

    def push_hop(self, hop: np.ndarray) -> np.ndarray:
        if hop.shape[0] != HOP_SAMPLES:
            raise ValueError(f"expected hop of {HOP_SAMPLES}, got {hop.shape[0]}")
        self._audio[:-HOP_SAMPLES] = self._audio[HOP_SAMPLES:]
        self._audio[-HOP_SAMPLES:] = hop
        self._filled = min(self._filled + HOP_SAMPLES, CLIP_SAMPLES)
        window = self._audio[-WINDOW_SAMPLES:]
        row = mfcc_frame(window)
        self._buf[:-1] = self._buf[1:]
        self._buf[-1, :, 0] = row
        return self._buf

    def ready(self) -> bool:
        return self._filled >= CLIP_SAMPLES

    def copy(self) -> np.ndarray:
        return self._buf.copy()


def features_from_clip(audio: np.ndarray) -> np.ndarray:
    """Compute a full 49×10×1 feature from ~1 s of 16 kHz float audio."""
    if len(audio) < CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        padded[: len(audio)] = audio
        audio = padded
    else:
        audio = audio[:CLIP_SAMPLES].astype(np.float32)
    out = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    for i in range(SPEC.n_frames):
        start = i * HOP_SAMPLES
        out[i, :, 0] = mfcc_frame(audio[start : start + WINDOW_SAMPLES])
    return out
