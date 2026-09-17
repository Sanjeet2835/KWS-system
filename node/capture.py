"""Turn hardware or a file into a 16 kHz mono int16 stream.

Sources:
  i2s  — Pi googlevoicehat / INMP441 (48 kHz S32 stereo → 16 kHz int16)
  usb  — any PortAudio / sounddevice input already at or near 16 kHz
  file — a WAV, for laptop bring-up before the Pi exists
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from scipy import signal

from shared.feature_spec import HOP_SAMPLES, INT16_MAX, SPEC


def int16_to_float(samples: np.ndarray) -> np.ndarray:
    return samples.astype(np.float32) / INT16_MAX


def float_to_int16(samples: np.ndarray) -> np.ndarray:
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


def s32_stereo_to_int16_mono(raw: np.ndarray) -> np.ndarray:
    """INMP441: 24-bit MSB-aligned in 32-bit slots, left channel if L/R is GND."""
    if raw.ndim == 1:
        left = raw
    else:
        left = raw[:, 0]
    # Arithmetic shift: keep the top 16 of the 24 meaningful bits.
    return (left >> 16).astype(np.int16)


def decimate_48k_to_16k(x: np.ndarray) -> np.ndarray:
    """Exact 3:1. We own the filter so Pi and tests stay reproducible."""
    return signal.resample_poly(x.astype(np.float32), 1, 3).astype(np.float32)


def highpass_sos(cutoff_hz: float = SPEC.highpass_hz, sr: int = SPEC.sample_rate) -> np.ndarray:
    """The one highpass definition. training.export_firmware ships these taps to C."""
    return signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos").astype(np.float64)


_HP_SOS = highpass_sos()


def highpass(x: np.ndarray, cutoff_hz: float = SPEC.highpass_hz, sr: int = SPEC.sample_rate) -> np.ndarray:
    sos = _HP_SOS if (cutoff_hz == SPEC.highpass_hz and sr == SPEC.sample_rate) else highpass_sos(cutoff_hz, sr)
    return signal.sosfilt(sos, x).astype(np.float32)


def load_wav_mono_16k(path: Path) -> np.ndarray:
    """Load any common WAV and return float32 mono at 16 kHz."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 4:
        data = np.frombuffer(raw, dtype="<i4")
        data = s32_stereo_to_int16_mono(data.reshape(-1, nch) if nch > 1 else data)
        audio = int16_to_float(data)
    elif sw == 2:
        data = np.frombuffer(raw, dtype="<i2")
        if nch > 1:
            data = data.reshape(-1, nch)[:, 0]
        audio = int16_to_float(data)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if sr == SPEC.capture_rate:
        audio = decimate_48k_to_16k(audio)
    elif sr != SPEC.sample_rate:
        g = np.gcd(sr, SPEC.sample_rate)
        audio = signal.resample_poly(audio, SPEC.sample_rate // g, sr // g).astype(np.float32)
    return highpass(audio)


def frames_from_array(audio: np.ndarray, hop: int = HOP_SAMPLES) -> Iterator[np.ndarray]:
    """Yield hop-sized float32 frames. Pads the tail with zeros once."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    n = len(audio)
    for start in range(0, n, hop):
        chunk = audio[start : start + hop]
        if len(chunk) < hop:
            padded = np.zeros(hop, dtype=np.float32)
            padded[: len(chunk)] = chunk
            yield padded
            return
        yield chunk


def iter_wav_frames(path: Path) -> Iterator[np.ndarray]:
    yield from frames_from_array(load_wav_mono_16k(path))


def iter_sounddevice_frames(device: int | str | None = None, samplerate: int | None = None) -> Iterator[np.ndarray]:
    """Laptop / USB mic. Optional: used when I2S is not available."""
    import sounddevice as sd

    sr = samplerate or SPEC.sample_rate
    hop = HOP_SAMPLES if sr == SPEC.sample_rate else int(sr * SPEC.hop_ms / 1000)
    with sd.InputStream(device=device, channels=1, samplerate=sr, dtype="float32", blocksize=hop) as stream:
        while True:
            block, _ = stream.read(hop)
            mono = block[:, 0]
            if sr == SPEC.capture_rate:
                mono = decimate_48k_to_16k(mono)
            elif sr != SPEC.sample_rate:
                g = np.gcd(sr, SPEC.sample_rate)
                mono = signal.resample_poly(mono, SPEC.sample_rate // g, sr // g).astype(np.float32)
            if len(mono) < HOP_SAMPLES:
                out = np.zeros(HOP_SAMPLES, dtype=np.float32)
                out[: len(mono)] = mono[:HOP_SAMPLES]
                yield highpass(out)
            else:
                yield highpass(mono[:HOP_SAMPLES])


def iter_alsa_i2s_frames(card: str = "sndrpigooglevoi") -> Iterator[np.ndarray]:
    """Pi I2S path. Requires python-alsaaudio (installed on the Pi, not on macOS)."""
    import alsaaudio

    inp = alsaaudio.PCM(
        alsaaudio.PCM_CAPTURE,
        alsaaudio.PCM_NORMAL,
        device=f"plughw:CARD={card},DEV=0",
    )
    inp.setchannels(2)
    inp.setrate(SPEC.capture_rate)
    inp.setformat(alsaaudio.PCM_FORMAT_S32_LE)
    # 20 ms at 48 kHz stereo = 960 frames. After /3 we have 320 @ 16 kHz.
    period = int(SPEC.capture_rate * SPEC.hop_ms / 1000)
    inp.setperiodsize(period)
    while True:
        length, data = inp.read()
        if length <= 0:
            continue
        raw = np.frombuffer(data, dtype="<i4")
        if raw.size < 2:
            continue
        stereo = raw.reshape(-1, 2)
        mono16 = s32_stereo_to_int16_mono(stereo)
        audio = decimate_48k_to_16k(int16_to_float(mono16))
        if len(audio) < HOP_SAMPLES:
            out = np.zeros(HOP_SAMPLES, dtype=np.float32)
            out[: len(audio)] = audio
            yield highpass(out)
        else:
            yield highpass(audio[:HOP_SAMPLES])


def open_source(kind: str, wav: Path | None = None, device: int | str | None = None) -> Iterator[np.ndarray]:
    if kind == "file":
        if wav is None:
            raise ValueError("--wav is required for --source file")
        yield from iter_wav_frames(wav)
        return
    if kind == "i2s":
        yield from iter_alsa_i2s_frames()
        return
    if kind in ("usb", "mic"):
        yield from iter_sounddevice_frames(device=device)
        return
    raise ValueError(f"unknown source {kind!r}; use file, i2s, or usb")
