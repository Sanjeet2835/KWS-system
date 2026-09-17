"""Short open-source-friendly beep. Generated in-process; no asset required."""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

from shared.config import ASSETS_DIR, CHIME_MUTE_S
from shared.feature_spec import SPEC


def render_chime(sr: int = SPEC.sample_rate, duration_s: float = 0.22) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    env = np.exp(-t * 8.0)
    wave = 0.35 * env * (np.sin(2 * np.pi * 880 * t) + 0.4 * np.sin(2 * np.pi * 1320 * t))
    return wave.astype(np.float32)


def play_chime_async() -> None:
    threading.Thread(target=_play, daemon=True).start()


def _play() -> None:
    audio = render_chime()
    ASSETS_DIR.mkdir(exist_ok=True)
    try:
        import sounddevice as sd

        sd.play(audio, SPEC.sample_rate, blocking=True)
        return
    except Exception:
        pass
    # Last resort: write a wav so aplay can be used on the Pi later.
    path = ASSETS_DIR / "chime.wav"
    _write_wav(path, audio)
    if sys.platform.startswith("linux"):
        import subprocess

        subprocess.Popen(["aplay", "-q", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(CHIME_MUTE_S)


def _write_wav(path, audio: np.ndarray) -> None:
    import wave

    pcm = np.clip(audio, -1, 1)
    pcm = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SPEC.sample_rate)
        wf.writeframes(pcm.tobytes())
