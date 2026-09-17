"""Synthesize a starter dataset for whatever KEYWORD is in config.

Open-source TTS only: Piper if installed, else macOS `say`, else `espeak`.
This is a bootstrap. Replace / mix with real voices before you trust accuracy.

  python -m training.generate_dataset
  python -m training.generate_dataset --keyword go --n-keyword 40
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import float_to_int16  # noqa: E402
from shared.config import DATA_DIR, KEYWORD, lookalikes  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, SPEC  # noqa: E402
from training.augment import NoiseBank, augment_waveform  # noqa: E402

_NOISE: NoiseBank | None = None

UNKNOWN_WORDS = (
    "hello",
    "computer",
    "please",
    "window",
    "kitchen",
    "number",
    "today",
    "music",
    "water",
    "friend",
    "india",
    "college",
)


def _write_wav(path: Path, audio: np.ndarray, sr: int = SPEC.sample_rate) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = float_to_int16(audio)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _read_wav_any(path: Path) -> np.ndarray:
    from node.capture import load_wav_mono_16k

    return load_wav_mono_16k(path)


# Voice variety is the only speaker diversity a synthetic bootstrap can offer,
# and Indic-accented voices matter for an Indic wake word. Filtered at runtime to
# whatever this machine actually has installed.
PREFERRED_VOICES = ("Rishi", "Lekha", "Samantha", "Daniel", "Karen", "Alex", "Moira", "Tessa")


def available_voices() -> tuple[str, ...]:
    if not shutil.which("say"):
        return ()
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return ()
    installed = {line.split()[0] for line in listing.splitlines() if line.strip()}
    return tuple(v for v in PREFERRED_VOICES if v in installed)


def tts_to_wav(text: str, dest: Path, rate_wpm: int = 170, voice: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("piper"):
        subprocess.run(
            ["piper", "--model", os_piper_model(), "--output_file", str(dest)],
            input=text.encode(),
            check=True,
        )
        return
    if shutil.which("say"):
        aiff = dest.with_suffix(".aiff")
        cmd = ["say", "-r", str(rate_wpm), "-o", str(aiff)]
        if voice:
            cmd += ["-v", voice]
        subprocess.run(cmd + [text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(dest)],
            check=True,
        )
        aiff.unlink(missing_ok=True)
        return
    if shutil.which("espeak"):
        subprocess.run(
            ["espeak", "-s", str(rate_wpm), "-w", str(dest), text],
            check=True,
        )
        return
    raise RuntimeError("no TTS found: install piper, or use macOS `say`, or install espeak")


def os_piper_model() -> str:
    import os

    m = os.environ.get("PIPER_MODEL")
    if not m:
        raise RuntimeError("piper is installed but PIPER_MODEL is not set")
    return m


def fit_clip(audio: np.ndarray) -> np.ndarray:
    # load_wav_mono_16k already highpassed; filtering twice would skew the low end.
    audio = audio.astype(np.float32)
    if len(audio) >= CLIP_SAMPLES:
        # take a random window if longer
        start = random.randint(0, len(audio) - CLIP_SAMPLES)
        return audio[start : start + CLIP_SAMPLES]
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    pad = (CLIP_SAMPLES - len(audio)) // 2
    out[pad : pad + len(audio)] = audio
    return out


def augment(audio: np.ndarray) -> np.ndarray:
    """Waveform augmentation with noise mixed at INMP441-realistic SNRs."""
    return augment_waveform(audio, noise_bank=_NOISE, rng=np.random.default_rng(random.getrandbits(32)))


def synthesize_word(word: str, dest_dir: Path, n: int, prefix: str, voices: tuple[str, ...] = ()) -> None:
    """One TTS render per (voice, rate) combination, then several augmented copies.

    Rendering is the slow part, so we render once and augment many times: the
    network sees far more variation for the same wall-clock cost.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    rates = (140, 160, 180, 200)
    combos = [(v, r) for v in (voices or (None,)) for r in rates]
    random.shuffle(combos)
    n_renders = min(len(combos), max(1, n // 3))
    renders: list[tuple[str, np.ndarray]] = []
    for voice, rate in combos[:n_renders]:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "u.wav"
            try:
                tts_to_wav(word, raw, rate_wpm=rate, voice=voice)
                renders.append((voice or "default", fit_clip(_read_wav_any(raw))))
            except Exception as exc:
                print(f"TTS failed for {word!r} voice={voice}: {exc}")
    if not renders:
        print(f"no TTS output for {word!r}; writing tone placeholders")
        t = np.linspace(0, CLIP_SAMPLES / SPEC.sample_rate, CLIP_SAMPLES, endpoint=False)
        renders = [("tone", (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))]

    for i in range(n):
        voice, audio = renders[i % len(renders)]
        token = voice.replace("_", "")
        _write_wav(dest_dir / f"{prefix}_{token}_{i:03d}.wav", augment(audio))


def write_silence(dest_dir: Path, n: int) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        noise = np.random.normal(0, random.uniform(0.0004, 0.003), CLIP_SAMPLES).astype(np.float32)
        _write_wav(dest_dir / f"silence_{i:03d}.wav", noise)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--n-keyword", type=int, default=60)
    p.add_argument("--n-unknown", type=int, default=80)
    p.add_argument("--n-silence", type=int, default=40)
    p.add_argument("--n-lookalike", type=int, default=30)
    return p.parse_args()


def main() -> None:
    global _NOISE
    args = parse_args()
    root = DATA_DIR / args.keyword
    _NOISE = NoiseBank([root / "noise_real"], rng=np.random.default_rng(0))
    voices = available_voices()
    print(f"writing dataset under {root} for keyword={args.keyword!r}")
    print(f"noise bank: {len(_NOISE)} real clips" if len(_NOISE) else "noise bank: synthetic only")
    print(f"tts voices: {', '.join(voices) if voices else 'default'}")

    synthesize_word(args.keyword, root / "keyword", args.n_keyword, "kw", voices)
    for word in UNKNOWN_WORDS:
        synthesize_word(word, root / "unknown", max(1, args.n_unknown // len(UNKNOWN_WORDS)), word, voices)
    extras = lookalikes() if args.keyword == KEYWORD else ()
    for word in extras:
        synthesize_word(word, root / "lookalike", args.n_lookalike, word, voices)
    write_silence(root / "silence", args.n_silence)
    print("done. next: python -m training.train --keyword", args.keyword)


if __name__ == "__main__":
    main()
