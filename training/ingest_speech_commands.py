"""Download Google Speech Commands v2 once and map it into data/<keyword>/.

marvin is a real-speaker word (~1.7k clips). The other 34 words become unknown,
and _background_noise_ is sliced into silence. Official testing_list.txt is the
validation set — it is speaker-disjoint from train, which is the number we quote.

  python -m training.ingest_speech_commands
  python -m training.ingest_speech_commands --keyword marvin --force

Does not use yes/no/go/stop as the keyword. Those stay in unknown/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import float_to_int16  # noqa: E402
from shared.config import DATA_DIR, KEYWORD  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, SPEC  # noqa: E402

ARCHIVE_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE_NAME = "speech_commands_v0.02.tar.gz"

# Too short / too common to be the wake word. Fine as unknowns.
SHORT_COMMANDS = frozenset({"yes", "no", "go", "stop", "on", "off", "up", "down"})
SKIP_DIRS = frozenset({"_background_noise_"})


def _load_raw_16k(path: Path) -> np.ndarray:
    """Float32 mono 16 kHz, no highpass — training applies that once."""
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw != 2:
        raise ValueError(f"{path.name}: expected int16, got width {sw}")
    data = np.frombuffer(raw, dtype="<i2")
    if nch > 1:
        data = data.reshape(-1, nch)[:, 0]
    audio = data.astype(np.float32) / 32768.0
    if sr != SPEC.sample_rate:
        from scipy import signal as sps

        g = np.gcd(sr, SPEC.sample_rate)
        audio = sps.resample_poly(audio, SPEC.sample_rate // g, sr // g).astype(np.float32)
    return audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--cache", type=Path, default=DATA_DIR / "_cache" / "speech_commands_v0.02")
    p.add_argument("--unknown-cap", type=int, default=5000, help="train unknowns besides the official test set")
    p.add_argument("--silence-cap", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _download(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    tar_path = cache.parent / ARCHIVE_NAME
    marker = cache / "testing_list.txt"
    if marker.exists():
        print(f"using existing unpack at {cache}")
        return cache
    if not tar_path.exists():
        print(f"downloading {ARCHIVE_URL}")
        print(f"  -> {tar_path}  (~2.3 GB, once)")
        tmp = tar_path.with_suffix(".part")
        urllib.request.urlretrieve(ARCHIVE_URL, tmp)
        tmp.rename(tar_path)
    print(f"unpacking {tar_path.name} ...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache)
    if not marker.exists():
        raise SystemExit(f"unpack of {tar_path} did not produce testing_list.txt")
    return cache


def _list_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = float_to_int16(audio[:CLIP_SAMPLES] if len(audio) >= CLIP_SAMPLES else audio)
    if len(pcm) < CLIP_SAMPLES:
        pad = np.zeros(CLIP_SAMPLES, dtype=np.int16)
        pad[: len(pcm)] = pcm
        pcm = pad
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SPEC.sample_rate)
        wf.writeframes(pcm.tobytes())


def _word_dirs(cache: Path) -> list[str]:
    out = []
    for p in sorted(cache.iterdir()):
        if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith("."):
            out.append(p.name)
    return out


def ingest(keyword: str, cache: Path, unknown_cap: int, silence_cap: int) -> dict:
    words = _word_dirs(cache)
    if keyword not in words:
        raise SystemExit(f"{keyword!r} is not a Speech Commands v2 word. Have: {', '.join(words)}")
    if keyword in SHORT_COMMANDS:
        raise SystemExit(f"{keyword!r} is too short for a wake word (high false-accept). Pick e.g. marvin / sheila / happy.")

    dest = DATA_DIR / keyword
    for sub in ("keyword", "unknown", "silence"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    testing = _list_file(cache / "testing_list.txt")
    validation = _list_file(cache / "validation_list.txt")
    held_out = testing | validation  # official val unused for train so test stays clean

    kw_src = cache / keyword
    keyword_files = sorted(kw_src.glob("*.wav"))
    val_files: list[str] = []
    n_kw_train = n_kw_val = 0
    for wav in keyword_files:
        rel = f"{keyword}/{wav.name}"
        target = dest / "keyword" / wav.name
        _link_or_copy(wav, target)
        if rel in testing:
            val_files.append(f"keyword/{wav.name}")
            n_kw_val += 1
        else:
            n_kw_train += 1

    other_words = [w for w in words if w != keyword]
    rng = random.Random(0)
    train_unknown: list[Path] = []
    n_unk_val = 0
    for word in other_words:
        folder = cache / word
        for wav in sorted(folder.glob("*.wav")):
            rel = f"{word}/{wav.name}"
            # Prefix the word so two speakers named the same hash in different
            # folders cannot collide, and so _speaker_of still sees `_nohash_`.
            dest_name = f"{word}_{wav.name}"
            dest_rel = f"unknown/{dest_name}"
            if rel in testing:
                _link_or_copy(wav, dest / "unknown" / dest_name)
                val_files.append(dest_rel)
                n_unk_val += 1
            elif rel not in held_out:
                train_unknown.append(wav)

    rng.shuffle(train_unknown)
    n_unk_train = 0
    for wav in train_unknown[:unknown_cap]:
        dest_name = f"{wav.parent.name}_{wav.name}"
        _link_or_copy(wav, dest / "unknown" / dest_name)
        n_unk_train += 1

    bg_dir = cache / "_background_noise_"
    silence_idx = 0
    n_sil_train = n_sil_val = 0
    bg_wavs = sorted(bg_dir.glob("*.wav")) if bg_dir.exists() else []
    slices: list[tuple[str, np.ndarray]] = []
    for wav in bg_wavs:
        try:
            audio = _load_raw_16k(wav)
        except Exception as exc:
            print(f"skip {wav.name}: {exc}")
            continue
        hop = CLIP_SAMPLES // 2
        for start in range(0, max(0, len(audio) - CLIP_SAMPLES) + 1, hop):
            slices.append((wav.stem, audio[start : start + CLIP_SAMPLES]))
    rng.shuffle(slices)
    slices = slices[:silence_cap]
    n_val_sil = max(1, len(slices) // 5)
    for i, (stem, clip) in enumerate(slices):
        name = f"silence_{stem}_{silence_idx:04d}.wav"
        silence_idx += 1
        _write_wav(dest / "silence" / name, clip)
        dest_rel = f"silence/{name}"
        if i < n_val_sil:
            val_files.append(dest_rel)
            n_sil_val += 1
        else:
            n_sil_train += 1

    summary = {
        "source": "speech_commands_v0.02",
        "license": "CC-BY-4.0",
        "keyword": keyword,
        "val_from": "testing_list.txt",
        "speaker_disjoint": True,
        "counts": {
            "train": {"keyword": n_kw_train, "unknown": n_unk_train, "silence": n_sil_train},
            "val": {"keyword": n_kw_val, "unknown": n_unk_val, "silence": n_sil_val},
        },
        "val_files": val_files,
        "n_val_files": len(val_files),
    }
    (dest / "split.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {dest / 'split.json'}")
    print("train", summary["counts"]["train"])
    print("val  ", summary["counts"]["val"])
    print(f"val is official Speech Commands testing_list.txt (speaker-disjoint)")
    return summary


def main() -> int:
    args = parse_args()
    dest = DATA_DIR / args.keyword
    split = dest / "split.json"
    if split.exists() and not args.force:
        data = json.loads(split.read_text())
        print(f"already ingested at {dest} (pass --force to redo)")
        print("train", data.get("counts", {}).get("train"))
        print("val  ", data.get("counts", {}).get("val"))
        return 0
    cache = _download(args.cache)
    ingest(args.keyword, cache, args.unknown_cap, args.silence_cap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
