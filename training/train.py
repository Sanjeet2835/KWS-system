"""Train DS-CNN-S, score it the way the problem statement scores it, and export.

Dataset layout under data/<keyword>/:
    keyword/  unknown/  lookalike/  silence/        synthetic TTS (bootstrap)
    keyword_real/  unknown_real/  lookalike_real/   recorded through the S3 mic
    silence_real/  noise_real/                      room tone / TV / podcast

Real clips recorded by tools.record_s3 are named <label>_<speaker>_<idx>.wav, so
one speaker can be held out entirely — the validation split is then a speaker the
model has never heard, which is the only honest way to quote TPR.

  python -m training.train
  python -m training.train --epochs 40 --holdout-speaker mayank
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import load_wav_mono_16k  # noqa: E402
from node.features import features_from_clip  # noqa: E402
from shared.config import (  # noqa: E402
    DATA_DIR,
    DEBOUNCE_HITS,
    KEYWORD,
    MODELS_DIR,
    SMOOTH_WINDOW,
    WAKE_SCORE_THRESHOLD,
)
from shared.feature_spec import CLIP_SAMPLES, FEATURE_SHAPE  # noqa: E402
from training.augment import (  # noqa: E402
    NoiseBank,
    augment_waveform,
    match_train_level,
    spec_augment_batch,
)
from training.model import N_CLASSES, build_dscnn_s  # noqa: E402

LABELS = ("keyword", "unknown", "silence")

# folder -> class index. *_real folders hold clips through the actual mic.
SOURCES = (
    ("keyword", 0, False),
    ("keyword_real", 0, True),
    ("unknown", 1, False),
    ("unknown_real", 1, True),
    ("lookalike", 1, False),
    ("lookalike_real", 1, True),
    ("silence", 2, False),
    ("silence_real", 2, True),
)


def _speaker_of(path: Path) -> str:
    """Speech Commands: <speaker>_nohash_<n>.wav. Else <label>_<speaker>_<idx>."""
    stem = path.stem
    if "_nohash_" in stem:
        # unknown/yes_9db2b698_nohash_0.wav → 9db2b698 (last token before _nohash_)
        left = stem.split("_nohash_")[0]
        return left.split("_")[-1]
    parts = stem.split("_")
    return parts[-2] if len(parts) >= 3 else "tts"


def _split_val_set(root: Path) -> set[str] | None:
    path = root / "split.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    files = data.get("val_files")
    return set(files) if files else None


class Split:
    def __init__(self) -> None:
        self.x: list[np.ndarray] = []
        self.y: list[int] = []
        self.real: list[bool] = []
        self.speaker: list[str] = []

    def add(self, feat: np.ndarray, label: int, real: bool, speaker: str) -> None:
        self.x.append(feat)
        self.y.append(label)
        self.real.append(real)
        self.speaker.append(speaker)

    def arrays(self):
        if not self.x:
            return (
                np.zeros((0, *FEATURE_SHAPE), np.float32),
                np.zeros((0,), np.int32),
                np.zeros((0,), bool),
            )
        return (
            np.stack(self.x).astype(np.float32),
            np.array(self.y, np.int32),
            np.array(self.real, bool),
        )

    def __len__(self) -> int:
        return len(self.x)


def slice_noise_as_unknown(root: Path, cap: int) -> list[np.ndarray]:
    """Long TV/podcast/room recordings become non-wake training clips.

    These are the samples that buy a low false-accept rate: they are exactly the
    audio the node will hear for hours without being spoken to.
    """
    folder = root / "noise_real"
    if not folder.exists() or cap <= 0:
        return []
    rng = np.random.default_rng(1)
    out: list[np.ndarray] = []
    wavs = sorted(folder.glob("*.wav"))
    if not wavs:
        return []
    per_file = max(1, cap // len(wavs))
    for wav in wavs:
        audio = load_wav_mono_16k(wav)
        if len(audio) < CLIP_SAMPLES:
            continue
        starts = rng.integers(0, len(audio) - CLIP_SAMPLES, size=per_file)
        for s in starts:
            out.append(features_from_clip(match_train_level(audio[int(s) : int(s) + CLIP_SAMPLES])))
    return out[:cap]


def collect(
    keyword: str,
    holdout_speaker: str | None,
    noise_cap: int,
    max_clips_per_folder: int | None = None,
):
    root = DATA_DIR / keyword
    if not root.exists():
        raise SystemExit(
            f"no dataset at {root}; run python -m training.ingest_speech_commands "
            f"or python -m training.generate_dataset --keyword {keyword}"
        )

    train, val = Split(), Split()
    speakers: set[str] = set()
    val_set = _split_val_set(root)
    rng = np.random.default_rng(2)
    noise_bank = NoiseBank([root / "noise_real"], rng)

    for folder_name, label, is_real in SOURCES:
        folder = root / folder_name
        if not folder.exists():
            continue
        wavs = sorted(folder.glob("*.wav"))
        if max_clips_per_folder is not None:
            # Keep both official-split sides present in a smoke run. This option
            # is intentionally opt-in; normal training still consumes all data.
            if val_set is not None:
                val_wavs = [w for w in wavs if f"{folder_name}/{w.name}" in val_set]
                train_wavs = [w for w in wavs if f"{folder_name}/{w.name}" not in val_set]
                wavs = train_wavs[:max_clips_per_folder] + val_wavs[:max_clips_per_folder]
            else:
                wavs = wavs[:max_clips_per_folder]
        for wav in wavs:
            speaker = _speaker_of(wav)
            if speaker != "tts":
                speakers.add(speaker)
            rel = f"{folder_name}/{wav.name}"
            # Speech Commands clips are real speakers (not our mic). Treat them as
            # real so INT8 calibration and the metrics file say so.
            real = is_real or "_nohash_" in wav.stem or (val_set is not None)
            audio = load_wav_mono_16k(wav)
            if len(audio) < CLIP_SAMPLES:
                pad = np.zeros(CLIP_SAMPLES, np.float32)
                pad[: len(audio)] = audio
                audio = pad
            else:
                audio = audio[:CLIP_SAMPLES].astype(np.float32)
            feat = features_from_clip(match_train_level(audio))
            n_loaded = len(train) + len(val)
            if n_loaded and n_loaded % 1000 == 0:
                print(f"  loaded {n_loaded} clips...", flush=True)
            if val_set is not None:
                dest = val if rel in val_set else train
            else:
                dest = val if (holdout_speaker and speaker == holdout_speaker) else train
            dest.add(feat, label, real, speaker)
            # Far-field copies of the keyword only (2×). Val stays clean.
            if dest is train and label == 0:
                for _ in range(2):
                    aug = augment_waveform(audio, noise_bank, rng, p_noise=0.9)
                    dest.add(features_from_clip(match_train_level(aug)), label, real, speaker)

    for feat in slice_noise_as_unknown(root, noise_cap):
        train.add(feat, 1, True, "noise")

    if not len(train):
        raise SystemExit(f"{root} has no wav files")
    return train, val, sorted(speakers), val_set is not None


def pick_holdout(root: Path, requested: str | None) -> str | None:
    """Choose which speaker becomes the validation set.

    Prefers a real S3-mic speaker (keyword_real/). If the dataset is still the
    TTS bootstrap, holds out one named voice instead so validation is at least a
    voice the trainer never saw — still not a live-mic number, but not a copy of
    the training renders either.
    """
    if requested:
        return requested
    real = root / "keyword_real"
    if real.exists():
        found = sorted({_speaker_of(w) for w in real.glob("*.wav")} - {"tts"})
        if len(found) >= 2:
            return found[-1]
        if len(found) == 1:
            return found[0]
    synth = root / "keyword"
    voices = sorted({_speaker_of(w) for w in synth.glob("*.wav")} - {"tts"}) if synth.exists() else []
    if len(voices) >= 2:
        return voices[-1]
    return None


class AugmentedBatches:
    """Feeds SpecAugment-masked batches. Regenerated every epoch by tf.data.

    Keyword clips are oversampled 2× and masked more aggressively so the
    minority class is not drowned by the unknown-word sea.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, batch: int, seed: int = 0) -> None:
        self.x, self.y, self.batch = x, y, batch
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        kw = np.where(self.y == 0)[0]
        extra = self.rng.choice(kw, size=len(kw), replace=True) if len(kw) else np.zeros(0, int)
        order = self.rng.permutation(np.concatenate([np.arange(len(self.x)), extra]))
        for i in range(0, len(order), self.batch):
            sel = order[i : i + self.batch]
            heavy = self.y[sel] == 0
            batch = spec_augment_batch(self.x[sel], self.rng)
            if heavy.any():
                batch[heavy] = spec_augment_batch(
                    batch[heavy],
                    self.rng,
                    n_time_masks=3,
                    max_time_mask=10,
                    n_freq_masks=2,
                    max_freq_mask=3,
                    p=1.0,
                )
            yield batch, self.y[sel]


def wake_metrics(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Per-clip wake decision at a threshold, which is what the node actually does."""
    p_kw = probs[:, 0]
    is_kw = y == 0
    n_kw = int(is_kw.sum())
    n_neg = int((~is_kw).sum())
    tp = int((p_kw[is_kw] >= threshold).sum()) if n_kw else 0
    fp = int((p_kw[~is_kw] >= threshold).sum()) if n_neg else 0
    per_class_fa = {}
    for cls in (1, 2):
        m = y == cls
        per_class_fa[LABELS[cls]] = round(float((p_kw[m] >= threshold).mean()), 4) if int(m.sum()) else None
    return {
        "threshold": round(threshold, 3),
        "n_keyword": n_kw,
        "n_negative": n_neg,
        "tpr": round(tp / n_kw, 4) if n_kw else None,
        "far": round(fp / n_neg, 4) if n_neg else None,
        "false_accepts": fp,
        "far_by_class": per_class_fa,
    }


def report(model, x: np.ndarray, y: np.ndarray, tag: str) -> dict:
    if not len(x):
        print(f"[{tag}] empty split, skipped")
        return {}
    probs = model.predict(x, verbose=0)
    pred = probs.argmax(axis=1)
    acc = float((pred == y).mean())
    print(f"\n[{tag}] n={len(x)} accuracy={acc:.4f}")
    conf = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    print("        pred:  " + "  ".join(f"{c:>8}" for c in LABELS))
    for i, name in enumerate(LABELS):
        print(f"  true {name:>8}: " + "  ".join(f"{conf[i, j]:>8}" for j in range(N_CLASSES)))

    curve = [wake_metrics(probs, y, t) for t in np.arange(0.30, 0.96, 0.05)]
    at_default = wake_metrics(probs, y, WAKE_SCORE_THRESHOLD)
    print(
        f"  @thr={WAKE_SCORE_THRESHOLD}: TPR={at_default['tpr']} "
        f"FAR={at_default['far']} ({at_default['false_accepts']} false accepts "
        f"of {at_default['n_negative']} negatives)"
    )
    print("  threshold sweep (thr / TPR / FAR):")
    for row in curve:
        print(f"    {row['threshold']:.2f}  {row['tpr']}  {row['far']}")
    return {
        "split": tag,
        "n": int(len(x)),
        "accuracy": round(acc, 4),
        "confusion": conf.tolist(),
        "at_default_threshold": at_default,
        "sweep": curve,
    }


def save_representative(x_real: np.ndarray, x_all: np.ndarray, dest: Path, n: int = 300) -> None:
    """MFCCs for INT8 calibration. Real mic features first, synthetic to fill."""
    rng = np.random.default_rng(0)
    picks = []
    if len(x_real):
        k = min(len(x_real), n)
        picks.append(x_real[rng.choice(len(x_real), k, replace=False)])
    used = sum(len(p) for p in picks)
    if used < n and len(x_all):
        k = min(len(x_all), n - used)
        picks.append(x_all[rng.choice(len(x_all), k, replace=False)])
    rep = np.concatenate(picks).astype(np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, rep)
    real_n = len(picks[0]) if len(x_real) else 0
    print(f"wrote {dest} shape={rep.shape} ({real_n} real / {len(rep) - real_n} synthetic)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument(
        "--holdout-speaker",
        default=None,
        help="real speaker used as validation; default = last one found",
    )
    p.add_argument("--noise-cap", type=int, default=400, help="clips sliced out of noise_real/")
    p.add_argument(
        "--max-clips-per-folder",
        type=int,
        default=None,
        help="limit each source folder per train/validation split (smoke tests only)",
    )
    p.add_argument("--no-specaugment", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = DATA_DIR / args.keyword
    gsc_split_exists = (root / "split.json").exists()
    holdout = None if gsc_split_exists else pick_holdout(root, args.holdout_speaker)
    train, val, speakers, gsc_split = collect(
        args.keyword, holdout, args.noise_cap, args.max_clips_per_folder
    )

    x_tr, y_tr, real_tr = train.arrays()
    x_va, y_va, _ = val.arrays()
    preview = speakers[:8] + (["…"] if len(speakers) > 8 else [])
    print(f"real speakers found: {len(speakers)} {preview or ['none']}")
    if gsc_split:
        print("validation: official Speech Commands testing_list.txt (speaker-disjoint)")
        holdout = "speech_commands_testing_list"
    else:
        print(f"holdout speaker: {holdout or 'none (falling back to a random split)'}")

    if not len(x_va):
        # No real speaker to hold out yet: keep a random slice so training is still scored.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(x_tr))
        x_tr, y_tr, real_tr = x_tr[idx], y_tr[idx], real_tr[idx]
        n_val = max(1, len(x_tr) // 8)
        x_va, y_va = x_tr[:n_val], y_tr[:n_val]
        x_tr, y_tr, real_tr = x_tr[n_val:], y_tr[n_val:], real_tr[n_val:]

    print(f"train n={len(x_tr)} ({int(real_tr.sum())} real)  val n={len(x_va)}")
    print("train class counts", {LABELS[i]: int((y_tr == i).sum()) for i in range(N_CLASSES)})
    print("val   class counts", {LABELS[i]: int((y_va == i).sum()) for i in range(N_CLASSES)})

    import tensorflow as tf
    from tensorflow import keras

    model = build_dscnn_s()
    class KeywordRecall(keras.metrics.Metric):
        """Val accuracy is dominated by 'unknown'. This is the scored class."""

        def __init__(self, name="keyword_recall", **kwargs):
            super().__init__(name=name, **kwargs)
            self.tp = self.add_weight(name="tp", initializer="zeros")
            self.fn = self.add_weight(name="fn", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
            is_kw = tf.equal(y_true, 0)
            hit = tf.logical_and(is_kw, tf.equal(pred, 0))
            miss = tf.logical_and(is_kw, tf.not_equal(pred, 0))
            self.tp.assign_add(tf.reduce_sum(tf.cast(hit, tf.float32)))
            self.fn.assign_add(tf.reduce_sum(tf.cast(miss, tf.float32)))

        def result(self):
            return self.tp / (self.tp + self.fn + 1e-7)

        def reset_state(self):
            self.tp.assign(0.0)
            self.fn.assign(0.0)

    model.compile(
        optimizer=keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", KeywordRecall()],
    )

    counts = {i: int((y_tr == i).sum()) for i in range(N_CLASSES)}
    max_c = max(counts.values()) or 1
    class_weight = {i: max_c / max(1, counts[i]) for i in range(N_CLASSES)}
    # Extra lift on the keyword class — TPR is the judged accuracy metric.
    class_weight[0] = class_weight[0] * 1.35
    print("class_weight", class_weight)

    class BestWakeCallback(keras.callbacks.Callback):
        """Ignore the all-keyword first epoch. Keep the best recall once val acc is real."""

        def __init__(self, patience: int = 12, min_val_acc: float = 0.90) -> None:
            super().__init__()
            self.patience = patience
            self.min_val_acc = min_val_acc
            self.best = -1.0
            self.wait = 0
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            acc = float(logs.get("val_accuracy") or 0.0)
            rec = float(logs.get("val_keyword_recall") or 0.0)
            if acc < self.min_val_acc:
                self.wait += 1
            elif rec > self.best + 1e-4:
                self.best = rec
                self.wait = 0
                self.best_weights = [w.copy() for w in self.model.get_weights()]
                print(f"  best val keyword recall={rec:.4f} (val acc={acc:.4f})", flush=True)
            else:
                self.wait += 1
            if self.wait >= self.patience:
                print(
                    f"Epoch {epoch + 1}: early stop, restoring recall={self.best:.4f}",
                    flush=True,
                )
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                self.model.stop_training = True

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        BestWakeCallback(patience=12),
    ]

    if args.no_specaugment:
        model.fit(
            x_tr,
            y_tr,
            validation_data=(x_va, y_va),
            epochs=args.epochs,
            batch_size=args.batch,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )
    else:
        gen = AugmentedBatches(x_tr, y_tr, args.batch)
        steps = max(1, len(x_tr) // args.batch)
        ds = (
            tf.data.Dataset.from_generator(
                gen,
                output_signature=(
                    tf.TensorSpec(shape=(None, *FEATURE_SHAPE), dtype=tf.float32),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ),
            )
            .repeat()
            .prefetch(tf.data.AUTOTUNE)
        )
        model.fit(
            ds,
            validation_data=(x_va, y_va),
            epochs=args.epochs,
            steps_per_epoch=steps,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )

    MODELS_DIR.mkdir(exist_ok=True)
    keras_path = MODELS_DIR / f"{args.keyword}.keras"
    model.save(keras_path)
    print(f"wrote {keras_path}")

    metrics = {
        "keyword": args.keyword,
        "holdout_speaker": holdout,
        "real_speakers": speakers[:64],
        "n_real_speakers": len(speakers),
        "val_is_speaker_disjoint": bool(gsc_split or holdout),
        "source": "speech_commands_v0.02" if gsc_split else "local",
        "smoothing": {"window": SMOOTH_WINDOW, "debounce_hits": DEBOUNCE_HITS},
        "splits": [],
    }
    for tag, xs, ys in (("train", x_tr, y_tr), ("val", x_va, y_va)):
        got = report(model, xs, ys, tag)
        if got:
            metrics["splits"].append(got)
    real_val = val.arrays()[2] if len(val) else None
    if real_val is not None and len(x_va) and bool(real_val.all()):
        metrics["val_is_real_speaker"] = True

    metrics_path = MODELS_DIR / f"{args.keyword}.metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")

    save_representative(x_tr[real_tr], x_tr, MODELS_DIR / f"{args.keyword}.rep.npy")

    from training.export_firmware import export_all

    export_all(model, args.keyword)

    if not holdout and not gsc_split:
        print(
            "\n"
            "NOTE: no real-speaker holdout, so the accuracy above is measured on augmented\n"
            "copies of the same handful of TTS renders that the model trained on. Expect it\n"
            "to read near 100% and to mean nothing about live performance. Record real clips\n"
            "with `python -m tools.record_s3 --label keyword --speaker <name> --count 30` for\n"
            "at least two speakers, then retrain — the last speaker found becomes validation.\n"
            "The quantiser is calibrated on this same synthetic set until then."
        )


if __name__ == "__main__":
    main()
