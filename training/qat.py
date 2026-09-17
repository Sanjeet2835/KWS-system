"""INT8 QAT on an already-trained DS-CNN-S.

Blocked on this repo: tensorflow-model-optimization 0.8 rejects Keras 3
Functional models (`You passed an instance of type: Functional`). Do not run
until we save a tf.keras (v2) checkpoint or tfmot catches up.

Far-field gain is currently train-time waveform copies + RMS match (training.train)
plus firmware clip_match_train_level. Next accuracy step is more 1 m S3 clips,
not QAT.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import KEYWORD, MODELS_DIR  # noqa: E402
from training.train import collect, pick_holdout, report  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--keyword", default=KEYWORD)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--holdout-speaker", default=None)
    p.add_argument("--noise-cap", type=int, default=400)
    return p.parse_args()


def quantize(model):
    import tensorflow_model_optimization as tfmot
    from tensorflow import keras

    annotate = tfmot.quantization.keras.quantize_annotate_layer

    def clone(layer):
        if isinstance(layer, keras.layers.ZeroPadding2D):
            return layer
        try:
            return annotate(layer)
        except Exception:
            return layer

    annotated = keras.models.clone_model(model, clone_function=clone)
    annotated.set_weights(model.get_weights())
    return tfmot.quantization.keras.quantize_apply(annotated)


def main() -> None:
    raise SystemExit(
        "QAT skipped: tfmot does not accept Keras 3 Functional models. "
        "The flashed net already has far-field augment + RMS match. "
        "Record 1 m keyword clips with record_s3 and retrain instead."
    )
    args = parse_args()
    keras_path = MODELS_DIR / f"{args.keyword}.keras"
    if not keras_path.exists():
        raise SystemExit(f"no {keras_path}; train first")

    from tensorflow import keras

    root = __import__("shared.config", fromlist=["DATA_DIR"]).DATA_DIR / args.keyword
    gsc = (root / "split.json").exists()
    holdout = None if gsc else pick_holdout(root, args.holdout_speaker)
    train, val, _, _ = collect(args.keyword, holdout, args.noise_cap)
    x_tr, y_tr, _ = train.arrays()
    x_va, y_va, _ = val.arrays()
    print(f"qat train n={len(x_tr)} val n={len(x_va)}")

    base = keras.models.load_model(keras_path, compile=False)
    q = quantize(base)
    q.compile(
        optimizer=keras.optimizers.Adam(5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    q.fit(
        x_tr,
        y_tr,
        validation_data=(x_va, y_va) if len(x_va) else None,
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=2,
    )
    report(q, x_va, y_va, "qat-val") if len(x_va) else None
    q.save(keras_path)
    from training.export_firmware import export_all

    export_all(q, args.keyword)
    print("qat done; flash firmware next")


if __name__ == "__main__":
    main()
