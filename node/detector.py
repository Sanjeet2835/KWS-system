"""One interface, two backends. Swap the .tflite; do not rewrite the loop."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from shared.config import (
    DEBOUNCE_HITS,
    KEYWORD,
    SMOOTH_WINDOW,
    WAKE_SCORE_THRESHOLD,
    tflite_path,
)
from shared.feature_spec import FEATURE_SHAPE


class Detection:
    __slots__ = ("awake", "score", "label", "backend")

    def __init__(self, awake: bool, score: float, label: str, backend: str) -> None:
        self.awake = awake
        self.score = score
        self.label = label
        self.backend = backend


class Smoother:
    def __init__(self, window: int = SMOOTH_WINDOW, hits: int = DEBOUNCE_HITS) -> None:
        self.window = window
        self.hits = hits
        self._scores: deque[float] = deque(maxlen=window)
        self._above = 0

    def update(self, score: float, threshold: float) -> tuple[float, bool]:
        self._scores.append(score)
        avg = float(sum(self._scores) / len(self._scores))
        if avg >= threshold:
            self._above += 1
        else:
            self._above = 0
        return avg, self._above >= self.hits

    def reset(self) -> None:
        self._scores.clear()
        self._above = 0


class StubDetector:
    """Fires on sustained speech energy encoded as a dummy 'keyword' score.

    Used only to prove capture → stream → ASR → UI before a model exists.
    Never present this as the competition model.
    """

    labels = (KEYWORD, "unknown", "silence")

    def __init__(self) -> None:
        self.smoother = Smoother()
        self.backend = "stub"

    def infer(self, features: np.ndarray, energy: float) -> Detection:
        # Map RMS into a fake posterior so the rest of the pipeline is identical.
        score = float(np.clip((energy - 0.008) / 0.04, 0.0, 1.0))
        avg, fire = self.smoother.update(score, WAKE_SCORE_THRESHOLD)
        return Detection(fire, avg, KEYWORD if fire else "speech", self.backend)


class TFLiteDetector:
    labels = (KEYWORD, "unknown", "silence")

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or tflite_path()
        if not path.exists():
            raise FileNotFoundError(f"no model at {path}; train first or use --detector stub")
        self.backend = "tflite"
        self.path = path
        self.smoother = Smoother()
        self._interpreter, self._input, self._output, self._quant = self._load(path)

    @staticmethod
    def _load(path: Path):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter as _I

            class tflite:  # noqa: N801
                Interpreter = _I

        interpreter = tflite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        quant = inp.get("quantization", (0.0, 0))
        return interpreter, inp, out, quant

    def infer(self, features: np.ndarray, energy: float) -> Detection:
        del energy
        x = features.astype(np.float32).reshape((1, *FEATURE_SHAPE))
        scale, zero = self._quant
        if self._input["dtype"] == np.int8 and scale not in (0, 0.0):
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        self._interpreter.set_tensor(self._input["index"], x)
        self._interpreter.invoke()
        y = self._interpreter.get_tensor(self._output["index"])[0]
        if y.dtype != np.float32:
            oscale, ozero = self._output.get("quantization", (1.0, 0))
            y = (y.astype(np.float32) - ozero) * (oscale or 1.0)
        # softmax if the export left logits
        if y.min() < 0 or y.max() > 1.2:
            e = np.exp(y - y.max())
            y = e / e.sum()
        idx = int(np.argmax(y))
        label = self.labels[idx] if idx < len(self.labels) else f"class{idx}"
        keyword_score = float(y[0]) if len(y) else 0.0
        avg, fire = self.smoother.update(keyword_score, WAKE_SCORE_THRESHOLD)
        return Detection(fire and label == KEYWORD, avg, label, self.backend)


def load_detector(kind: str, model_path: Path | None = None):
    if kind == "stub":
        return StubDetector()
    if kind == "tflite":
        return TFLiteDetector(model_path)
    if kind == "auto":
        path = model_path or tflite_path()
        if path.exists():
            return TFLiteDetector(path)
        return StubDetector()
    raise ValueError(f"unknown detector {kind}")
