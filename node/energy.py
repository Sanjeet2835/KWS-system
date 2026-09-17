"""Cheap silence skip. This is what keeps idle CPU down."""

from __future__ import annotations

import numpy as np

from shared.config import ENERGY_CONSECUTIVE_FRAMES, ENERGY_RMS_THRESHOLD


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))


class EnergyGate:
    def __init__(
        self,
        threshold: float = ENERGY_RMS_THRESHOLD,
        consecutive: int = ENERGY_CONSECUTIVE_FRAMES,
    ) -> None:
        self.threshold = threshold
        self.consecutive = consecutive
        self._hits = 0
        self.last_rms = 0.0

    def speechy(self, frame: np.ndarray) -> bool:
        self.last_rms = rms(frame)
        if self.last_rms >= self.threshold:
            self._hits = min(self._hits + 1, self.consecutive)
        else:
            self._hits = max(self._hits - 1, 0)
        return self._hits >= self.consecutive

    def reset(self) -> None:
        self._hits = 0
