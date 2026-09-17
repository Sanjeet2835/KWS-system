"""Fixed-size float32 ring. Preallocated at startup."""

from __future__ import annotations

import numpy as np


class SampleRing:
    def __init__(self, n_samples: int) -> None:
        self.n = int(n_samples)
        self._buf = np.zeros(self.n, dtype=np.float32)
        self._i = 0
        self._filled = 0

    def push(self, frame: np.ndarray) -> None:
        x = frame.astype(np.float32, copy=False)
        n = x.shape[0]
        if n >= self.n:
            self._buf[:] = x[-self.n :]
            self._i = 0
            self._filled = self.n
            return
        end = self._i + n
        if end <= self.n:
            self._buf[self._i : end] = x
        else:
            first = self.n - self._i
            self._buf[self._i :] = x[:first]
            self._buf[: n - first] = x[first:]
        self._i = (self._i + n) % self.n
        self._filled = min(self._filled + n, self.n)

    def snapshot(self) -> np.ndarray:
        if self._filled < self.n:
            out = np.zeros(self.n, dtype=np.float32)
            out[-self._filled :] = self._buf[: self._filled]
            return out
        return np.concatenate((self._buf[self._i :], self._buf[: self._i]))
