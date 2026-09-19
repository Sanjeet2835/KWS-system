"""Laptop checks that need no mic, no Pi, no TensorFlow."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import float_to_int16, frames_from_array, int16_to_float  # noqa: E402
from node.detector import StubDetector  # noqa: E402
from node.energy import EnergyGate  # noqa: E402
from node.features import FeatureRing, features_from_clip, log_mel_frame  # noqa: E402
from node.ring import SampleRing  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, FEATURE_SHAPE, HOP_SAMPLES, WINDOW_SAMPLES  # noqa: E402
from shared.protocol import pack_end, pack_frame, pack_header, unpack_header  # noqa: E402


def test_features() -> None:
    rng = np.random.default_rng(0)
    clip = rng.normal(0, 0.1, CLIP_SAMPLES).astype(np.float32)
    feats = features_from_clip(clip)
    assert feats.shape == FEATURE_SHAPE, feats.shape
    assert np.isfinite(feats).all()
    ring = FeatureRing()
    for hop in frames_from_array(clip):
        ring.push_hop(hop)
    assert ring.ready()
    assert ring.copy().shape == FEATURE_SHAPE
    row = log_mel_frame(clip[:WINDOW_SAMPLES])
    assert row.shape == (FEATURE_SHAPE[1],)


def test_ring() -> None:
    r = SampleRing(100)
    r.push(np.arange(60, dtype=np.float32))
    snap = r.snapshot()
    assert snap.shape == (100,)
    assert np.allclose(snap[-60:], np.arange(60))
    r.push(np.arange(60, 120, dtype=np.float32))
    snap = r.snapshot()
    assert snap.shape == (100,)
    assert np.allclose(snap, np.arange(20, 120))


def test_protocol() -> None:
    h = pack_header(500, 16000)
    preroll, sr = unpack_header(h)
    assert preroll == 500 and sr == 16000
    frame = pack_frame(b"\x00\x01" * 10)
    assert frame[:2] == b"\x14\x00"
    assert pack_end() == b"\x00\x00"


def test_stub_and_energy() -> None:
    gate = EnergyGate(threshold=0.01, consecutive=3)
    quiet = np.zeros(HOP_SAMPLES, dtype=np.float32)
    loud = np.full(HOP_SAMPLES, 0.2, dtype=np.float32)
    assert gate.speechy(quiet) is False
    assert gate.speechy(loud) is False
    assert gate.speechy(loud) is False
    assert gate.speechy(loud) is True
    det = StubDetector()
    feats = np.zeros(FEATURE_SHAPE, dtype=np.float32)
    # warm the smoother
    last = None
    for _ in range(8):
        last = det.infer(feats, energy=0.08)
    assert last is not None
    assert last.backend == "stub"


def test_int16_roundtrip() -> None:
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    y = int16_to_float(float_to_int16(x))
    assert np.max(np.abs(x - y)) < 4e-5


if __name__ == "__main__":
    test_features()
    test_ring()
    test_protocol()
    test_stub_and_energy()
    test_int16_roundtrip()
    print("smoke_test: all ok")
