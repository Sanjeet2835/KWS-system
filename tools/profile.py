"""Print KWS-subsystem numbers we can put on the deck.

On the Pi this is honest *subsystem* RAM, not whole-OS RAM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.features import FeatureRing  # noqa: E402
from node.ring import SampleRing  # noqa: E402
from shared.config import KEYWORD, PREROLL_MS, tflite_path  # noqa: E402
from shared.feature_spec import CLIP_SAMPLES, FEATURE_SHAPE, HOP_SAMPLES, SPEC  # noqa: E402


def main() -> None:
    preroll = SampleRing(int(SPEC.sample_rate * PREROLL_MS / 1000))
    feats = FeatureRing()
    hop = np.zeros(HOP_SAMPLES, dtype=np.float32)
    preroll.push(hop)
    feats.push_hop(hop)

    buf_bytes = preroll._buf.nbytes + feats._buf.nbytes + feats._audio.nbytes
    print(f"keyword={KEYWORD}")
    print(f"feature_shape={FEATURE_SHAPE}  clip_samples={CLIP_SAMPLES}")
    print(f"preallocated_buffers_bytes={buf_bytes} ({buf_bytes / 1024:.1f} KB)")

    path = tflite_path()
    if path.exists():
        flash = path.stat().st_size
        print(f"tflite_flash_bytes={flash} ({flash / 1024:.1f} KB)  file={path.name}")
        try:
            from node.detector import TFLiteDetector

            det = TFLiteDetector(path)
            print(f"tflite_loaded backend={det.backend}")
        except Exception as exc:
            print(f"tflite load skipped: {exc}")
    else:
        print(f"no INT8 model yet at {path} (train to get flash + arena numbers)")

    print("idle CPU: pin the node to one core on the Pi and use `top -d 1`.")
    print("Do not report whole-board RAM as the KWS footprint.")


if __name__ == "__main__":
    main()
