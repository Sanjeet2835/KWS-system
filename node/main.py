"""Always-on node: capture → gate → features → detect → stream.

Usage (laptop, no Pi yet):
  python -m node.main --source file --wav /tmp/speech.wav

Usage (Pi, I2S mic):
  python -m node.main --source i2s --detector auto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow `python -m node.main` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from node.capture import open_source  # noqa: E402
from node.chime import play_chime_async  # noqa: E402
from node.detector import load_detector  # noqa: E402
from node.energy import EnergyGate  # noqa: E402
from node.features import FeatureRing  # noqa: E402
from node.ring import SampleRing  # noqa: E402
from node.stream import AsrClient  # noqa: E402
from shared.config import (  # noqa: E402
    ASR_HOST,
    ASR_PORT,
    CHIME_MUTE_S,
    KEYWORD,
    PREROLL_MS,
    REFRACTORY_S,
    SILENCE_END_S,
    STREAM_TIMEOUT_S,
)
from shared.feature_spec import HOP_SAMPLES, SPEC  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sahayak KWS node")
    p.add_argument("--source", choices=("file", "i2s", "usb"), default="usb")
    p.add_argument("--wav", type=Path, default=None)
    p.add_argument("--device", default=None, help="sounddevice input id or name")
    p.add_argument("--detector", choices=("auto", "stub", "tflite"), default="auto")
    p.add_argument("--host", default=ASR_HOST)
    p.add_argument("--port", type=int, default=ASR_PORT)
    p.add_argument("--no-chime", action="store_true")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    detector = load_detector(args.detector)
    gate = EnergyGate()
    feats = FeatureRing()
    preroll = SampleRing(int(SPEC.sample_rate * PREROLL_MS / 1000))
    client = AsrClient(args.host, args.port)

    print(f"keyword={KEYWORD} detector={detector.backend} source={args.source} asr={args.host}:{args.port}")

    streaming = False
    session = None
    last_wake = -1e9
    muted_until = 0.0
    silence_s = 0.0
    stream_started = 0.0
    hop_s = HOP_SAMPLES / SPEC.sample_rate

    try:
        for hop in open_source(args.source, wav=args.wav, device=args.device):
            now = time.monotonic()
            preroll.push(hop)
            feats.push_hop(hop)
            speechy = gate.speechy(hop)

            if streaming and session is not None:
                session.send_audio(hop)
                if not speechy:
                    silence_s += hop_s
                else:
                    silence_s = 0.0
                if silence_s >= SILENCE_END_S or (now - stream_started) > STREAM_TIMEOUT_S:
                    session.close()
                    session = None
                    streaming = False
                    print("stream end")
                continue

            if now < muted_until or (now - last_wake) < REFRACTORY_S:
                continue
            if not speechy or not feats.ready():
                continue

            det = detector.infer(feats.copy(), gate.last_rms)
            if not det.awake:
                continue

            print(f"WAKE word={KEYWORD} score={det.score:.2f} backend={det.backend}", flush=True)
            last_wake = now
            muted_until = now + CHIME_MUTE_S
            if not args.no_chime:
                play_chime_async()
            try:
                session = client.session(preroll.snapshot())
            except OSError as exc:
                print(f"asr connect failed: {exc}")
                continue
            streaming = True
            stream_started = now
            silence_s = 0.0
    finally:
        if session is not None:
            session.close()


if __name__ == "__main__":
    run(parse_args())
