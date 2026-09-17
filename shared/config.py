"""Project-wide knobs. Change KEYWORD here, then regenerate data and retrain."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Product name on the UI and serial banner. Independent of the spoken word.
PRODUCT_NAME = "Anuvani"

# The spoken wake word. Lowercase. One word is easiest.
# After changing this: ingest or generate_dataset → train → flash the node.
# marvin is a Speech Commands v2 word (thousands of real speakers). The PS
# forbids pretrained Alexa / Hey Google engines, not this dataset.
KEYWORD = os.environ.get("SAHAYAK_KEYWORD", "marvin").strip().lower()

# Lookalikes used as hard negatives when KEYWORD is sahayak.
# For marvin, unknowns come from the other Speech Commands words instead.
DEFAULT_LOOKALIKES = ("sahayata", "sahay", "sahayogi")

# Network. Node and server stay separate even on one Pi so an ESP32 can replace the node.
ASR_HOST = os.environ.get("SAHAYAK_ASR_HOST", "127.0.0.1")
ASR_PORT = int(os.environ.get("SAHAYAK_ASR_PORT", "8765"))
TELEMETRY_PORT = int(os.environ.get("SAHAYAK_TEL_PORT", "8766"))

# Detector behaviour
ENERGY_RMS_THRESHOLD = 0.003
ENERGY_CONSECUTIVE_FRAMES = 3
WAKE_SCORE_THRESHOLD = 0.52
SMOOTH_WINDOW = 3
DEBOUNCE_HITS = 3
REFRACTORY_S = 1.8
CHIME_MUTE_S = 0.6
SILENCE_END_S = 3.5
PREROLL_MS = 500
STREAM_TIMEOUT_S = 10.0

# UI
UI_WIDTH = 480
UI_HEIGHT = 320

# Paths
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"


def lookalikes() -> tuple[str, ...]:
    if KEYWORD == "sahayak":
        return DEFAULT_LOOKALIKES
    return ()


def tflite_path() -> Path:
    return MODELS_DIR / f"{KEYWORD}.int8.tflite"


def float_model_path() -> Path:
    return MODELS_DIR / f"{KEYWORD}.keras"
