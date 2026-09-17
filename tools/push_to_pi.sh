#!/usr/bin/env bash
# Copy this repo to the Pi over SSH after the SD card is imaged with SSH on.
#
#   ./tools/push_to_pi.sh pi@192.168.1.42
#
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@pi-hostname-or-ip [remote-dir]"
  exit 1
fi
TARGET="$1"
DEST="${2:-~/sahayak}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
rsync -avz --delete \
  --exclude '.venv' \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'data' \
  --exclude '.DS_Store' \
  --exclude 'models/vosk-model*' \
  --exclude 'models/*.zip' \
  --exclude '.pio' \
  --exclude 'firmware/.pio' \
  --exclude '*.o' \
  --exclude '*.elf' \
  --exclude '*.map' \
  "$ROOT/" "$TARGET:$DEST/"
echo "copied to $TARGET:$DEST"
echo "Vosk models on the Pi were left in place (excluded from --delete)."
