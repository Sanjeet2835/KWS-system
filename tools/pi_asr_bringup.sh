#!/usr/bin/env bash
# Pi ASR bring-up only: push UI, keep/install Vosk, restart sahayak-asr.
# Does not flash firmware or retrain.
#
#   ./tools/pi_asr_bringup.sh pi@10.50.97.202
#
set -euo pipefail
TARGET="${1:-pi@10.50.97.202}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${TARGET#*@}"

echo "=== SSH / ping $HOST ==="
ping -c 2 "$HOST" || true

echo "=== push (Vosk dirs excluded from --delete) ==="
bash "$ROOT/tools/push_to_pi.sh" "$TARGET"

echo "=== Vosk + restart on Pi ==="
ssh -o StrictHostKeyChecking=accept-new "$TARGET" bash -s <<'REMOTE'
set -euo pipefail
cd /home/pi/sahayak
mkdir -p models
SMALL=vosk-model-small-en-us-0.15
LGRAPH=vosk-model-en-us-0.22-lgraph
fetch_vosk() {
  local name="$1"
  local zip="/tmp/${name}.zip"
  if [ -d "models/$name" ]; then
    echo "$name already present"
    return 0
  fi
  echo "downloading $name…"
  local url="https://alphacephei.com/vosk/models/${name}.zip"
  if command -v curl >/dev/null; then
    curl -L --fail -o "$zip" "$url"
  else
    wget -O "$zip" "$url"
  fi
  python3 -m zipfile -e "$zip" models/
  rm -f "$zip"
}
# Keep the 40 MB small model as fallback; prefer the 128 MB lgraph for accuracy.
fetch_vosk "$SMALL"
fetch_vosk "$LGRAPH" || echo "lgraph download failed; ASR will use $SMALL"
ls -d models/vosk-model* 2>/dev/null || true

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt vosk
# Some vosk wheels have no __version__; import is the real check.
.venv/bin/python -c "from vosk import KaldiRecognizer, Model; print('vosk import ok')"

if sudo -n systemctl restart sahayak-asr 2>/dev/null; then
  echo "restarted via systemctl"
else
  echo "systemctl needs a password; sending SIGKILL to the old python (pygame ignores SIGTERM)"
  OLD=$(systemctl show -p MainPID --value sahayak-asr 2>/dev/null || echo 0)
  echo "old_pid=$OLD"
  if [ -n "$OLD" ] && [ "$OLD" != 0 ]; then
    kill -9 "$OLD" || true
  fi
  sleep 2
fi
echo "new_pid=$(systemctl show -p MainPID --value sahayak-asr 2>/dev/null || echo 0)"
systemctl is-active sahayak-asr || true
grep PRODUCT_NAME shared/config.py | head -1
grep PAGES server/ui.py | head -1
journalctl -u sahayak-asr -n 20 --no-pager | tail -20
REMOTE

echo "=== DONE ==="
