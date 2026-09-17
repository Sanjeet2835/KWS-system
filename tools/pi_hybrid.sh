#!/usr/bin/env bash
# Run ON the Pi (pi@sahayak). Pulls hybrid ASR from the Mac HTTP share,
# stores GEMINI_API_KEY in ~/sahayak/.env (mode 600), restarts sahayak-asr.
set -euo pipefail
MAC="${SAHAYAK_MAC_HTTP:-http://10.50.97.115:8000}"
cd /home/pi/sahayak

curl -fsSL -o server/asr_server.py "$MAC/asr_server.py"
curl -fsSL -o server/ui.py "$MAC/ui.py"
curl -fsSL -o server/state.py "$MAC/state.py"
curl -fsSL -o /tmp/sahayak-asr.service "$MAC/sahayak-asr.service"
if sudo -n true 2>/dev/null; then
  sudo cp /tmp/sahayak-asr.service /etc/systemd/system/sahayak-asr.service
  sudo systemctl daemon-reload
fi

if grep -q "_gemini_transcribe" server/asr_server.py; then
  echo "hybrid asr_server.py ok"
else
  echo "download did not contain gemini — abort" >&2
  exit 1
fi

if sudo -n true 2>/dev/null; then
  sudo apt-get install -y -qq fonts-noto-core fonts-lohit-deva >/dev/null 2>&1 || true
fi

if [ -f /home/pi/sahayak/.env ] && grep -q '^GEMINI_API_KEY=.' /home/pi/sahayak/.env; then
  echo "keeping existing .env"
else
  if [ -z "${GEMINI_API_KEY:-}" ]; then
    echo
    echo "Paste Gemini API key, then Enter (hidden, not shown):"
    IFS= read -r -s GEMINI_API_KEY
    echo
  fi
  GEMINI_API_KEY="$(printf '%s' "${GEMINI_API_KEY:-}" | tr -d '[:space:]')"
  if [ -z "$GEMINI_API_KEY" ]; then
    echo "no key — keeping local whisper only"
  else
    umask 077
    printf 'GEMINI_API_KEY=%s\nSAHAYAK_GEMINI_MODEL=gemini-2.5-flash\nSAHAYAK_WHISPER=distil-small.en\n' \
      "$GEMINI_API_KEY" > /home/pi/sahayak/.env
    chmod 600 /home/pi/sahayak/.env
    echo "wrote /home/pi/sahayak/.env (mode 600)"
  fi
fi
unset GEMINI_API_KEY

OLD=$(systemctl show -p MainPID --value sahayak-asr 2>/dev/null || echo 0)
echo "old_pid=$OLD"
if sudo -n systemctl restart sahayak-asr 2>/dev/null; then
  echo "restarted via systemd"
else
  if [ -n "$OLD" ] && [ "$OLD" != 0 ]; then
    kill -9 "$OLD" || true
  fi
  sleep 3
fi
NEW=$(systemctl show -p MainPID --value sahayak-asr 2>/dev/null || echo 0)
echo "new_pid=$NEW"
systemctl is-active sahayak-asr || true
echo "waiting for asr backend= (whisper load can take ~1 min)..."
for i in $(seq 1 45); do
  if journalctl -u sahayak-asr -n 40 --no-pager | grep -q "asr backend="; then
    break
  fi
  sleep 2
done
journalctl -u sahayak-asr -n 25 --no-pager | tail -25
