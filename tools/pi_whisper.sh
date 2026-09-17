#!/usr/bin/env bash
# Run ON the Pi (you are already logged in as pi). Installs Hugging Face
# faster-whisper base.en and restarts ASR. Vosk stays as fallback.
set -euo pipefail
cd /home/pi/sahayak
.venv/bin/pip install -q faster-whisper
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); print('whisper base (en+hi) ok')"
OLD=$(systemctl show -p MainPID --value sahayak-asr 2>/dev/null || echo 0)
echo "old_pid=$OLD"
if [ -n "$OLD" ] && [ "$OLD" != 0 ]; then
  kill -9 "$OLD" || true
fi
sleep 8
systemctl is-active sahayak-asr || true
journalctl -u sahayak-asr -n 15 --no-pager | tail -15
