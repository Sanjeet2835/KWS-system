"""Socket listener + streaming ASR. Vosk if present, otherwise a local mock.

  python -m server.asr_server
  python -m server.asr_server --no-ui
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import socket
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.state import Board  # noqa: E402
from shared.config import (  # noqa: E402
    ASR_HOST,
    ASR_PORT,
    KEYWORD,
    MODELS_DIR,
    ROOT,
    TELEMETRY_PORT,
    UI_HEIGHT,
    UI_WIDTH,
)
from shared.feature_spec import SPEC  # noqa: E402
from shared.protocol import FRAME_LEN_STRUCT, HEADER_BYTES, unpack_header  # noqa: E402

BOARD = Board(keyword=KEYWORD)
_GEMINI_PROMPT = (
    "Transcribe only this clip. Hindi and Hinglish words MUST be Devanagari "
    "(example: लाइट ऑन करो, पंखा बंद करो). Keep English words in Latin. "
    "Output the transcript only — no quotes, no translation, no extra sentence. "
    "Do not continue or repeat any previous utterance. "
    "If the clip is silence or only a wake word, output nothing."
)


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


def find_vosk_model() -> Path | None:
    if not MODELS_DIR.exists():
        return None
    models = [p for p in MODELS_DIR.glob("vosk-model*") if p.is_dir()]
    if not models:
        return None

    def rank(p: Path) -> tuple:
        n = p.name.lower()
        # lgraph (~128 MB) is the Pi-sized English model: better than *-small-*,
        # far lighter than the 1.8 GB vosk-model-en-us-0.22.
        if "lgraph" in n:
            return (0, n)
        if "small" not in n:
            return (1, n)
        return (2, n)

    return sorted(models, key=rank)[0]


def _pcm16_to_float(pcm: bytes):
    import numpy as np

    if not pcm:
        return np.zeros(0, np.float32)
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if 0.012 < peak < 0.35:
        x = np.clip(x * (0.35 / peak), -1.0, 1.0)
    return x


def _pcm16_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _gemini_key() -> str:
    return os.environ.get("GEMINI_API_KEY", "").strip()


def _gemini_transcribe(pcm: bytes) -> str:
    key = _gemini_key()
    if not key or len(pcm) < SPEC.sample_rate:  # <0.5 s
        return ""
    model = os.environ.get("SAHAYAK_GEMINI_MODEL", "gemini-2.5-flash").strip()
    wav = _pcm16_wav(pcm, SPEC.sample_rate)
    body = json.dumps(
        {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": base64.b64encode(wav).decode("ascii"),
                            }
                        },
                        {"text": _GEMINI_PROMPT},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
        }
    ).encode("utf-8")
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={key}"
    )
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:180].decode("utf-8", "replace")
        print(f"gemini http {exc.code}: {detail}", flush=True)
        return ""
    except Exception as exc:
        print(f"gemini failed: {exc}", flush=True)
        return ""
    try:
        parts = payload["candidates"][0]["content"]["parts"]
        text = " ".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        return ""
    if text.startswith("```"):
        text = text.strip("`").strip()
    return text


def _whisper_decode(model, pcm: bytes) -> str:
    audio = _pcm16_to_float(pcm)
    if len(audio) < SPEC.sample_rate // 8:
        return ""
    lang = os.environ.get("SAHAYAK_WHISPER_LANG", "").strip() or None
    segs, _ = model.transcribe(
        audio,
        language=lang,
        beam_size=1,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False,
        without_timestamps=True,
    )
    return " ".join(s.text.strip() for s in segs if s.text).strip()


class Transcriber:
    """Whisper on a worker thread so the panel updates while audio is still arriving."""

    def __init__(self) -> None:
        _load_dotenv()
        self._whisper = None
        self._recognizer = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._busy = False
        self._partial = ""
        self._gen = 0
        self._gemini_on = bool(_gemini_key())
        try:
            from faster_whisper import WhisperModel

            name = os.environ.get("SAHAYAK_WHISPER", "distil-small.en")
            print(
                f"asr loading whisper={name} gemini={'yes' if self._gemini_on else 'no'}",
                flush=True,
            )
            self._whisper = WhisperModel(name, device="cpu", compute_type="int8")
            BOARD.backend = f"whisper:{name}"
            if self._gemini_on:
                BOARD.backend += "+gemini"
            print(f"asr backend={BOARD.backend}", flush=True)
        except Exception as exc:
            print(f"faster-whisper unavailable ({exc}); trying vosk")
        if self._whisper is None:
            self.model_path = find_vosk_model()
            if self.model_path is not None:
                try:
                    from vosk import KaldiRecognizer, Model

                    self._recognizer = KaldiRecognizer(Model(str(self.model_path)), SPEC.sample_rate)
                    self._recognizer.SetWords(True)
                    BOARD.backend = f"vosk:{self.model_path.name}"
                except Exception as exc:
                    print(f"vosk load failed ({exc}); using mock")
        if self._whisper is None and self._recognizer is None:
            BOARD.backend = "mock"

    def reset(self) -> None:
        with self._lock:
            self._gen += 1
            self._buf.clear()
            self._partial = ""
        if self._recognizer is not None:
            self._recognizer.Reset()

    def _kick_whisper(self) -> None:
        # Gemini owns the final transcript. Live Whisper (English-only) was
        # painting leftover + prompt-biased words onto the panel mid-stream.
        if self._gemini_on or self._whisper is None or self._busy:
            return
        with self._lock:
            snap = bytes(self._buf)
            gen = self._gen
        if len(snap) < SPEC.sample_rate:  # <0.5 s
            return
        self._busy = True

        def _run() -> None:
            try:
                text = _whisper_decode(self._whisper, snap)
                with self._lock:
                    stale = gen != self._gen
                if stale or not text:
                    return
                self._partial = text
                BOARD.transcript = text
                BOARD.transcript_partial = True
            except Exception as exc:
                print(f"whisper decode failed: {exc}", flush=True)
            finally:
                self._busy = False

        threading.Thread(target=_run, daemon=True).start()

    def feed(self, pcm: bytes) -> str:
        if self._whisper is not None:
            with self._lock:
                self._buf.extend(pcm)
            self._kick_whisper()
            return self._partial
        if self._recognizer is None:
            return ""
        self._buf.extend(pcm)
        chunk = max(2, (SPEC.sample_rate * 2 // 5) * 2)
        text = ""
        while len(self._buf) >= chunk:
            piece = bytes(self._buf[:chunk])
            del self._buf[:chunk]
            if self._recognizer.AcceptWaveform(piece):
                got = _vosk_text(self._recognizer.Result())
            else:
                got = _vosk_text(self._recognizer.PartialResult())
            if got:
                text = got
        return text

    def finish(self) -> str:
        if self._whisper is not None:
            with self._lock:
                snap = bytes(self._buf)
                gen = self._gen
            gemini_box: list[str] = [""]

            def _run_gemini() -> None:
                gemini_box[0] = _gemini_transcribe(snap)

            gthread = None
            if self._gemini_on:
                gthread = threading.Thread(target=_run_gemini, daemon=True)
                gthread.start()
            deadline = time.monotonic() + 8.0
            while self._busy and time.monotonic() < deadline:
                time.sleep(0.05)
            local = ""
            if not self._gemini_on:
                try:
                    local = _whisper_decode(self._whisper, snap) or self._partial
                except Exception as exc:
                    print(f"whisper final failed: {exc}", flush=True)
                    local = self._partial
            elif gthread is not None:
                gthread.join(12.0)
                with self._lock:
                    stale = gen != self._gen
                if stale:
                    return ""
                if gemini_box[0]:
                    print("asr final=gemini", flush=True)
                    return gemini_box[0]
                print("asr final=whisper-fallback", flush=True)
                try:
                    local = _whisper_decode(self._whisper, snap) or self._partial
                except Exception as exc:
                    print(f"whisper final failed: {exc}", flush=True)
                    local = self._partial
            return local
        if self._recognizer is None:
            return ""
        if self._buf:
            self._recognizer.AcceptWaveform(bytes(self._buf))
            self._buf.clear()
        return _vosk_text(self._recognizer.FinalResult())


def _vosk_text(raw: str) -> str:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    return (data.get("text") or data.get("partial") or "").strip()


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
    return bytes(buf)


TRANSCRIBE_LOCK = threading.Lock()


def handle_client(conn: socket.socket, addr, transcriber: Transcriber) -> None:
    t_accept = time.monotonic()
    try:
        header = _recv_exact(conn, HEADER_BYTES)
    except (ConnectionError, OSError):
        # ESP32 reachability probe: TCP connect, then close. Not a stream.
        conn.close()
        return
    preroll_ms, rate = unpack_header(header)
    t_first = None
    pcm_all = bytearray()
    with TRANSCRIBE_LOCK:
        BOARD.status = "AWAKE"
        BOARD.transcript = ""
        BOARD.transcript_partial = True
        BOARD.streams += 1
        BOARD.log(f"connect {addr[0]} preroll={preroll_ms}ms")
        transcriber.reset()
        partial = ""
        # Drop only the wake preroll the node already labelled. Extra tail skip
        # was eating the first command word and, on a false wake, previous speech.
        skip = max(0, int(preroll_ms * rate * 2 / 1000))

        try:
            while True:
                (nbytes,) = FRAME_LEN_STRUCT.unpack(_recv_exact(conn, 2))
                if nbytes == 0:
                    break
                payload = _recv_exact(conn, nbytes)
                if t_first is None:
                    t_first = time.monotonic()
                    BOARD.note_latency((t_first - t_accept) * 1000.0)
                pcm_all.extend(payload)
                BOARD.bytes_in += len(payload)
                if skip:
                    n = min(skip, len(payload))
                    payload = payload[n:]
                    skip -= n
                    if not payload:
                        continue
                text = transcriber.feed(payload)
                if text:
                    partial = text
                    BOARD.transcript = text
                    BOARD.transcript_partial = True
        except (ConnectionError, struct.error, ValueError, OSError) as exc:
            BOARD.log(f"stream error: {exc}")
        finally:
            final = transcriber.finish() or partial
            if transcriber._whisper is None and transcriber._recognizer is None:
                seconds = len(pcm_all) / (2 * rate)
                final = f"[mock] {seconds:.1f}s audio, {len(pcm_all)} bytes"
            BOARD.transcript = final
            BOARD.transcript_partial = False
            BOARD.status = "LISTENING"
            BOARD.log(f"done: {(BOARD.transcript or '')[:48]}")
            conn.close()


def serve_telemetry(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"TEL1 listening on 0.0.0.0:{port}", flush=True)
    while True:
        raw, addr = sock.recvfrom(256)
        line = raw.decode("ascii", "replace").strip()
        if not line.startswith("TEL1"):
            continue
        fields = {"src": addr[0]}
        for part in line.split():
            if "=" in part:
                k, v = part.split("=", 1)
                fields[k] = v
        try:
            BOARD.s3_state = fields.get("state", "listen")
            BOARD.s3_cpu = float(fields.get("cpu", "0"))
            # PS 26172: "<10% CPU utilization while idling in continuous
            # listening mode." Stream / speech CPU is not scored.
            if "idle" in fields:
                BOARD.s3_idle_cpu = float(fields["idle"])
            elif BOARD.s3_state == "listen":
                BOARD.s3_idle_cpu = BOARD.s3_cpu
            BOARD.s3_heap_kb = int(float(fields.get("heap", "0")))
            BOARD.s3_rssi = int(float(fields.get("rssi", "0")))
            BOARD.s3_rms = float(fields.get("rms", "0"))
            BOARD.s3_seen = time.monotonic()
            if "kw" in fields:
                BOARD.confidence = float(fields["kw"])
            BOARD.s3_mfcc_ms = float(fields.get("mfcc_us", "0")) / 1000.0
            BOARD.s3_invoke_ms = float(fields.get("inv_us", "0")) / 1000.0
            BOARD.s3_arena_kb = float(fields.get("arena", "0")) / 1024.0
            BOARD.s3_wakes = int(float(fields.get("wakes", "0")))
            if "lat_ms" in fields:
                BOARD.s3_lat_ms = float(fields["lat_ms"])
        except ValueError:
            continue


def serve(host: str, port: int, transcriber: Transcriber) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(4)
    print(f"ASR listening on {host}:{port} backend={BOARD.backend} keyword={KEYWORD}", flush=True)
    while True:
        conn, addr = sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr, transcriber), daemon=True).start()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=ASR_HOST)
    p.add_argument("--port", type=int, default=ASR_PORT)
    p.add_argument("--no-ui", action="store_true")
    return p.parse_args()


def main() -> None:
    # pygame's dummy/fbdev loop ignores SIGTERM; systemd then waits 90s.
    def _die(_signum, _frame) -> None:
        os._exit(0)

    signal.signal(signal.SIGTERM, _die)
    signal.signal(signal.SIGINT, _die)

    args = parse_args()
    transcriber = Transcriber()
    threading.Thread(target=serve, args=(args.host, args.port, transcriber), daemon=True).start()
    threading.Thread(target=serve_telemetry, args=(TELEMETRY_PORT,), daemon=True).start()
    if args.no_ui:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return
    from server.ui import run_ui

    run_ui(BOARD, UI_WIDTH, UI_HEIGHT)


if __name__ == "__main__":
    main()
