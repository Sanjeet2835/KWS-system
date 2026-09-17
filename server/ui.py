"""480×320 Anuvani panel. On-screen ‹ › and tabs change pages; swipe is extra."""

from __future__ import annotations

import fcntl
import json
import math
import mmap
import os
import select
import socket
import struct
import time
from collections import Counter
from pathlib import Path

from server.state import Board
from shared.config import (
    ENERGY_RMS_THRESHOLD,
    KEYWORD,
    MODELS_DIR,
    PRODUCT_NAME,
    ROOT,
    WAKE_SCORE_THRESHOLD,
)

# Night-radio booth: indigo ink, marigold live, paper type. Not a lab printout.
BG = (16, 20, 28)
PANEL = (30, 38, 50)
LINE = (46, 56, 72)
INK = (234, 238, 244)
MUTED = (130, 142, 156)
ACCENT = (232, 154, 48)
GOOD = (56, 186, 130)
WARN = (214, 118, 64)
PAGES = ("LIVE", "QUOTA", "PATH", "ABOUT")

_EVENT = struct.Struct("llHHi")
EV_KEY, EV_ABS, EV_SYN = 0x01, 0x03, 0x00
ABS_X, ABS_Y, ABS_PRESSURE = 0x00, 0x01, 0x18
ABS_MT_POSITION_X, ABS_MT_POSITION_Y = 0x35, 0x36
BTN_TOUCH, BTN_LEFT = 0x14A, 0x110
_LIFT_IDLE_S = 0.25

QUOTA_RAM_KB = 256.0
QUOTA_IDLE_CPU = 10.0
QUOTA_ARENA_KB = 64.0

_TOUCH_NAME_HINTS = ("ads7846", "xpt2046", "stmpe", "tsc2046", "goodix", "ft5", "touch")
_TOUCH_SKIP = ("vc4", "hdmi", "cec", "power", "pwr", "sleep", "lid", "headset")


def _ioctl_ior(kind: str, nr: int, size: int) -> int:
    return (2 << 30) | (size << 16) | (ord(kind) << 8) | nr


def _ioctl_iow(kind: str, nr: int, size: int) -> int:
    return (1 << 30) | (size << 16) | (ord(kind) << 8) | nr


EVIOCGRAB = _ioctl_iow("E", 0x90, 4)


def _parse_input_devices(text: str) -> list[tuple[str, str, int]]:
    found: list[tuple[str, str, int]] = []
    name = ""
    abs_bits = 0
    handlers = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            event = _handler_event(handlers)
            if event:
                found.append((event, name, abs_bits))
            name, abs_bits, handlers = "", 0, ""
            continue
        if line.startswith("N: Name="):
            name = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("H: Handlers="):
            handlers = line.split("=", 1)[1]
        elif line.startswith("B: ABS="):
            abs_bits = 0
            for part in line.split("=", 1)[1].strip().split():
                try:
                    abs_bits |= int(part, 16)
                except ValueError:
                    pass
    event = _handler_event(handlers)
    if event:
        found.append((event, name, abs_bits))
    return found


def _handler_event(handlers: str) -> str:
    for token in handlers.split():
        if token.startswith("event") and token[5:].isdigit():
            return f"/dev/input/{token}"
    return ""


def find_touch_path() -> tuple[str | None, str]:
    env = os.environ.get("SAHAYAK_TOUCH", "").strip()
    if env:
        return env, env
    proc = Path("/proc/bus/input/devices")
    text = proc.read_text(errors="replace") if proc.exists() else ""
    devices = _parse_input_devices(text)
    ranked: list[tuple[int, str, str]] = []
    for path, name, abs_bits in devices:
        low = name.lower()
        if any(skip in low for skip in _TOUCH_SKIP):
            continue
        has_xy = bool(abs_bits & 0x3) or any(h in low for h in _TOUCH_NAME_HINTS)
        if not has_xy:
            continue
        score = 10 if any(h in low for h in _TOUCH_NAME_HINTS) else 1
        ranked.append((score, path, name))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    if ranked:
        return ranked[0][1], ranked[0][2]
    for ev in sorted(Path("/dev/input").glob("event*")):
        return str(ev), ev.name
    return None, ""


def _norm(value: int, lo: int, hi: int) -> float:
    return min(1.0, max(0.0, (value - lo) / max(1, hi - lo)))


def _mappings(nx: float, ny: float) -> list[tuple[float, float]]:
    return [
        (nx, ny),
        (1.0 - nx, ny),
        (nx, 1.0 - ny),
        (1.0 - nx, 1.0 - ny),
        (ny, nx),
        (1.0 - ny, nx),
        (ny, 1.0 - nx),
        (1.0 - ny, 1.0 - nx),
    ]


class TouchSwipe:
    def __init__(self) -> None:
        self.fd = None
        self.path = ""
        self.name = ""
        self.rest = b""
        self.x = self.y = 0
        self.xmin = self.ymin = 0
        self.xmax = self.ymax = 4095
        self.down = False
        self.origin_x: int | None = None
        self.origin_y: int | None = None
        self.fired = False
        self.last_event_at = 0.0
        self._open()

    def _open(self) -> None:
        path, name = find_touch_path()
        if not path:
            print("touch: no evdev node", flush=True)
            return
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            print(f"touch: open {path} failed ({exc})", flush=True)
            self.fd = None
            return
        self.path = path
        self.name = name
        try:
            fcntl.ioctl(self.fd, EVIOCGRAB, 1)
        except OSError as exc:
            print(f"touch: grab skipped ({exc})", flush=True)
        self._load_abs_range()
        print(
            f"touch: {path} ({name or 'unknown'}) event={_EVENT.size}B "
            f"x={self.xmin}:{self.xmax} y={self.ymin}:{self.ymax}",
            flush=True,
        )

    def _load_abs_range(self) -> None:
        if self.fd is None:
            return
        buf = bytearray(24)
        for axis, lo_attr, hi_attr in (
            (ABS_X, "xmin", "xmax"),
            (ABS_Y, "ymin", "ymax"),
        ):
            try:
                fcntl.ioctl(self.fd, _ioctl_ior("E", 0x40 + axis, 24), buf)
            except OSError:
                continue
            _value, minimum, maximum, _fuzz, _flat, _res = struct.unpack("iiiiii", buf)
            if maximum > minimum:
                setattr(self, lo_attr, minimum)
                setattr(self, hi_attr, maximum)

    def _thresh(self) -> tuple[int, int]:
        tx = max(80, int(0.12 * (self.xmax - self.xmin)))
        ty = max(80, int(0.12 * (self.ymax - self.ymin)))
        return tx, ty

    def _begin(self) -> None:
        if self.down:
            return
        self.down = True
        self.fired = False
        self.origin_x = None
        self.origin_y = None

    def _reset(self) -> None:
        self.down = False
        self.fired = False
        self.origin_x = None
        self.origin_y = None

    def _swipe_step(self) -> int:
        dx = 0 if self.origin_x is None else self.x - self.origin_x
        dy = 0 if self.origin_y is None else self.y - self.origin_y
        tx, ty = self._thresh()
        if abs(dx) < tx and abs(dy) < ty:
            return 0
        if abs(dx) >= abs(dy):
            return 1 if dx < 0 else -1
        return 1 if dy < 0 else -1

    def _hit_nav(self, layout: dict, width: int, height: int) -> int:
        nx = _norm(self.x, self.xmin, self.xmax)
        ny = _norm(self.y, self.ymin, self.ymax)
        votes: list[int] = []
        for fx, fy in _mappings(nx, ny):
            sx, sy = fx * (width - 1), fy * (height - 1)
            for rect in layout["prev"]:
                if rect.collidepoint(sx, sy):
                    votes.append(-1)
            for rect in layout["next"]:
                if rect.collidepoint(sx, sy):
                    votes.append(1)
            for i, rect in enumerate(layout["tabs"]):
                if rect.collidepoint(sx, sy):
                    votes.append(10 + i)
        if not votes:
            return 0
        choice, n = Counter(votes).most_common(1)[0]
        if n < 2 and len(set(votes)) > 1:
            return 0
        return choice

    def _lift(self, layout: dict, width: int, height: int) -> int:
        step = 0
        if not self.fired:
            step = self._swipe_step() or self._hit_nav(layout, width, height)
        self._reset()
        return step

    def poll(self, layout: dict, width: int, height: int) -> int:
        if self.fd is None:
            return 0
        delta = 0
        while True:
            ready, _, _ = select.select([self.fd], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(self.fd, _EVENT.size * 32)
            except BlockingIOError:
                break
            if not chunk:
                break
            self.rest += chunk
            while len(self.rest) >= _EVENT.size:
                raw = self.rest[: _EVENT.size]
                self.rest = self.rest[_EVENT.size :]
                _sec, _usec, etype, code, value = _EVENT.unpack(raw)
                self.last_event_at = time.monotonic()
                if etype == EV_ABS:
                    if code in (ABS_X, ABS_MT_POSITION_X):
                        self.x = value
                        if not self.down:
                            self._begin()
                        if self.origin_x is None:
                            self.origin_x = value
                    elif code in (ABS_Y, ABS_MT_POSITION_Y):
                        self.y = value
                        if not self.down:
                            self._begin()
                        if self.origin_y is None:
                            self.origin_y = value
                    elif code == ABS_PRESSURE:
                        if value > 0 and not self.down:
                            self._begin()
                        elif value == 0 and self.down:
                            delta = self._lift(layout, width, height) or delta
                elif etype == EV_KEY and code in (BTN_TOUCH, BTN_LEFT):
                    if value:
                        self._begin()
                    elif self.down:
                        delta = self._lift(layout, width, height) or delta
                elif etype == EV_SYN and self.down:
                    if self.origin_x is None:
                        self.origin_x = self.x
                    if self.origin_y is None:
                        self.origin_y = self.y
                    if not self.fired:
                        step = self._swipe_step()
                        if step:
                            self.fired = True
                            delta = step
        if self.down and self.last_event_at and (time.monotonic() - self.last_event_at) > _LIFT_IDLE_S:
            extra = self._lift(layout, width, height)
            delta = extra or delta
        return delta


def _open_fbdev(width: int, height: int):
    fb_path = Path(os.environ.get("SDL_FBDEV", "/dev/fb0"))
    if not fb_path.exists():
        return None
    fd = os.open(fb_path, os.O_RDWR)
    mapping = mmap.mmap(fd, width * height * 2, mmap.MAP_SHARED, mmap.PROT_WRITE)
    os.close(fd)
    return mapping


def _to_rgb565(surface) -> bytes:
    import pygame

    rgb = pygame.image.tostring(surface, "RGB")
    out = bytearray(len(rgb) // 3 * 2)
    j = 0
    for i in range(0, len(rgb), 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        packed = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[j] = packed & 0xFF
        out[j + 1] = packed >> 8
        j += 2
    return bytes(out)


def _lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.50.97.164", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "—"


_DEV_FONTS = (
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
_DEV_FONTS_BOLD = (
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _ui_font(size: int, bold: bool = False):
    import pygame

    for path in (_DEV_FONTS_BOLD if bold else _DEV_FONTS) + _DEV_FONTS:
        if os.path.isfile(path):
            try:
                return pygame.font.Font(path, size)
            except pygame.error:
                continue
    return pygame.font.SysFont(
        "Noto Sans Devanagari,Noto Sans,Lohit Devanagari,FreeSans,DejaVu Sans,sans",
        size,
        bold=bold,
    )


def _wrap(font, text, max_w, max_lines):
    words = (text or "—").split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if font.size(trial)[0] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines[:max_lines]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def load_scorecard() -> dict:
    keyword = KEYWORD
    metrics = _load_json(MODELS_DIR / f"{keyword}.metrics.json")
    tuning = _load_json(ROOT / "results" / "tuning.json")
    bench = _load_json(ROOT / "results" / "bench.json")
    far = _load_json(ROOT / "results" / "far.json")
    tpr = _load_json(ROOT / "results" / "tpr.json")
    latency = _load_json(ROOT / "results" / "latency.json")
    val = {}
    for split in metrics.get("splits") or []:
        if split.get("split") == "val":
            val = split
            break
    rec = tuning.get("recommended") or {}
    model_kb = 0.0
    tflite = MODELS_DIR / f"{keyword}.int8.tflite"
    if tflite.exists():
        model_kb = tflite.stat().st_size / 1024.0
    idle = (bench.get("idle_cpu_pct") or {}).get("mean")
    return {
        "keyword": keyword,
        "speakers": metrics.get("n_real_speakers"),
        "val_tpr": (val.get("at_default_threshold") or {}).get("tpr"),
        "tune_tpr": rec.get("tpr"),
        "tune_far": rec.get("far_per_hour"),
        "model_kb": model_kb,
        "idle_cpu": idle,
        "invoke_ms": ((bench.get("bench") or {}).get("inv_avg") or 0) / 1000.0,
        "arena_used": ((bench.get("tflm") or {}).get("arena_used") or 0) / 1024.0,
        "live_tpr": (tpr.get("tpr") if tpr else None),
        "live_far": (far.get("far_per_hour") if far else None),
        "lat_ms": (latency.get("median_ms") if latency else None),
        "ram_kb": 186.3,
    }


def _nav_layout(pygame, width: int, height: int) -> dict:
    prev_top = pygame.Rect(6, 4, 44, 34)
    next_top = pygame.Rect(width - 50, 4, 44, 34)
    prev_bot = pygame.Rect(6, height - 42, 44, 36)
    next_bot = pygame.Rect(width - 50, height - 42, 44, 36)
    tab_x = 56
    tab_w = (width - 112) // 4
    tabs = [pygame.Rect(tab_x + i * tab_w + 2, height - 40, tab_w - 4, 32) for i in range(4)]
    return {"prev": [prev_top, prev_bot], "next": [next_top, next_bot], "tabs": tabs}


def run_ui(board: Board, width: int, height: int) -> None:
    try:
        import pygame
    except ImportError:
        print("pygame not installed; UI disabled")
        while True:
            time.sleep(1)

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.display.init()
    try:
        pygame.display.set_mode((1, 1))
    except pygame.error:
        pass
    screen = pygame.Surface((width, height))
    fb = _open_fbdev(width, height)
    clock = pygame.time.Clock()
    brand = _ui_font(20, bold=True)
    title = _ui_font(22, bold=True)
    body = _ui_font(16)
    tiny = _ui_font(13)
    mono = pygame.font.SysFont("DejaVu Sans Mono,DejaVu Sans,monospace", 14)
    page = 0
    touch = TouchSwipe()
    ip = _lan_ip()
    last_sys = 0.0
    score = load_scorecard()
    layout = _nav_layout(pygame, width, height)

    while True:
        now = time.monotonic()
        nav = touch.poll(layout, width, height)
        if nav >= 10:
            page = nav - 10
        elif nav:
            page = (page + nav) % len(PAGES)
        if now - last_sys > 2.0:
            ip = _lan_ip()
            last_sys = now

        screen.fill(BG)
        _chrome(screen, pygame, brand, tiny, layout, page, width, height, touch.fd is not None)

        if page == 0:
            _page_live(screen, pygame, board, title, body, tiny, mono, width, now)
        elif page == 1:
            _page_quota(screen, pygame, board, title, body, tiny, mono, width, score)
        elif page == 2:
            _page_path(screen, pygame, board, title, body, tiny, width)
        else:
            _page_about(screen, pygame, board, title, body, tiny, width, ip, score)

        if fb is not None:
            fb.seek(0)
            fb.write(_to_rgb565(screen))
        clock.tick(12)


def _round(screen, pygame, rect, color, radius=8) -> None:
    pygame.draw.rect(screen, color, rect, border_radius=radius)


def _chrome(screen, pygame, brand, tiny, layout, page, width, height, touch_ok) -> None:
    pygame.draw.rect(screen, PANEL, (0, 0, width, 42))
    pygame.draw.rect(screen, PANEL, (0, height - 46, width, 46))
    screen.blit(brand.render(PRODUCT_NAME, True, INK), (58, 8))
    screen.blit(tiny.render(f"say  {KEYWORD}", True, MUTED), (172, 12))

    for rect, glyph in ((layout["prev"][0], "‹"), (layout["prev"][1], "‹"), (layout["next"][0], "›"), (layout["next"][1], "›")):
        _round(screen, pygame, rect, LINE, 6)
        g = brand.render(glyph, True, INK)
        screen.blit(g, (rect.centerx - g.get_width() // 2, rect.centery - g.get_height() // 2 - 1))

    for i, rect in enumerate(layout["tabs"]):
        on = i == page
        _round(screen, pygame, rect, ACCENT if on else LINE, 6)
        label = tiny.render(PAGES[i], True, BG if on else MUTED)
        screen.blit(label, (rect.centerx - label.get_width() // 2, rect.centery - label.get_height() // 2))

    if not touch_ok:
        screen.blit(tiny.render("touch off", True, WARN), (width - 118, 14))


def _bar(screen, pygame, x, y, w, h, frac, color) -> None:
    _round(screen, pygame, pygame.Rect(x, y, w, h), LINE, 4)
    fill = max(0, int(w * max(0.0, min(1.0, frac))))
    if fill:
        _round(screen, pygame, pygame.Rect(x, y, fill, h), color, 4)


def _mood(board: Board) -> str:
    if board.status == "AWAKE":
        return "think" if board.transcript_partial else "awake"
    return "sleep"


def _draw_ring(screen, pygame, cx: int, cy: int, mood: str, t: float) -> None:
    r = 34
    pygame.draw.circle(screen, PANEL, (cx, cy), r)
    pygame.draw.circle(screen, LINE, (cx, cy), r, 2)
    if mood == "sleep":
        pulse = r - 8 + int(3 * math.sin(t * 1.8))
        pygame.draw.circle(screen, ACCENT, (cx, cy), pulse, 2)
    elif mood == "think":
        pygame.draw.arc(screen, ACCENT, (cx - r, cy - r, 2 * r, 2 * r), t * 3.2, t * 3.2 + 1.8, 3)
    else:
        pygame.draw.circle(screen, ACCENT, (cx, cy), r, 3)
        pygame.draw.circle(screen, ACCENT, (cx, cy), 5)


def _page_live(screen, pygame, board, title, body, tiny, mono, width, now) -> None:
    mood = _mood(board)
    copy = {
        "sleep": ("Listening", f"Waiting for  {board.keyword or KEYWORD}"),
        "awake": ("Awake", "Keyword caught — streaming speech"),
        "think": ("Hearing you", "Pause — transcript in देवनागरी + English"),
    }[mood]
    screen.blit(title.render(copy[0], True, ACCENT if mood != "sleep" else INK), (16, 52))
    screen.blit(tiny.render(copy[1], True, MUTED), (16, 78))
    _draw_ring(screen, pygame, width - 52, 86, mood, now)

    fill = max(0.0, min(1.0, board.confidence))
    screen.blit(tiny.render("wake score", True, MUTED), (16, 102))
    _bar(screen, pygame, 16, 120, width - 120, 12, fill, GOOD if fill >= WAKE_SCORE_THRESHOLD else ACCENT)
    screen.blit(mono.render(f"{fill:.2f}", True, INK), (width - 92, 114))

    card = pygame.Rect(16, 148, width - 32, 112)
    _round(screen, pygame, card, PANEL, 10)
    heading = "Hearing…" if board.transcript_partial else "Last heard"
    screen.blit(tiny.render(heading, True, ACCENT if board.transcript_partial else MUTED), (28, 156))
    shown = board.transcript or "Nothing yet. Say the word, then speak."
    if board.transcript_partial and int(now * 2) % 2 == 0:
        shown = shown + " ▌"
    y = 176
    for line in _wrap(body, shown, width - 64, 3):
        screen.blit(body.render(line, True, INK), (28, y))
        y += 22
    wakes = f"{board.s3_wakes} wakes" if board.s3_wakes else "node quiet"
    screen.blit(tiny.render(wakes, True, MUTED), (28, 236))


def _fmt(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{100.0 * float(value):.0f}%"
    if kind == "cpu":
        return f"{float(value):.1f}%"
    if kind == "ms":
        return f"{float(value):.0f} ms"
    if kind == "kb":
        return f"{float(value):.0f} KB"
    if kind == "far":
        return f"{float(value):.1f}/h"
    return str(value)


def _page_quota(screen, pygame, board, title, body, tiny, mono, width, score) -> None:
    age = time.monotonic() - board.s3_seen if board.s3_seen else 1e9
    online = age < 3.0
    idle = board.s3_idle_cpu if (online and board.s3_idle_cpu) else (score.get("idle_cpu") or 0.0)
    ram_kb = score.get("ram_kb") or 186.3
    model_kb = score.get("model_kb") or 42.0
    tpr = score.get("live_tpr") if score.get("live_tpr") is not None else score.get("tune_tpr")
    far = score.get("live_far") if score.get("live_far") is not None else score.get("tune_far")
    lat = board.last_latency_ms if board.last_latency_ms is not None else score.get("lat_ms")
    streaming = online and board.s3_state in ("wake", "speech")

    screen.blit(title.render("What the PS scores", True, INK), (16, 52))
    if streaming:
        link = "Idle listen only — stream CPU is not scored"
    elif online:
        link = "ESP32-S3 live · idle listening"
    else:
        link = "Last bench on this Pi"
    screen.blit(tiny.render(link, True, GOOD if online else MUTED), (16, 78))

    cards = (
        ("Idle CPU", _fmt(idle, "cpu"), idle / QUOTA_IDLE_CPU if idle else 0, idle < QUOTA_IDLE_CPU, "limit 10%"),
        ("Listen RAM", _fmt(ram_kb, "kb"), ram_kb / QUOTA_RAM_KB, ram_kb < QUOTA_RAM_KB, "limit 256 KB"),
        ("INT8 model", _fmt(model_kb, "kb"), min(1.0, model_kb / 80.0), True, "on the node"),
    )
    x = 16
    for label, value, frac, ok, note in cards:
        box = pygame.Rect(x, 100, 144, 86)
        _round(screen, pygame, box, PANEL, 10)
        screen.blit(tiny.render(label, True, MUTED), (x + 10, 108))
        screen.blit(mono.render(value, True, INK), (x + 10, 128))
        _bar(screen, pygame, x + 10, 154, 124, 8, min(1.0, frac), GOOD if ok else WARN)
        screen.blit(tiny.render(note, True, MUTED), (x + 10, 166))
        x += 150

    pills = (
        ("Catch rate", _fmt(tpr, "pct")),
        ("False / h", _fmt(far, "far")),
        ("Word → Pi", _fmt(lat, "ms")),
    )
    x = 16
    for label, value in pills:
        box = pygame.Rect(x, 198, 144, 52)
        _round(screen, pygame, box, PANEL, 10)
        screen.blit(tiny.render(label, True, MUTED), (x + 10, 204))
        screen.blit(body.render(value, True, INK), (x + 10, 222))
        x += 150


def _page_path(screen, pygame, board, title, body, tiny, width) -> None:
    screen.blit(title.render("On the device, then the Pi", True, INK), (16, 52))
    screen.blit(tiny.render("Radio stays off until the word is local.", True, MUTED), (16, 78))

    age = time.monotonic() - board.s3_seen if board.s3_seen else 1e9
    online = age < 3.0
    node = board.s3_state if online else "offline"
    if board.status == "AWAKE":
        if board.transcript and not board.transcript_partial:
            active = 5
        elif board.transcript:
            active = 5
        else:
            active = 4
    elif node == "wake":
        active = 3
    elif node == "speech":
        active = 2
    elif node == "listen":
        active = 0 if board.s3_rms < ENERGY_RMS_THRESHOLD else 1
    else:
        active = -1

    steps = (
        ("Mic", "INMP441"),
        ("Gate", "quiet=off"),
        ("KWS", "on S3"),
        ("Wake", board.keyword or KEYWORD),
        ("Stream", "KWS1"),
        ("ASR", "Gemini"),
    )
    for i, (head, tail) in enumerate(steps):
        col, row = i % 3, i // 3
        x, y = 16 + col * 152, 104 + row * 78
        on = i == active
        box = pygame.Rect(x, y, 144, 66)
        _round(screen, pygame, box, ACCENT if on else PANEL, 10)
        ink = BG if on else INK
        dim = BG if on else MUTED
        screen.blit(body.render(head, True, ink), (x + 12, y + 10))
        screen.blit(tiny.render(tail, True, dim), (x + 12, y + 36))


def _page_about(screen, pygame, board, title, body, tiny, width, ip, score) -> None:
    screen.blit(title.render(PRODUCT_NAME, True, INK), (16, 52))
    screen.blit(tiny.render("PS 26172  ·  custom wake word  ·  hardware", True, MUTED), (16, 78))
    lines = (
        f"You say  {board.keyword or KEYWORD}.  The ESP32-S3 decides.",
        "Audio leaves the chip only after that wake.",
        "Gemini writes Hindi in देवनागरी; English stays Latin.",
        "After wake, this Pi is the remote ASR. Idle CPU is the only CPU score.",
        "No Alexa, no Hey Google, no commercial wake SDK.",
        f"Trained on {score.get('speakers') or '2,236'} real speakers.  TFLM + ESP-NN.",
        f"Pi  {ip}   {board.backend}",
    )
    y = 108
    for line in lines:
        for wrapped in _wrap(body, line, width - 32, 2):
            screen.blit(body.render(wrapped, True, INK), (16, y))
            y += 22
