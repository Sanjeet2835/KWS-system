"""One place that knows how to talk to the S3 over USB serial.

The measurement tools all need the same three things: open the port without
resetting the board, send a command, and parse the node's log lines. Keeping that
here means a firmware log-format change is a one-file fix.

On the S3's native USB there is no UART bridge: DTR drives GPIO0 and RTS drives
reset, in the USB-Serial-JTAG peripheral itself. macOS asserts DTR when a port is
opened, which holds GPIO0 low, so a board reset while a host is attached strands
the chip in ROM download mode instead of running the app. `reset_into_app()`
handles that; plain `S3Link()` never resets, so a measurement run is safe.
"""

from __future__ import annotations

import glob
import os
import re
import time
from dataclasses import dataclass, field

DEFAULT_BAUD = 921600

# Native USB-JTAG enumerates as cu.usbmodem*; UART-USB (CH340/CP2102) as
# cu.usbserial* / SLAB. macOS reassigns the suffix when you replug, so never
# hardcode usbmodem14201.
_PORT_GLOBS = (
    "/dev/cu.usbmodem*",
    "/dev/cu.usbserial*",
    "/dev/cu.SLAB_USBtoUART*",
    "/dev/cu.wchusbserial*",
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
)


def list_s3_ports() -> list[str]:
    found: list[str] = []
    for pattern in _PORT_GLOBS:
        found.extend(sorted(glob.glob(pattern)))
    # Prefer native USB-JTAG when both ports of a DevKit are plugged in.
    modem = [p for p in found if "usbmodem" in p]
    other = [p for p in found if p not in modem]
    return modem + other


def default_port() -> str:
    env = os.environ.get("SAHAYAK_S3_PORT")
    if env:
        if not os.path.exists(env):
            raise SystemExit(f"SAHAYAK_S3_PORT={env} does not exist")
        return env
    ports = list_s3_ports()
    if ports:
        return ports[0]
    raise SystemExit(
        "No ESP32 serial port. Plug the S3 USB-C in (wait 2s) and retry. "
        "Looked for /dev/cu.usbmodem* and /dev/cu.usbserial*."
    )


# Back-compat for `from tools.s3_link import DEFAULT_PORT`. Prefer default_port().
DEFAULT_PORT = "/dev/cu.usbmodem14201"

WAKE_RE = re.compile(
    r"WAKE t=(?P<t>\d+) score=(?P<score>[\d.]+) pkw=(?P<pkw>[\d.]+) "
    r"pun=(?P<pun>[\d.]+) psil=(?P<psil>[\d.]+) mfcc_us=(?P<mfcc_us>\d+) inv_us=(?P<inv_us>\d+)"
)
TEL_RE = re.compile(r"TEL1 (?P<body>.+)")
MIC_RE = re.compile(r"MIC rms=(?P<rms>[\d.eE+-]+) peak=(?P<peak>[\d.eE+-]+) dc=(?P<dc>[\d.eE+-]+)")
STATUS_RE = re.compile(r"STATUS (?P<body>.+)")


@dataclass
class Wake:
    host_time: float
    node_ms: int
    score: float
    p_keyword: float
    p_unknown: float
    p_silence: float
    mfcc_us: int
    invoke_us: int


@dataclass
class Event:
    kind: str  # "wake" | "tel" | "mic" | "status" | "log"
    raw: str
    host_time: float
    wake: Wake | None = None
    fields: dict = field(default_factory=dict)


def _kv(body: str) -> dict:
    out: dict = {}
    for token in body.split():
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        try:
            out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            out[k] = v
    return out


def reset_into_app(port: str | None = None, baud: int = DEFAULT_BAUD) -> None:
    """Reboot the node into its application and wait for USB to come back.

    The DTR True->False transition is deliberate: pyserial only emits a control
    request when the line changes, and macOS has already asserted DTR, so setting
    it False on a freshly opened port is a no-op and GPIO0 stays low. Releasing it
    explicitly before dropping RESET is what makes the chip boot from flash rather
    than sit in "waiting for download".
    """
    import serial

    port = port or default_port()
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.5
    ser.open()
    try:
        ser.dtr = True
        time.sleep(0.1)
        ser.rts = True  # hold in reset
        time.sleep(0.1)
        ser.dtr = False  # release GPIO0 while still in reset
        time.sleep(0.1)
        ser.rts = False  # boot
    finally:
        ser.close()

    # The reset re-enumerates USB-Serial-JTAG, so the old device node goes stale.
    for _ in range(40):
        time.sleep(0.1)
        if not os.path.exists(port):
            break
    for _ in range(150):
        time.sleep(0.1)
        if os.path.exists(port):
            break
    time.sleep(1.2)


class S3Link:
    def __init__(
        self,
        port: str | None = None,
        baud: int = DEFAULT_BAUD,
        timeout: float = 1.0,
        reset: bool = False,
        open_attempts: int = 10,
    ) -> None:
        import serial

        port = port or default_port()
        self.port = port
        if reset:
            reset_into_app(port, baud)

        last: Exception | None = None
        for _ in range(open_attempts):
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = baud
            self.ser.timeout = timeout
            # Never toggle DTR/RTS here: on USB-JTAG that would reset the S3 and
            # wipe the run in progress.
            self.ser.dtr = False
            self.ser.rts = False
            try:
                self.ser.open()
                self.ser.read(0)
                break
            except Exception as exc:  # port still settling after a re-enumeration
                last = exc
                try:
                    self.ser.close()
                except Exception:
                    pass
                time.sleep(1.0)
        else:
            raise SystemExit(f"could not open {port}: {last}")

        time.sleep(0.3)
        self.ser.reset_input_buffer()

    def send(self, cmd: str) -> None:
        self.ser.write(f"{cmd}\n".encode())
        self.ser.flush()

    def status(self, timeout: float = 3.0) -> dict:
        self.send("STATUS")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for ev in self.read(0.5):
                if ev.kind == "status":
                    return ev.fields
        return {}

    def read(self, seconds: float):
        """Yield events for up to `seconds`. Returns when the window closes."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            raw = self.ser.readline().decode("utf-8", "replace").strip()
            if not raw:
                continue
            now = time.monotonic()
            m = WAKE_RE.search(raw)
            if m:
                g = m.groupdict()
                yield Event(
                    "wake",
                    raw,
                    now,
                    wake=Wake(
                        host_time=now,
                        node_ms=int(g["t"]),
                        score=float(g["score"]),
                        p_keyword=float(g["pkw"]),
                        p_unknown=float(g["pun"]),
                        p_silence=float(g["psil"]),
                        mfcc_us=int(g["mfcc_us"]),
                        invoke_us=int(g["inv_us"]),
                    ),
                )
                continue
            m = TEL_RE.search(raw)
            if m:
                yield Event("tel", raw, now, fields=_kv(m.group("body")))
                continue
            m = MIC_RE.search(raw)
            if m:
                yield Event("mic", raw, now, fields={k: float(v) for k, v in m.groupdict().items()})
                continue
            m = STATUS_RE.search(raw)
            if m:
                yield Event("status", raw, now, fields=_kv(m.group("body")))
                continue
            yield Event("log", raw, now)

    def close(self) -> None:
        self.ser.close()

    def __enter__(self) -> S3Link:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
