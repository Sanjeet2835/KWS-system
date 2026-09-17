"""Serial monitor for the S3 that can actually reboot the board into its app.

`pio device monitor` opens the port with DTR asserted, which on the S3's native
USB holds GPIO0 low; any reset from that state lands in ROM download mode and the
firmware never runs. This uses tools.s3_link, which releases GPIO0 before
dropping RESET.

  python -m tools.s3_monitor --reset          # reboot and watch the boot log
  python -m tools.s3_monitor                  # attach without disturbing the node
  python -m tools.s3_monitor --reset --send STATUS
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.s3_link import DEFAULT_BAUD, S3Link  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None, help="serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--reset", action="store_true", help="reboot into the app first")
    p.add_argument("--send", action="append", default=[], help="command to send after attaching")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with S3Link(args.port, args.baud, reset=args.reset) as link:
        for cmd in args.send:
            link.send(cmd)
            time.sleep(0.1)
        n = 0
        for ev in link.read(args.seconds):
            n += 1
            print(ev.raw, flush=True)
    print(f"--- {n} lines in {args.seconds:.0f}s")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
