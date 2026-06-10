"""``python -m pleiades.webui`` — launch the control panel.

Usage::

    python -m pleiades.webui                 # auto-pick a port, open browser
    python -m pleiades.webui --port 8800     # fixed port
    python -m pleiades.webui --no-browser    # headless / remote

Works the same on Linux and Windows.
"""

from __future__ import annotations

import argparse

# Use the package-level run() (guarded), not server.run directly, so the
# loopback guard is scoped to the chosen bind host on this path too.
from . import run


def main() -> None:
    ap = argparse.ArgumentParser(prog="pleiades-ui",
                                 description="Pleiades local control panel")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None,
                    help="port (default: first free port from 8750)")
    ap.add_argument("--no-browser", action="store_true",
                    help="do not auto-open a browser window")
    args = ap.parse_args()
    run(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
