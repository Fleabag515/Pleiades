"""PyInstaller entrypoint for the bundled Pleiades backend.

Equivalent to ``pleiades ui --host <h> --port <p> --no-browser`` but calls
straight into ``pleiades.webui.run()`` so we skip Click's CLI machinery
(which PyInstaller has to freeze too, for no benefit in a single-purpose
bundle that always launches the web UI headless).

Built via ``pyinstaller backend.spec`` from this directory. See
``desktop/backend-build/build.sh`` for the full reproducible build command.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    # The Electron main process spawns this exe with piped stdio (see
    # spawnBackend() in desktop/src/main/index.ts), and on Windows a piped
    # (non-console) stdout/stderr defaults to the legacy ANSI codepage
    # (cp1252), not UTF-8 -- confirmed live: a plain "->"-free run crashed
    # here with UnicodeEncodeError on an em-dash in a startup print(), which
    # would have taken the whole desktop app down on every single launch.
    # Force UTF-8 with lossy fallback so cosmetic console/log text can never
    # crash this process again, regardless of what gets printed later.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(prog="pleiades-backend")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    from pleiades.webui import run

    run(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
