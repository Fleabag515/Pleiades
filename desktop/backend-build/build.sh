#!/usr/bin/env bash
# Build the self-contained Pleiades backend bundle with PyInstaller.
#
# Reproduce with:
#   ~/Pleiades/desktop/backend-build/build.sh
#
# Uses whichever python built ~/Pleiades/.venv (PLEIADES_BACKEND_PY overrides
# this -- install.sh's install_desktop_app() passes its own already-resolved
# interpreter so this script doesn't have to re-derive it). Requires
# PyInstaller there (one-time): <that python> -m pip install pyinstaller
#
# Output: ~/Pleiades/desktop/backend-build/dist/pleiades-backend/
#   (a folder containing the `pleiades-backend` launcher + _internal/ deps --
#   onedir, not onefile, so it starts instantly and doesn't self-extract a
#   ~1GB+ dependency tree on every launch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PY="${PLEIADES_BACKEND_PY:-}"
if [ -z "$VENV_PY" ]; then
    # No override given (standalone run): the venv's own bin/ only ever has
    # the specific minor version it was created with, never python3.12
    # unless that's literally what created it -- fall back through what's
    # actually likely to exist rather than assuming 3.12.
    for c in python3.12 python3.11 python3.13 python3.10 python3; do
        [ -x "$REPO_ROOT/.venv/bin/$c" ] && { VENV_PY="$REPO_ROOT/.venv/bin/$c"; break; }
    done
fi
if [ -z "$VENV_PY" ] || [ ! -x "$VENV_PY" ]; then
    echo "error: no usable python found under $REPO_ROOT/.venv/bin/. Build the venv first (see repo CLAUDE.md)." >&2
    exit 1
fi

cd "$REPO_ROOT/desktop/backend-build"
"$VENV_PY" -m PyInstaller --noconfirm --clean backend.spec

echo
echo "Built: $REPO_ROOT/desktop/backend-build/dist/pleiades-backend/pleiades-backend"
du -sh "$REPO_ROOT/desktop/backend-build/dist/pleiades-backend"
