#!/usr/bin/env bash
# Build the self-contained Pleiades backend bundle with PyInstaller.
#
# Reproduce with:
#   ~/Pleiades/desktop/backend-build/build.sh
#
# Requires PyInstaller in the dev venv (one-time):
#   ~/Pleiades/.venv/bin/python3.12 -m pip install pyinstaller
#
# Output: ~/Pleiades/desktop/backend-build/dist/pleiades-backend/
#   (a folder containing the `pleiades-backend` launcher + _internal/ deps --
#   onedir, not onefile, so it starts instantly and doesn't self-extract a
#   ~1GB+ dependency tree on every launch)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENV_PY="$REPO_ROOT/.venv/bin/python3.12"
if [ ! -x "$VENV_PY" ]; then
    echo "error: $VENV_PY not found. Build the venv first (see repo CLAUDE.md)." >&2
    exit 1
fi

cd "$REPO_ROOT/desktop/backend-build"
"$VENV_PY" -m PyInstaller --noconfirm --clean backend.spec

echo
echo "Built: $REPO_ROOT/desktop/backend-build/dist/pleiades-backend/pleiades-backend"
du -sh "$REPO_ROOT/desktop/backend-build/dist/pleiades-backend"
