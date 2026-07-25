#!/usr/bin/env bash
# Fetch a pinned, official Node runtime binary to bundle alongside Anamnesis's
# pruned node_modules (see build.sh + docs/specs/2026-07-25-anamnesis-
# vendoring-and-installer-plan.md Part 1b) -- so a packaged Pleiades install
# needs no system-installed Node at all. Mirrors pleiades/runtime.py's own
# "fetch an official prebuilt for this machine" pattern for llama-server.
#
# Only the `node` binary itself is extracted (not npm/corepack/docs/etc) --
# nothing on the end-user's machine ever runs `npm install`; the bundled
# node_modules is already pruned and complete from build.sh.
#
# Pinned version note: Node 22 (Maintenance LTS) and Node 24 (current Active
# LTS as of 2026-07) are both reasonable; picked 24 as the more forward-
# looking choice per the 2026-07-25 better-sqlite3->node:sqlite migration's
# own recommendation (see project memory) -- node:sqlite's stability keeps
# improving with newer Node, so there's no reason to pin backward here.
#
# Reproduce with:
#   ~/Pleiades/anamnesis/build-native/fetch-node.sh
#
# Output: ~/Pleiades/anamnesis/build-native/dist/node-runtime/<platform>/node
set -euo pipefail

NODE_VERSION="24.18.0"   # bump deliberately, not automatically -- re-verify
                          # the busy_timeout/enableForeignKeyConstraints
                          # behavior in test/history.test.js still passes
                          # against whatever version this becomes before
                          # bumping in a released build.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../anamnesis
BUILD_DIR="$REPO_ROOT/build-native"

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
if [ "$PLATFORM" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
  echo "error: this script has only been built + verified for Linux x86_64." >&2
  echo "       Got: $PLATFORM $ARCH." >&2
  exit 1
fi

TARGET="linux-x64"
ARCHIVE="node-v${NODE_VERSION}-${TARGET}.tar.xz"
URL="https://nodejs.org/dist/v${NODE_VERSION}/${ARCHIVE}"
SHASUMS_URL="https://nodejs.org/dist/v${NODE_VERSION}/SHASUMS256.txt"

OUT_DIR="$BUILD_DIR/dist/node-runtime/$TARGET"
WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

say() { printf "==> %s\n" "$*"; }

say "Fetching Node v${NODE_VERSION} ($TARGET)"
curl -fsSL "$URL" -o "$WORK_DIR/$ARCHIVE"
curl -fsSL "$SHASUMS_URL" -o "$WORK_DIR/SHASUMS256.txt"

say "Verifying checksum"
EXPECTED="$(grep " ${ARCHIVE}\$" "$WORK_DIR/SHASUMS256.txt" | cut -d' ' -f1)"
if [ -z "$EXPECTED" ]; then
  echo "error: no checksum entry found for $ARCHIVE in SHASUMS256.txt" >&2
  exit 1
fi
ACTUAL="$(sha256sum "$WORK_DIR/$ARCHIVE" | cut -d' ' -f1)"
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "error: checksum mismatch for $ARCHIVE" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 1
fi
say "Checksum OK ($ACTUAL)"

say "Extracting just the node binary (not npm/corepack/docs/etc)"
tar -xJf "$WORK_DIR/$ARCHIVE" -C "$WORK_DIR" "node-v${NODE_VERSION}-${TARGET}/bin/node"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
cp "$WORK_DIR/node-v${NODE_VERSION}-${TARGET}/bin/node" "$OUT_DIR/node"
chmod +x "$OUT_DIR/node"

say "Verifying the extracted binary actually runs"
ACTUAL_VERSION="$("$OUT_DIR/node" --version)"
if [ "$ACTUAL_VERSION" != "v${NODE_VERSION}" ]; then
  echo "error: extracted node reports '$ACTUAL_VERSION', expected 'v${NODE_VERSION}'" >&2
  exit 1
fi

say "OK -- $ACTUAL_VERSION at $OUT_DIR/node"
du -sh "$OUT_DIR/node"
