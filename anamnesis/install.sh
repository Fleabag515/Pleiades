#!/usr/bin/env bash
# install.sh — install the anamnesis CLI on Linux or macOS.
# Does NOT register a system service — run `anamnesis install` for that.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Fleabag515/anamnesis/main/install.sh | bash

set -euo pipefail

REPO="https://github.com/Fleabag515/anamnesis.git"
INSTALL_DIR="${ANAMNESIS_INSTALL_DIR:-$HOME/.local/share/anamnesis}"
BIN_DIR="${ANAMNESIS_BIN_DIR:-$HOME/.local/bin}"
MIN_NODE_MAJOR=18

# ─── Colour helpers ──────────────────────────────────────────────────────────
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*" >&2; }

# ─── Locate / install Node ───────────────────────────────────────────────────
find_node() {
  local nvm_root="${NVM_DIR:-$HOME/.nvm}"
  if [ -d "$nvm_root/versions/node" ]; then
    local pick
    pick="$(ls -1 "$nvm_root/versions/node" | grep -E '^v[0-9]*[02468]\.' | sort -V | tail -1 || true)"
    if [ -n "$pick" ] && [ -x "$nvm_root/versions/node/$pick/bin/node" ]; then
      echo "$nvm_root/versions/node/$pick/bin/node"; return
    fi
  fi
  command -v node 2>/dev/null || true
}

NODE_BIN="$(find_node)"
if [ -z "$NODE_BIN" ]; then
  yellow "Node.js not found — installing via nvm..."
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  export NVM_DIR="$HOME/.nvm"
  # shellcheck disable=SC1091
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
  nvm install --lts
  NODE_BIN="$(find_node)"
fi

node_major="$("$NODE_BIN" -e 'console.log(process.versions.node.split(".")[0])')"
if [ "$node_major" -lt "$MIN_NODE_MAJOR" ]; then
  red "Node $node_major found but $MIN_NODE_MAJOR+ required. Install Node $MIN_NODE_MAJOR+ and retry."
  exit 1
fi

NPM_BIN="$(dirname "$NODE_BIN")/npm"
green "Node $("$NODE_BIN" --version) at $NODE_BIN"

# ─── Clone or update repo ────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
  yellow "Updating existing install at $INSTALL_DIR..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  yellow "Installing to $INSTALL_DIR..."
  git clone --depth=1 "$REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
"$NPM_BIN" install --omit=dev --silent

# ─── Create wrapper script on PATH ──────────────────────────────────────────
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/anamnesis" << WRAPPER
#!/usr/bin/env bash
exec "$NODE_BIN" "$INSTALL_DIR/src/cli.js" "\$@"
WRAPPER
chmod +x "$BIN_DIR/anamnesis"

# ─── Ensure BIN_DIR is on PATH ──────────────────────────────────────────────
if ! echo ":$PATH:" | grep -q ":$BIN_DIR:"; then
  for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
    if [ -f "$rc" ]; then
      echo "export PATH=\"\$PATH:$BIN_DIR\"" >> "$rc"
      yellow "Added $BIN_DIR to PATH in $rc"
      break
    fi
  done
fi

echo ""
green "✓ anamnesis installed"
echo "  Run: anamnesis new       — create your first character"
echo "  Run: anamnesis install   — register as a system service (optional)"
echo ""
echo "  If 'anamnesis' is not found, open a new terminal or run:"
echo "    export PATH=\"\$PATH:$BIN_DIR\""
