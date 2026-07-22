#!/usr/bin/env bash
# Pleiades one-line installer  (Linux / macOS)
#
#   curl -fsSL https://raw.githubusercontent.com/Fleabag515/Pleiades/main/install.sh | bash
#
# Pass options after the pipe, e.g.:
#   curl -fsSL .../install.sh | bash -s -- --gpu --dir ~/apps/Pleiades
#
# Options:
#   --dir DIR             install location     (default: ~/Pleiades, or $PLEIADES_DIR)
#   --branch NAME         git branch           (default: main, or $PLEIADES_BRANCH)
#   --cpu                 force CPU build of llama-cpp-python
#   --gpu                 force GPU build of llama-cpp-python
#   --core                core only (skip browser, SearXNG, Discord, native runtime)
#   --no-browser          skip Camoufox
#   --no-searxng          skip SearXNG (Docker)
#   --no-discord          skip the Discord extra
#   --no-native-runtime   skip 'pleiades runtime install' (native llama-server)
#   -h, --help            show this help
set -euo pipefail

REPO_URL="${PLEIADES_REPO:-https://github.com/Fleabag515/Pleiades.git}"
INSTALL_DIR="${PLEIADES_DIR:-$HOME/Pleiades}"
BRANCH="${PLEIADES_BRANCH:-main}"
GPU_MODE="auto"          # auto | cpu | gpu
WITH_BROWSER=1
WITH_SEARXNG=1
WITH_DISCORD=1
WITH_NATIVE_RUNTIME=1

# ---- pretty output ---------------------------------------------------------
if [ -t 1 ]; then B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'; else B=""; G=""; Y=""; R=""; N=""; fi
say()  { printf "%s==>%s %s%s%s\n" "$B$G" "$N" "$B" "$*" "$N"; }
info() { printf "    %s\n" "$*"; }
warn() { printf "%s!!  %s%s\n" "$Y" "$*" "$N"; }
err()  { printf "%sxx  %s%s\n" "$R" "$*" "$N" >&2; }
die()  { err "$*"; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---- args ------------------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) INSTALL_DIR="${2:?}"; shift 2;;
    --branch) BRANCH="${2:?}"; shift 2;;
    --cpu) GPU_MODE="cpu"; shift;;
    --gpu) GPU_MODE="gpu"; shift;;
    --core) WITH_BROWSER=0; WITH_SEARXNG=0; WITH_DISCORD=0; WITH_NATIVE_RUNTIME=0; shift;;
    --no-browser) WITH_BROWSER=0; shift;;
    --no-searxng) WITH_SEARXNG=0; shift;;
    --no-discord) WITH_DISCORD=0; shift;;
    --no-native-runtime) WITH_NATIVE_RUNTIME=0; shift;;
    --no-desktop-app) WITH_DESKTOP_APP=0; shift;;
    -h|--help) sed -n '2,30p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//'; exit 0;;
    *) warn "ignoring unknown option: $1"; shift;;
  esac
done

# ---- platform + package manager -------------------------------------------
OS="$(uname -s)"
case "$OS" in
  Linux)  PLATFORM="linux";;
  Darwin) PLATFORM="macos";;
  *) die "Unsupported OS '$OS'. Use install.ps1 on Windows.";;
esac

SUDO=""
[ "$(id -u)" -ne 0 ] && have sudo && SUDO="sudo"

PM=""
if [ "$PLATFORM" = "macos" ]; then
  have brew && PM="brew"
else
  for p in apt-get dnf pacman zypper; do have "$p" && { PM="$p"; break; }; done
fi

pm_install() {
  case "$PM" in
    apt-get) $SUDO apt-get update -y >/dev/null && $SUDO apt-get install -y "$@";;
    dnf)     $SUDO dnf install -y "$@";;
    pacman)  $SUDO pacman -Sy --noconfirm "$@";;
    zypper)  $SUDO zypper --non-interactive install "$@";;
    brew)    brew install "$@";;
    *) return 1;;
  esac
}

# ---- prerequisites (hybrid: auto-install runtimes, guide for Docker) -------
ensure_base() {
  have git  || { say "Installing git";  pm_install git  || die "Please install git and re-run."; }
  have curl || pm_install curl || true
}

PYBIN=""
pyok() { "$1" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)' >/dev/null 2>&1; }
ensure_python() {
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    have "$c" && pyok "$c" && { PYBIN="$c"; break; }
  done
  if [ -z "$PYBIN" ]; then
    say "Installing Python 3.10+ and build tools"
    case "$PM" in
      apt-get) pm_install python3 python3-venv python3-pip python3-dev build-essential cmake;;
      dnf)     pm_install python3 python3-pip python3-devel gcc gcc-c++ make cmake;;
      pacman)  pm_install python python-pip base-devel cmake;;
      zypper)  pm_install python3 python3-pip python3-devel gcc-c++ make cmake;;
      brew)    pm_install python cmake;;
      *) die "Could not auto-install Python. Install Python >=3.10 and re-run.";;
    esac
    for c in python3.12 python3.11 python3 python; do have "$c" && pyok "$c" && { PYBIN="$c"; break; }; done
  fi
  [ -n "$PYBIN" ] || die "Python >=3.10 unavailable after install attempt."
  # make sure venv + a C/C++ toolchain exist for the llama-cpp-python build
  case "$PM" in
    apt-get) pm_install python3-venv build-essential cmake >/dev/null 2>&1 || true;;
    dnf)     pm_install gcc gcc-c++ make cmake          >/dev/null 2>&1 || true;;
    pacman)  pm_install base-devel cmake                >/dev/null 2>&1 || true;;
    zypper)  pm_install gcc-c++ make cmake              >/dev/null 2>&1 || true;;
    brew)    have cmake || pm_install cmake || true;;
  esac
  info "Using $("$PYBIN" --version 2>&1)"
}

ensure_node() {
  have npm && return 0
  say "Installing Node.js + npm (for Anamnesis + the desktop app build)"
  case "$PM" in
    apt-get|dnf|zypper) pm_install nodejs npm;;
    pacman)             pm_install nodejs npm;;
    brew)               pm_install node;;
    *) warn "Could not auto-install Node. Install Node.js + npm, then 'npm i -g anamnesis'.";;
  esac
}

DOCKER_OK=0
check_docker() {
  if have docker && docker info >/dev/null 2>&1; then DOCKER_OK=1; return; fi
  if have docker; then
    warn "Docker is installed but the daemon isn't reachable (start it, or add your user to the 'docker' group and re-login)."
  else
    warn "Docker not found — it powers the built-in SearXNG web search."
    info "Install Docker Engine: https://docs.docker.com/engine/install/"
  fi
  info "SearXNG will be skipped now; bring it up later with 'pleiades search up'."
}

# ---- GPU detection ---------------------------------------------------------
CMAKE_ARGS=""
GPU_DESC="CPU"
resolve_gpu() {
  case "$GPU_MODE" in
    cpu) CMAKE_ARGS=""; GPU_DESC="CPU (forced)"; return;;
    gpu)
      if [ "$PLATFORM" = "macos" ]; then CMAKE_ARGS="-DGGML_METAL=on"; GPU_DESC="Apple Metal (forced)";
      else CMAKE_ARGS="-DGGML_CUDA=on"; GPU_DESC="CUDA (forced)"; fi
      return;;
  esac
  if [ "$PLATFORM" = "macos" ] && [ "$(uname -m)" = "arm64" ]; then
    CMAKE_ARGS="-DGGML_METAL=on"; GPU_DESC="Apple Metal (auto)"
  elif have nvidia-smi && nvidia-smi -L >/dev/null 2>&1; then
    CMAKE_ARGS="-DGGML_CUDA=on"; GPU_DESC="NVIDIA CUDA (auto)"
  elif have rocminfo; then
    CMAKE_ARGS="-DGGML_HIPBLAS=on"; GPU_DESC="AMD ROCm (auto)"
  elif ls /sys/class/drm/card*/device/vendor 2>/dev/null | xargs grep -l 0x1002 >/dev/null 2>&1 && have vulkaninfo; then
    # AMD GPU present but no ROCm toolkit — Vulkan backend needs no ROCm install.
    CMAKE_ARGS="-DGGML_VULKAN=on"; GPU_DESC="AMD via Vulkan (auto — install ROCm for best speed)"
  else
    CMAKE_ARGS=""; GPU_DESC="CPU (no GPU detected)"
  fi
}

# ---- repo ------------------------------------------------------------------
clone_repo() {
  if [ -d "$INSTALL_DIR/.git" ]; then
    say "Updating existing checkout at $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch origin "$BRANCH" >/dev/null 2>&1 || warn "fetch failed; using existing checkout."
    # Hard-reset to the remote branch so shallow clones / minor divergence update
    # cleanly. Untracked files (e.g. your .env) are preserved by reset.
    if ! git -C "$INSTALL_DIR" reset --hard "origin/$BRANCH" >/dev/null 2>&1; then
      git -C "$INSTALL_DIR" pull --ff-only >/dev/null 2>&1 || warn "Could not update; using existing checkout."
    fi
  else
    say "Cloning Pleiades into $INSTALL_DIR"
    git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  fi
}

# ---- python package --------------------------------------------------------
EXTRAS=""
build_extras() {
  local e="ui"   # the control panel (pleiades ui) is always available
  [ "$WITH_BROWSER" = "1" ] && e="$e,browser"
  [ "$WITH_DISCORD" = "1" ] && e="$e,discord"
  EXTRAS="$e"
}

install_python_pkg() {
  say "Creating virtual environment (.venv)"
  "$PYBIN" -m venv "$INSTALL_DIR/.venv"
  # shellcheck disable=SC1091
  . "$INSTALL_DIR/.venv/bin/activate"
  pip install --upgrade pip wheel setuptools >/dev/null

  local spec="."
  [ -n "$EXTRAS" ] && spec=".[$EXTRAS]"
  say "Installing Pleiades — $GPU_DESC build of llama-cpp-python (this can take several minutes)"
  if ! ( cd "$INSTALL_DIR" && CMAKE_ARGS="$CMAKE_ARGS" pip install -e "$spec" ); then
    if [ -n "$CMAKE_ARGS" ]; then
      warn "GPU build failed (missing CUDA/ROCm toolkit?). Retrying with a CPU build."
      ( cd "$INSTALL_DIR" && pip install -e "$spec" ) || die "pip install failed."
      GPU_DESC="CPU (GPU build failed — install the CUDA toolkit and reinstall to enable GPU)"
    else
      die "pip install failed."
    fi
  fi
}

# ---- external services -----------------------------------------------------
install_anamnesis() {
  if ! have npm; then warn "npm missing; skipping Anamnesis. Install Node, then 'npm i -g anamnesis'."; return; fi
  installed="$(npm ls -g --depth=0 anamnesis 2>/dev/null | grep -o 'anamnesis@[0-9][^ ]*' | head -n1 | cut -d@ -f2 || true)"
  latest="$(npm view anamnesis version 2>/dev/null || true)"
  if [ -n "$installed" ] && [ -n "$latest" ] && [ "$installed" = "$latest" ]; then
    say "Anamnesis $installed already installed and current — skipping"
    return
  fi
  if [ -n "$installed" ]; then
    say "Updating Anamnesis $installed -> ${latest:-latest}"
  else
    say "Installing Anamnesis (memory proxy)"
  fi
  info "npm pulls large native deps here (node-llama-cpp, transformers) — a few minutes of output is normal."
  npm install -g anamnesis --no-audit --no-fund || $SUDO npm install -g anamnesis --no-audit --no-fund || { warn "Could not install anamnesis globally; run 'npm i -g anamnesis' yourself."; return; }
  have anamnesis && anamnesis status >/dev/null 2>&1 || true
}

fetch_browser() {
  [ "$WITH_BROWSER" = "1" ] || return 0
  say "Fetching Camoufox browser"
  ( cd "$INSTALL_DIR" && . .venv/bin/activate && python -m camoufox fetch ) || warn "camoufox fetch failed; run 'python -m camoufox fetch' later."
}

install_native_runtime() {
  # The native llama-server binary (ggml-org/llama.cpp releases) is what
  # unlocks MoE expert offload (--n-cpu-moe) and is generally faster than the
  # bundled python server — autofit (pleiades/autofit.py) only gets to use
  # the good placement strategies when this is present. Without it, every
  # model launch silently falls back to the dense-only python path.
  [ "$WITH_NATIVE_RUNTIME" = "1" ] || return 0
  say "Installing native llama-server runtime (MoE expert offload)"
  if ! ( cd "$INSTALL_DIR" && "$INSTALL_DIR/.venv/bin/pleiades" runtime install ); then
    warn "Native runtime install failed; falling back to the bundled python server."
    info "Run it later from the venv:  pleiades runtime install"
  fi
}

make_env() {
  if [ ! -f "$INSTALL_DIR/.env" ] && [ -f "$INSTALL_DIR/.env.example" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    say "Created .env — set PLEIADES_MODEL_PATH to your .gguf before chatting"
  fi
}

start_searxng() {
  [ "$WITH_SEARXNG" = "1" ] || return 0
  if [ "$DOCKER_OK" != "1" ]; then warn "Skipping SearXNG (Docker not ready). Later: 'pleiades search up'."; return 0; fi
  say "Starting SearXNG (docker compose)"
  ( cd "$INSTALL_DIR" && { docker compose up -d searxng || docker-compose up -d searxng; } ) || warn "Could not start SearXNG."
}

link_cli() {
  # Make `pleiades` work from anywhere, no venv activation needed.
  mkdir -p "$HOME/.local/bin"
  ln -sf "$INSTALL_DIR/.venv/bin/pleiades" "$HOME/.local/bin/pleiades"
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) warn "$HOME/.local/bin is not on your PATH — add it to use 'pleiades' directly." ;;
  esac
}

install_desktop_app() {
  # Builds the Electron desktop app + its self-contained PyInstaller backend
  # bundle, then installs the resulting .deb so Pleiades shows up as a real,
  # searchable application in the system menu (Cinnamon/GNOME/KDE all pick up
  # /usr/share/applications/*.desktop the moment a .deb registers one) --
  # not a manually-created desktop file that a fresh install/machine would
  # lack, since this whole step lives in the installer + electron-builder.yml,
  # not something done by hand outside the build.
  [ "$WITH_DESKTOP_APP" = "1" ] || return 0
  [ "$PLATFORM" = "linux" ] || { info "Desktop app auto-install is Linux-only for now; see desktop/README for macOS/Windows."; return 0; }
  have npm || { warn "Skipping desktop app (no npm) — build it later: cd desktop && npm install && npm run dist:linux"; return 0; }

  say "Building the desktop app (Electron + bundled backend)"
  (
    set -e
    VENV_PY="$INSTALL_DIR/.venv/bin/python3.12"
    [ -x "$VENV_PY" ] || VENV_PY="$INSTALL_DIR/.venv/bin/python3"
    "$VENV_PY" -m pip show pyinstaller >/dev/null 2>&1 || "$VENV_PY" -m pip install pyinstaller
    "$INSTALL_DIR/desktop/backend-build/build.sh"
    cd "$INSTALL_DIR/desktop"
    npm install
    npm run dist:linux
  ) || { warn "Desktop app build failed — CLI/webui install is unaffected. Build it later: cd desktop && npm install && npm run dist:linux"; return 0; }

  DEB="$(find "$INSTALL_DIR/desktop/dist" -maxdepth 1 -name '*.deb' -print -quit 2>/dev/null || true)"
  if [ -z "$DEB" ]; then
    warn "Desktop app built but no .deb found under desktop/dist — install the AppImage manually instead."
    return 0
  fi

  say "Installing the desktop app ($DEB) — this needs sudo to register it system-wide"
  if have apt-get; then
    sudo apt-get install -y "$DEB" || { sudo dpkg -i "$DEB" && sudo apt-get install -f -y; } \
      || { warn "Could not install the .deb automatically. Install it yourself: sudo apt install $DEB"; return 0; }
  else
    sudo dpkg -i "$DEB" \
      || { warn "Could not install the .deb automatically. Install it yourself: sudo dpkg -i $DEB"; return 0; }
  fi
  info "Pleiades is now in your application menu/search."
}

finish() {
  echo
  say "Pleiades installed at $INSTALL_DIR  ($GPU_DESC)"
  HW_LINE="$("$INSTALL_DIR/.venv/bin/pleiades" hw 2>/dev/null | head -3 || true)"
  [ -n "$HW_LINE" ] && printf '%s\n' "$HW_LINE" | sed 's/^/  /'
  cat <<EOT

  Everything is built. Next step — just use it:

    pleiades ui                   # open the control panel in your browser

  Or from the terminal:
    pleiades model fetch bartowski/Llama-3.2-3B-Instruct-GGUF   # best quant for THIS machine
    pleiades new alice            # create a character (memory + email + vault + browser)
    pleiades chat alice           # talk to it

  GPU offload is automatic (n_gpu_layers=auto). Docs: README.md
EOT
}

main() {
  say "Pleiades installer  (platform: $PLATFORM, pkg-mgr: ${PM:-none})"
  ensure_base
  ensure_python
  [ "$WITH_DISCORD" = "1" ] || true
  ensure_node
  check_docker
  resolve_gpu
  build_extras
  clone_repo
  install_python_pkg
  install_anamnesis
  fetch_browser
  install_native_runtime
  make_env
  start_searxng
  link_cli
  install_desktop_app
  finish
}
main "$@"
