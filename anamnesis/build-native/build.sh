#!/usr/bin/env bash
# Build a pruned, self-contained copy of Anamnesis's node_modules for bundling
# into the Pleiades desktop app (mirrors desktop/backend-build/build.sh's role
# for the Python backend -- see that file + docs/specs/2026-07-25-anamnesis-
# vendoring-and-installer-plan.md Part 1b for the "why").
#
# What this does NOT do: cross-compile for other platforms. This machine can
# only produce (and verify) a Linux x64 payload. Windows/macOS need their own
# run of this script on that platform -- see the plan doc's phase map, this is
# explicitly flagged there as a multi-session piece.
#
# Reproduce with:
#   ~/Pleiades/anamnesis/build-native/build.sh
#
# Output: ~/Pleiades/anamnesis/build-native/dist/anamnesis/
#   (a full copy of src/ + a pruned node_modules/, ready for extraResources)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # .../anamnesis
BUILD_DIR="$REPO_ROOT/build-native"
DIST_DIR="$BUILD_DIR/dist/anamnesis"

PLATFORM="$(uname -s)"
ARCH="$(uname -m)"
if [ "$PLATFORM" != "Linux" ] || [ "$ARCH" != "x86_64" ]; then
  echo "error: this script has only been built + verified for Linux x86_64." >&2
  echo "       Got: $PLATFORM $ARCH. See the header comment before extending it." >&2
  exit 1
fi

say()  { printf "==> %s\n" "$*"; }

say "Cleaning previous build output"
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

say "Copying Anamnesis source (src/, package.json, config.json) -- not node_modules, that gets a fresh install below"
rsync -a --exclude 'node_modules' --exclude 'build-native' --exclude '.git' "$REPO_ROOT/" "$DIST_DIR/"

say "Installing production dependencies fresh into the build copy"
( cd "$DIST_DIR" && npm ci --omit=dev --no-audit --no-fund )

BEFORE_SIZE="$(du -sh "$DIST_DIR/node_modules" | cut -f1)"

say "Pruning @node-llama-cpp: keeping only linux-x64 (CPU)"
# Deliberate choice, not an oversight -- see the plan doc / project memory for
# the full reasoning: this powers Anamnesis's background extraction/foresight/
# persona-drift helper LLM, not the user-facing chat model. Its own default
# gpuLayerBudgetMB (512, see src/lib/brain.js) is modest by design -- CPU-only
# for a small background helper is an acceptable v1 trade against the ~680MB
# saved by not shipping 5 GPU/other-arch variants nobody's install will ever
# load. Revisit only if this specific component's speed becomes a real,
# measured complaint -- not preemptively.
NLC_DIR="$DIST_DIR/node_modules/@node-llama-cpp"
if [ -d "$NLC_DIR" ]; then
  for variant in "$NLC_DIR"/*/; do
    name="$(basename "$variant")"
    if [ "$name" != "linux-x64" ]; then
      rm -rf "$variant"
    fi
  done
fi

say "Pruning onnxruntime-node: keeping only linux/x64, dropping its GPU execution providers"
# LocalEmbedder (src/lib/local-embedder.js) runs all-MiniLM-L6-v2 CPU-only --
# the CUDA/TensorRT execution provider .so files (the overwhelming majority
# of this package's size: ~328MB of ~370MB in the linux/x64 dir alone) are
# never loaded. Verified via a fresh find/du pass on 2026-07-25, not assumed.
ORT_NAPI_DIR="$DIST_DIR/node_modules/onnxruntime-node/bin/napi-v3"
if [ -d "$ORT_NAPI_DIR" ]; then
  for os_dir in "$ORT_NAPI_DIR"/*/; do
    os_name="$(basename "$os_dir")"
    if [ "$os_name" != "linux" ]; then
      rm -rf "$os_dir"
    else
      for arch_dir in "$os_dir"*/; do
        arch_name="$(basename "$arch_dir")"
        if [ "$arch_name" != "x64" ]; then
          rm -rf "$arch_dir"
        fi
      done
    fi
  done
  rm -f "$ORT_NAPI_DIR/linux/x64/libonnxruntime_providers_cuda.so"
  rm -f "$ORT_NAPI_DIR/linux/x64/libonnxruntime_providers_tensorrt.so"
fi

AFTER_SIZE="$(du -sh "$DIST_DIR/node_modules" | cut -f1)"

say "Verifying the pruned copy still boots and answers a real health check"
SCRATCH_HOME="$(mktemp -d)"
mkdir -p "$SCRATCH_HOME/.anamnesis"
cat > "$SCRATCH_HOME/.anamnesis/daemon.json" << 'EOF'
{ "controlPort": 19898 }
EOF
( HOME="$SCRATCH_HOME" node "$DIST_DIR/src/daemon.js" > "$BUILD_DIR/last-verify-boot.log" 2>&1 & echo $! > "$BUILD_DIR/last-verify.pid" )
sleep 3
HEALTH="$(curl -s --max-time 5 http://127.0.0.1:19898/status || true)"
VERIFY_PID="$(cat "$BUILD_DIR/last-verify.pid" 2>/dev/null || true)"
[ -n "$VERIFY_PID" ] && kill -9 "$VERIFY_PID" 2>/dev/null || true
rm -rf "$SCRATCH_HOME" "$BUILD_DIR/last-verify.pid"

if [[ "$HEALTH" != *'"status":"ok"'* ]]; then
  echo "error: pruned build failed its own boot/health check. Log:" >&2
  cat "$BUILD_DIR/last-verify-boot.log" >&2
  exit 1
fi
rm -f "$BUILD_DIR/last-verify-boot.log"

say "OK -- pruned copy boots and reports healthy: $HEALTH"
say "node_modules: $BEFORE_SIZE -> $AFTER_SIZE"
say "Built: $DIST_DIR"
du -sh "$DIST_DIR"
