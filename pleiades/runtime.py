"""Inference runtime management.

Pleiades can run models through two OpenAI-compatible runtimes:

  * the bundled python server (`llama_cpp.server`) — always present, but it
    cannot express MoE expert offload (`--n-cpu-moe` / `--override-tensor`);
  * the native llama.cpp `llama-server` binary — faster and unlocks the MoE
    split autofit wants. `pleiades runtime install` fetches an official
    prebuilt from ggml-org/llama.cpp releases into ~/.pleiades/runtime/.

`find_native()` + `caps()` tell autofit what the machine can actually do.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable, Optional

from . import config
from .autofit import RuntimeCaps


class RuntimeError_(RuntimeError):
    pass


def runtime_dir() -> Path:
    d = config.PLEIADES_HOME / "runtime"
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_native(prefer_moe_opts: bool = False) -> Optional[str]:
    """Path to a llama-server binary.

    Resolution order: PLEIADES_RUNTIME_BIN (explicit override) → managed
    copies under ~/.pleiades/runtime → PATH. Two managed builds may
    coexist: the mainline build and the MoE-prefill fork (thecodacus/
    llama.cpp — pinned-host-memory + expert-prefetch, env-gated, token-
    identical) under runtime/moe-fork/. MoE models prefer the fork when
    present; dense models prefer mainline. Either runs both — this is a
    preference, not a requirement.
    """
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    override = os.environ.get("PLEIADES_RUNTIME_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    managed = sorted(runtime_dir().rglob(exe))
    if managed:
        def rank(p: Path) -> int:
            s = str(p)
            # Benchmarked on this box (RTX 2080 Ti, PCIe 3.0, Ornith-35B
            # 256-expert MoE, -ncmoe 40, ub 2048, r=2, quiet machine):
            #   mainline: pp2048 735 t/s · tg64 26.8 t/s
            #   fork+opts: pp2048 794 t/s (+8%) · tg64 24.9 t/s (−5%)
            # Chat is decode-bound → mainline stays the default everywhere.
            # The fork's pinning/prefetch pays on PCIe4 boxes and prefill-
            # heavy batch jobs; opt in via PLEIADES_RUNTIME_BIN.
            if "cuda-main" in s:
                return 0                             # our mainline CUDA build
            if "moe-fork" in s:
                return 1                             # benchmarked: opt-in only
            return 2                                 # legacy prebuilts (e.g. vulkan zip)
        _ = prefer_moe_opts  # kept for API stability; see benchmark note
        return str(min(managed, key=rank))
    return shutil.which("llama-server")


def find_native_cpp_engine() -> Optional[str]:
    """Path to `pleiades-engine-server` -- the from-scratch libllama-based
    engine built under `engine/` (see
    docs/specs/2026-07-21-native-inference-engine-design.md), NOT the
    upstream `llama-server` binary `find_native()` resolves above.

    Deliberately kept separate from `find_native()`/`rank()`: that function
    returns a path assumed to accept llama-server's actual CLI flag shape
    (-m/-c/-ngl/--jinja/-fa/-ctk/-ctv/--n-cpu-moe/--spec-type/...), and
    `build_command()` builds those flags assuming whatever binary it gets
    back understands all of them. `pleiades-engine-server` only supports
    load/resize/chat-completion today -- no flash-attention toggle, no KV
    quantization, no MoE expert offload, no speculative decoding. Reusing
    the same ranked list would either silently drop flags it doesn't
    understand or require brittle per-flag filtering inside that branch.
    A second consultant (Kimi, asked independently) agreed: keep this a
    clearly separate resolver + a clearly separate `launch.py` branch,
    with its own feature flag, until real feature parity exists -- don't
    make the capability gap implicit.

    Resolution order: PLEIADES_NATIVE_CPP_ENGINE_BIN (explicit override) ->
    `engine/build/pleiades-engine-server` relative to this checkout (how
    Phase 1-4 built it locally; there's no `pleiades runtime install`-style
    packaged release of this yet) -> PATH.
    """
    exe = "pleiades-engine-server.exe" if os.name == "nt" else "pleiades-engine-server"
    override = os.environ.get("PLEIADES_NATIVE_CPP_ENGINE_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    repo_root = Path(__file__).resolve().parent.parent
    local_build = repo_root / "engine" / "build" / exe
    if local_build.is_file():
        return str(local_build)
    return shutil.which(exe)


def native_env(binary: str) -> dict:
    """Environment needed to RUN a managed llama-server binary.

    Locally-built trees link their impl libraries with absolute rpaths that
    break the moment the tree is moved or renamed (runtime/src →
    runtime/cuda-main); official release zips use $ORIGIN and don't care.
    Putting the binary's own directory on LD_LIBRARY_PATH makes both cases
    relocatable. Callers merge this over os.environ.
    """
    d = str(Path(binary).parent)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": f"{d}:{cur}" if cur else d}


_caps_cache: dict[str, RuntimeCaps] = {}


def caps(prefer_moe_opts: bool = False) -> RuntimeCaps:
    """What the best available runtime supports (probed once per binary)."""
    native = find_native(prefer_moe_opts=prefer_moe_opts)
    key = native or "python"
    if key in _caps_cache:
        return _caps_cache[key]
    c = RuntimeCaps()
    if native:
        try:
            out = subprocess.run([native, "--help"], capture_output=True,
                                 text=True, timeout=10,
                                 env={**os.environ, **native_env(native)})
            text = out.stdout + out.stderr
            c.native = True
            c.moe_offload = ("--n-cpu-moe" in text or "--override-tensor" in text
                             or "-ot" in text)
            # Provenance-based: only our managed fork build ships the
            # env-gated prefill opts. Setting the env vars on a mainline
            # build is harmless (ignored), so a false negative here costs
            # nothing and a false positive is impossible.
            c.moe_prefill_opts = "moe-fork" in native
        except (OSError, subprocess.SubprocessError):
            pass
    _caps_cache[key] = c
    return c


def clear_caps_cache() -> None:
    _caps_cache.clear()


# --------------------------------------------------------------------------- #
# Install (prebuilt from ggml-org/llama.cpp releases)
# --------------------------------------------------------------------------- #
def _platform_tokens() -> tuple[list[str], str]:
    """(os tokens, arch token) to match against release asset names."""
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if sys.platform == "darwin":
        return ["macos"], arch
    if os.name == "nt":
        return ["win"], arch
    return ["ubuntu", "linux"], arch


def _backend_priority() -> list[str]:
    """Preferred backend tokens, best first, based on detected hardware."""
    from .hardware import detect
    hw = detect()
    gpu = hw.gpu
    if gpu and gpu.vendor == "nvidia":
        return ["cuda", "vulkan", ""]
    if gpu and gpu.vendor == "amd":
        return ["vulkan", "hip", "rocm", ""]
    if gpu and gpu.vendor == "intel":
        # Vulkan needs only the normal graphics driver (broadly available,
        # incl. older Arc/iGPU generations); SYCL/oneAPI can be faster on
        # recent Arc/Battlemage but needs the heavier oneAPI runtime
        # installed, so it's the upgrade path, not the default.
        return ["vulkan", "sycl", ""]
    if gpu and gpu.vendor == "apple":
        return [""]  # macos builds ship Metal by default
    return ["", "vulkan"]


# Real llama.cpp CI matrix names outside our two supported arches (verified
# 2026-07-24 against the live ggml-org/llama.cpp release assets, which
# included "llama-*-bin-ubuntu-s390x.tar.gz" -- with neither an "x64" nor
# "arm64" token, it was slipping through the fallback branch below as if it
# were a generic/archless x64 build). Exclude explicitly rather than only
# positively matching "x64"/"arm64", since the fallback branch exists for
# genuinely archless names (there are none right now, but the previous,
# narrower version of this filter assumed there always would be).
_FOREIGN_ARCH_TOKENS = ("s390x", "ppc64le", "ppc64", "riscv64", "mips64", "sparc64")


def pick_asset(assets: list[dict]) -> Optional[dict]:
    """Choose the best release asset for this machine. Pure logic, testable."""
    os_tokens, arch = _platform_tokens()
    names = [(a, a.get("name", "").lower()) for a in assets
             if a.get("name", "").lower().endswith((".zip", ".tar.gz"))
             and not a.get("name", "").lower().startswith(("cudart-", "llama-ui"))
             and "xcframework" not in a.get("name", "").lower()
             and "-ui." not in a.get("name", "").lower()]
    plat = [(a, n) for a, n in names
            if any(t in n for t in os_tokens)
            and (arch in n or ("x64" not in n and "arm64" not in n
                                and not any(f in n for f in _FOREIGN_ARCH_TOKENS)))]
    if not plat:
        return None
    for backend in _backend_priority():
        for a, n in plat:
            if backend and backend in n:
                return a
            if not backend and not any(b in n for b in
                    ("cuda", "vulkan", "hip", "rocm", "sycl", "cann", "openvino", "opencl")):
                return a
    return plat[0][0]


_CUDA_VERSION_RE = re.compile(r"-cuda-(\d+\.\d+)-")


def _cudart_asset_for(main_name: str, assets: list[dict]) -> Optional[dict]:
    """Find the cudart-*.zip sidecar matching a chosen win-cuda-*.zip asset's
    CUDA version (e.g. "llama-b1-bin-win-cuda-12.4-x64.zip" ->
    "cudart-llama-bin-win-cuda-12.4-x64.zip") -- llama.cpp ships these as
    separate downloads (verified against the live b10107 release: the main
    Windows CUDA build does NOT bundle cudart/cublas DLLs itself), so
    llama-server.exe is missing them on any machine without the CUDA
    toolkit already installed unless this sidecar is also fetched and
    unzipped alongside it. None if main_name isn't a CUDA build or no
    version-matched sidecar exists in this release."""
    m = _CUDA_VERSION_RE.search(main_name.lower())
    if not m:
        return None
    version = m.group(1)
    for a in assets:
        name = a.get("name", "").lower()
        if name.startswith("cudart-") and name.endswith(".zip") and f"-cuda-{version}-" in name:
            return a
    return None


def install(log: Optional[Callable[[str], None]] = None) -> str:
    """Download the best llama.cpp prebuilt for this machine. Returns the path."""
    import httpx

    say = log or (lambda _m: None)
    say("looking up the latest llama.cpp release…")
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        r = c.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        r.raise_for_status()
        rel = r.json()
        assets = rel.get("assets", [])
        asset = pick_asset(assets)
        if not asset:
            raise RuntimeError_(
                "No matching prebuilt for this platform in the latest llama.cpp "
                "release. Build from source: https://github.com/ggml-org/llama.cpp")
        say(f"downloading {asset['name']} ({rel.get('tag_name', '')}) …")
        dest = runtime_dir() / asset["name"]
        with c.stream("GET", asset["browser_download_url"]) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f.write(chunk)

        # Windows CUDA builds don't bundle cublas/cudart DLLs -- llama-
        # server.exe fails to start on a machine without the CUDA toolkit
        # already installed unless this sidecar is fetched too (see
        # _cudart_asset_for's docstring; real asset layout verified
        # directly against the live release, not assumed).
        cudart_dest = None
        if os.name == "nt":
            cudart_asset = _cudart_asset_for(asset["name"], assets)
            if cudart_asset:
                say(f"downloading {cudart_asset['name']} (cudart/cublas DLLs) …")
                cudart_dest = runtime_dir() / cudart_asset["name"]
                with c.stream("GET", cudart_asset["browser_download_url"]) as resp:
                    resp.raise_for_status()
                    with open(cudart_dest, "wb") as f:
                        for chunk in resp.iter_bytes(1024 * 1024):
                            f.write(chunk)

    say("extracting…")
    target = runtime_dir() / "llama.cpp"
    if target.exists():
        shutil.rmtree(target)
    if dest.name.endswith(".tar.gz"):
        import tarfile
        with tarfile.open(dest, "r:gz") as t:
            t.extractall(target, filter="data")
    else:
        with zipfile.ZipFile(dest) as z:
            z.extractall(target)
    dest.unlink(missing_ok=True)

    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    found = list(target.rglob(exe))
    if not found:
        raise RuntimeError_("Archive did not contain llama-server.")
    binary = found[0]
    if os.name != "nt":
        for f in binary.parent.iterdir():  # binaries + bundled shared libs
            try:
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
            except OSError:
                pass

    if cudart_dest is not None:
        # Flat DLLs at the zip root (verified against the real b10107
        # cudart-llama-bin-win-cuda-12.4-x64.zip: cublas64_12.dll,
        # cublasLt64_12.dll, cudart64_12.dll, no subdirectories) --
        # extract directly next to llama-server.exe so Windows' normal
        # DLL search order (same directory as the .exe) finds them.
        with zipfile.ZipFile(cudart_dest) as z:
            z.extractall(binary.parent)
        cudart_dest.unlink(missing_ok=True)

    clear_caps_cache()
    say(f"installed {binary}")
    return str(binary)


def status() -> dict:
    native = find_native()
    c = caps()
    out = {"native": native, "moe_offload": c.moe_offload}
    if native:
        try:
            o = subprocess.run([native, "--version"], capture_output=True,
                               text=True, timeout=10)
            m = re.search(r"\bb?\d{4,}\b", o.stdout + o.stderr)
            out["version"] = m.group(0) if m else ""
        except (OSError, subprocess.SubprocessError):
            out["version"] = ""
    return out
