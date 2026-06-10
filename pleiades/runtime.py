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


def find_native() -> Optional[str]:
    """Path to a llama-server binary: our managed copy first, then PATH."""
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    managed = list(runtime_dir().rglob(exe))
    if managed:
        return str(managed[0])
    return shutil.which("llama-server")


_caps_cache: dict[str, RuntimeCaps] = {}


def caps() -> RuntimeCaps:
    """What the best available runtime supports (probed once per process)."""
    native = find_native()
    key = native or "python"
    if key in _caps_cache:
        return _caps_cache[key]
    c = RuntimeCaps()
    if native:
        try:
            out = subprocess.run([native, "--help"], capture_output=True,
                                 text=True, timeout=10)
            text = out.stdout + out.stderr
            c.native = True
            c.moe_offload = ("--n-cpu-moe" in text or "--override-tensor" in text
                             or "-ot" in text)
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
    if gpu and gpu.vendor == "apple":
        return [""]  # macos builds ship Metal by default
    return ["", "vulkan"]


def pick_asset(assets: list[dict]) -> Optional[dict]:
    """Choose the best release asset for this machine. Pure logic, testable."""
    os_tokens, arch = _platform_tokens()
    names = [(a, a.get("name", "").lower()) for a in assets
             if a.get("name", "").lower().endswith(".zip")]
    plat = [(a, n) for a, n in names
            if any(t in n for t in os_tokens) and (arch in n or "x64" not in n and "arm64" not in n)]
    if not plat:
        return None
    for backend in _backend_priority():
        for a, n in plat:
            if backend and backend in n:
                return a
            if not backend and not any(b in n for b in ("cuda", "vulkan", "hip", "rocm", "sycl", "cann")):
                return a
    return plat[0][0]


def install(log: Optional[Callable[[str], None]] = None) -> str:
    """Download the best llama.cpp prebuilt for this machine. Returns the path."""
    import httpx

    say = log or (lambda _m: None)
    say("looking up the latest llama.cpp release…")
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        r = c.get("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest")
        r.raise_for_status()
        rel = r.json()
        asset = pick_asset(rel.get("assets", []))
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

    say("extracting…")
    target = runtime_dir() / "llama.cpp"
    if target.exists():
        shutil.rmtree(target)
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
