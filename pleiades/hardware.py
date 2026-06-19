"""Hardware detection + model placement planning.

Pleiades is a self-building inference engine: this module figures out what the
machine has (NVIDIA, AMD, or Apple Silicon GPU; RAM; cores) and maps a GGUF
model onto it — how many layers to offload (`n_gpu_layers`) and which
quantization to download — without the user tuning anything.

Everything here is heuristic-with-margin: estimates are conservative, the
reasoning is always reported, and an explicit integer override always wins.
No third-party dependencies; detection shells out to `nvidia-smi`/`rocm-smi`
or reads sysfs, and the GGUF header parser is pure Python.
"""

from __future__ import annotations

import json
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #


@dataclass
class GPU:
    vendor: str          # nvidia | amd | apple
    name: str
    vram_total: int      # bytes
    vram_free: int       # bytes (== total if unknown)


@dataclass
class Hardware:
    gpus: list[GPU] = field(default_factory=list)
    ram_total: int = 0       # bytes
    ram_available: int = 0   # bytes
    cpu_threads: int = 1
    unified_memory: bool = False  # Apple Silicon: GPU shares system RAM

    @property
    def gpu(self) -> Optional[GPU]:
        """The biggest GPU (llama.cpp single-GPU default)."""
        return max(self.gpus, key=lambda g: g.vram_total) if self.gpus else None

    def describe(self) -> str:
        lines = []
        for g in self.gpus:
            lines.append(f"GPU: {g.name} ({g.vendor}) — "
                         f"{_gb(g.vram_free)} free / {_gb(g.vram_total)} VRAM")
        if not self.gpus:
            lines.append("GPU: none detected")
        lines.append(f"RAM: {_gb(self.ram_available)} available / {_gb(self.ram_total)}"
                     + (" (unified with GPU)" if self.unified_memory else ""))
        lines.append(f"CPU threads: {self.cpu_threads}")
        return "\n".join(lines)


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.1f} GiB"


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _detect_nvidia() -> list[GPU]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                total = int(float(parts[1])) * 1024 ** 2  # MiB -> bytes
                free = int(float(parts[2])) * 1024 ** 2
            except ValueError:
                continue
            gpus.append(GPU("nvidia", parts[0], total, free))
    return gpus


# PowerShell: AMD adapters + dedicated VRAM from the display-class registry key.
# Win32_VideoController.AdapterRAM is a uint32 (caps at 4 GiB), so
# HardwareInformation.qwMemorySize is the only honest source. QWORD on most
# drivers, REG_BINARY on some — both handled. Output: JSON array.
_WIN_AMD_PS = (
    # Registry configs outlive removed hardware (a swapped-out card leaves a
    # ghost entry with full qwMemorySize) — intersect with live adapters.
    "$live=@(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue"
    " | ForEach-Object Name);"
    "$k='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
    "{4d36e968-e325-11ce-bfc1-08002be10318}';"
    "$out=@(Get-ChildItem $k -ErrorAction SilentlyContinue | ForEach-Object {"
    "$p=Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue;"
    "if(($p.DriverDesc -match 'AMD|Radeon') -and ($live -contains $p.DriverDesc)"
    " -and $p.'HardwareInformation.qwMemorySize'){"
    "$v=$p.'HardwareInformation.qwMemorySize';"
    "if($v -is [byte[]]){$v=[System.BitConverter]::ToInt64($v,0)};"
    "[pscustomobject]@{name=$p.DriverDesc;vram=[int64]$v}}});"
    "ConvertTo-Json -InputObject $out -Compress"
)


def _parse_win_amd(out: str) -> list[GPU]:
    """Parse _WIN_AMD_PS JSON into GPUs. Free VRAM unknown -> free == total
    (module convention, see GPU dataclass)."""
    try:
        rows = json.loads(out or "[]")
    except ValueError:
        return []
    if isinstance(rows, dict):  # ConvertTo-Json collapses single-element arrays
        rows = [rows]
    gpus, seen = [], set()
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        try:
            total = int(r.get("vram") or 0)
        except (TypeError, ValueError):
            continue
        name = str(r.get("name") or "AMD GPU")
        if total > 0 and (name, total) not in seen:  # per-config duplicate keys
            seen.add((name, total))
            gpus.append(GPU("amd", name, total, total))
    return gpus


def _detect_amd_windows() -> list[GPU]:
    """AMD on Windows: registry via PowerShell. No vendor SDK, no ROCm needed."""
    out = _run(["powershell", "-NoProfile", "-Command", _WIN_AMD_PS], timeout=15.0)
    return _parse_win_amd(out)


def _detect_amd() -> list[GPU]:
    """AMD VRAM via sysfs (works without ROCm installed), rocm-smi as backup."""
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor.lower() != "0x1002":  # AMD
            continue
        try:
            total = int((dev / "mem_info_vram_total").read_text().strip())
            used_path = dev / "mem_info_vram_used"
            used = int(used_path.read_text().strip()) if used_path.is_file() else 0
        except (OSError, ValueError):
            continue
        name = "AMD GPU"
        try:
            name = (dev / "product_name").read_text().strip() or name
        except OSError:
            pass
        gpus.append(GPU("amd", name, total, max(total - used, 0)))
    if gpus:
        return gpus
    # rocm-smi fallback (CSV: device,"VRAM Total Memory (B)","VRAM Total Used Memory (B)")
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    for line in out.splitlines()[1:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 3:
            try:
                total, used = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            gpus.append(GPU("amd", parts[0] or "AMD GPU", total, max(total - used, 0)))
    return gpus


def _ram() -> tuple[int, int]:
    """(total, available) bytes, cross-platform, stdlib only."""
    if sys.platform == "linux":
        info: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                num = rest.strip().split()
                if num:
                    info[key] = int(num[0]) * 1024
        except (OSError, ValueError):
            return 0, 0
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", total)
        return total, avail
    if sys.platform == "darwin":
        try:
            total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
        except ValueError:
            total = 0
        return total, int(total * 0.7)  # macOS reclaims aggressively; be conservative
    if os.name == "nt":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys, stat.ullAvailPhys
    return 0, 0


def detect() -> Hardware:
    """Detect GPUs, RAM, and CPU threads. Cheap enough to call per launch."""
    hw = Hardware()
    hw.cpu_threads = os.cpu_count() or 1
    hw.ram_total, hw.ram_available = _ram()

    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Apple Silicon: the GPU shares system RAM (unified memory).
        hw.unified_memory = True
        budget = int(hw.ram_total * 0.75)  # llama.cpp Metal working-set guidance
        hw.gpus = [GPU("apple", f"Apple Silicon ({platform.processor() or 'arm64'})",
                       budget, budget)]
        return hw

    hw.gpus = _detect_nvidia()
    if sys.platform == "linux":
        hw.gpus += _detect_amd()
    elif os.name == "nt" and not hw.gpus:
        # nvidia-smi found nothing — check for AMD (Radeon offloads via the
        # native Vulkan llama-server runtime; see install.ps1 / autofit).
        hw.gpus += _detect_amd_windows()
    return hw


# --------------------------------------------------------------------------- #
# GGUF metadata
# --------------------------------------------------------------------------- #

_GGUF_MAGIC = b"GGUF"
# value-type code -> struct fmt (fixed-size scalars)
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass
class GGUFMeta:
    path: str
    file_size: int
    n_layers: int = 0
    n_embd: int = 0
    n_head: int = 0
    n_head_kv: int = 0
    n_ctx_train: int = 0
    quant: str = ""
    architecture: str = ""
    # FFN / MoE structure (0 = absent/unknown)
    ffn_len: int = 0                 # dense feed-forward width
    expert_count: int = 0            # total experts (0 = dense model)
    expert_used: int = 0             # experts active per token
    expert_ffn_len: int = 0          # per-expert FFN width
    shared_ffn_len: int = 0          # shared-expert FFN width

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1

    @property
    def estimated(self) -> bool:
        return self.n_layers == 0


def _read_scalar(f: BinaryIO, vtype: int):
    if vtype in _SCALARS:
        fmt = _SCALARS[vtype]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if vtype == _STRING:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")
    if vtype == _ARRAY:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in _SCALARS:  # skip without materialising
            f.seek(count * struct.calcsize(_SCALARS[etype]), 1)
        elif etype == _STRING:
            for _ in range(count):
                (n,) = struct.unpack("<Q", f.read(8))
                f.seek(n, 1)
        else:
            raise ValueError(f"nested arrays unsupported (etype={etype})")
        return None  # arrays are never needed for planning
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_gguf_meta(path: "str | Path") -> GGUFMeta:
    """Parse the GGUF header for the handful of keys placement needs.

    Never raises on malformed files — returns size-only metadata instead, and
    the planner falls back to size-based estimates.
    """
    p = Path(path)
    meta = GGUFMeta(path=str(p), file_size=p.stat().st_size if p.is_file() else 0)
    try:
        with open(p, "rb") as f:
            if f.read(4) != _GGUF_MAGIC:
                return meta
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:  # v1 used 32-bit counts; ancient, skip parsing
                return meta
            f.seek(8, 1)  # tensor_count
            (n_kv,) = struct.unpack("<Q", f.read(8))
            kvs: dict[str, object] = {}
            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", f.read(8))
                key = f.read(klen).decode("utf-8", errors="replace")
                (vtype,) = struct.unpack("<I", f.read(4))
                val = _read_scalar(f, vtype)
                if val is not None:
                    kvs[key] = val
        arch = str(kvs.get("general.architecture", ""))
        meta.architecture = arch

        def _i(key: str) -> int:
            v = kvs.get(f"{arch}.{key}")
            return int(v) if isinstance(v, (int, float)) else 0

        meta.n_layers = _i("block_count")
        meta.n_embd = _i("embedding_length")
        meta.n_head = _i("attention.head_count")
        meta.n_head_kv = _i("attention.head_count_kv") or meta.n_head
        meta.n_ctx_train = _i("context_length")
        meta.ffn_len = _i("feed_forward_length")
        meta.expert_count = _i("expert_count")
        meta.expert_used = _i("expert_used_count")
        meta.expert_ffn_len = _i("expert_feed_forward_length")
        meta.shared_ffn_len = _i("expert_shared_feed_forward_length")
        ftype = kvs.get("general.file_type")
        if isinstance(ftype, int):
            meta.quant = _FILE_TYPES.get(ftype, f"ftype-{ftype}")
    except (OSError, ValueError, struct.error):
        pass
    return meta


# general.file_type -> human quant label (llama.cpp LLAMA_FTYPE_*)
_FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
               9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
               14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
               18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 24: "IQ4_NL",
               25: "IQ3_S", 26: "IQ2_S", 27: "IQ4_XS", 30: "BF16"}


# --------------------------------------------------------------------------- #
# Placement planning
# --------------------------------------------------------------------------- #

# Fixed costs we reserve before placing layers: compute graph / scratch buffers
# plus runtime context. Deliberately conservative.
_OVERHEAD = 800 * 1024 ** 2
_DEFAULT_LAYERS = 32  # estimate when the header could not be parsed


@dataclass
class Plan:
    n_gpu_layers: int          # -1 all, 0 cpu, else partial
    n_layers: int              # total layers in the model
    reason: str
    fits_fully: bool

    def describe(self) -> str:
        return self.reason


def _kv_bytes(meta: GGUFMeta, n_ctx: int) -> int:
    """KV cache estimate (f16 K+V per layer)."""
    if not (meta.n_layers and meta.n_embd and meta.n_head):
        # ~order-of-magnitude fallback: 1 MiB per layer per 1k ctx
        return kv_bytes_per_token(meta) * max(n_ctx, 512)
    return kv_bytes_per_token(meta) * n_ctx


def plan(meta: GGUFMeta, n_ctx: int, hw: Optional[Hardware] = None) -> Plan:
    """Map a model onto the detected hardware: how many layers go to the GPU."""
    hw = hw or detect()
    n_layers = meta.n_layers or _DEFAULT_LAYERS
    gpu = hw.gpu

    if gpu is None:
        return Plan(0, n_layers, "no GPU detected — running on CPU "
                    f"({hw.cpu_threads} threads)", False)

    budget = gpu.vram_free - _OVERHEAD
    kv_total = _kv_bytes(meta, n_ctx)
    # +1 layer's worth approximates the output/embedding tensors.
    per_layer = meta.file_size / (n_layers + 1) if meta.file_size else 0
    need = meta.file_size + kv_total

    if per_layer <= 0:
        return Plan(0, n_layers, "could not size the model — running on CPU", False)

    if need <= budget:
        est = " (estimated layers)" if meta.estimated else ""
        return Plan(-1, n_layers,
                    f"full offload: model {_gb(meta.file_size)} + KV {_gb(kv_total)} "
                    f"fits {_gb(gpu.vram_free)} free on {gpu.name}{est}", True)

    kv_per_layer = kv_total / n_layers
    fit = int(budget // (per_layer + kv_per_layer))
    fit = max(min(fit, n_layers), 0)
    if fit == 0:
        return Plan(0, n_layers,
                    f"model {_gb(meta.file_size)} + KV {_gb(kv_total)} exceeds "
                    f"{_gb(gpu.vram_free)} free on {gpu.name} — running on CPU", False)
    return Plan(fit, n_layers,
                f"partial offload: {fit}/{n_layers} layers onto {gpu.name} "
                f"({_gb(gpu.vram_free)} free; model {_gb(meta.file_size)}, "
                f"KV {_gb(kv_total)})", False)


def resolve_layers(value: "int | str", model_path: str, n_ctx: int,
                   hw: Optional[Hardware] = None) -> tuple[int, str]:
    """Turn an `n_gpu_layers` setting into a concrete int at launch time.

    Integers (and integer strings) pass through untouched; "auto"/"" plans
    from the hardware. Returns (layers, reason).
    """
    if isinstance(value, bool):  # guard: bool is an int subclass
        value = int(value)
    if isinstance(value, int):
        return value, "explicit setting"
    text = str(value).strip().lower()
    if text not in ("", "auto"):
        try:
            return int(text), "explicit setting"
        except ValueError:
            pass
    p = plan(read_gguf_meta(model_path), n_ctx, hw)
    return p.n_gpu_layers, p.reason


# --------------------------------------------------------------------------- #
# Quant selection (for the HF downloader)
# --------------------------------------------------------------------------- #

# Best-first when the hardware can take it. F16/BF16 excluded on purpose:
# Q8_0 is near-lossless at half the size.
_QUANT_PREFERENCE = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M",
                     "Q4_K_S", "IQ4_XS", "Q4_0", "Q3_K_L", "Q3_K_M", "IQ3_S",
                     "Q3_K_S", "Q2_K", "IQ2_S"]
# When layers won't fit on the GPU and the CPU does the work, Q4_K_M is the
# sweet spot — Q8/Q6 are ~2x the bytes for marginal quality, i.e. ~2x slower.
_CPU_PREFERENCE = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q4_0", "Q3_K_M",
                   "Q3_K_S", "Q2_K"]
_QUANT_RE = re.compile(r"(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|BF16|F16|F32)",
                       re.IGNORECASE)
_SPLIT_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)


def quant_of(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else ""


def is_mmproj(filename: str) -> bool:
    """Multimodal projector sidecars (mmproj-*.gguf) are not standalone models.

    Left in the candidate pool they win every "fits in VRAM" contest (~1 GB)
    and the fetcher proudly downloads a file llama.cpp can't load as a model.
    """
    return Path(filename).name.lower().startswith("mmproj")


def is_mtp(filename: str) -> bool:
    """Multi-token-prediction draft-head sidecars (*-MTP.gguf, mtp-*.gguf, or
    files under an 'MTP/' folder) are not standalone models either.

    Their filenames embed a real quant label (e.g. 'gemma-...-BF16-MTP.gguf')
    but the file is a tiny ~1 GB draft head, not the full model. Left in the
    candidate pool, a sidecar like that is far smaller than the real BF16/
    F16/Q8_0 file of the same label, so the "keep the smallest file per
    quant" dedup in group_split_ggufs picks the sidecar and reports its tiny
    size + inflated tok/s estimate as if it were the real quant.
    """
    return "mtp" in Path(filename).name.lower()


def group_split_ggufs(files: list[dict]) -> list[dict]:
    """Group multi-part GGUFs (-00001-of-00003.gguf) into one logical entry.

    Input/output dicts: {"name": str, "size": int} (+ "parts": [names] on groups).
    Multimodal projector sidecars (mmproj-*) and multi-token-prediction
    sidecars (*-MTP.gguf) are excluded — they're companions to a model,
    never the model.
    """
    groups: dict[str, dict] = {}
    out: list[dict] = []
    for f in files:
        name = f["name"]
        if not name.lower().endswith(".gguf") or is_mmproj(name) or is_mtp(name):
            continue
        m = _SPLIT_RE.search(name)
        if not m:
            out.append({"name": name, "size": int(f.get("size", 0)), "parts": [name]})
            continue
        base = name[: m.start()]
        g = groups.setdefault(base, {"name": name.replace(m.group(0), ".gguf"),
                                     "size": 0, "parts": []})
        g["size"] += int(f.get("size", 0))
        g["parts"].append(name)
    for g in groups.values():
        g["parts"].sort()
        out.append(g)
    return out


def pick_quant(files: list[dict], hw: Optional[Hardware] = None,
               n_ctx: int = 8192) -> Optional[dict]:
    """Choose the best GGUF from an HF repo file list for this machine.

    files: [{"name": ..., "size": ...}]. Returns the chosen (grouped) entry
    with an added "why", or None if nothing fits/exists.
    """
    hw = hw or detect()
    candidates = group_split_ggufs(files)
    if not candidates:
        return None
    by_quant = {}
    for c in candidates:
        q = quant_of(c["name"])
        if q and (q not in by_quant or c["size"] < by_quant[q]["size"]):
            by_quant[q] = c

    gpu = hw.gpu
    kv_margin = max(n_ctx, 2048) * 96 * 1024  # rough KV+overhead per ctx token
    if gpu:
        # Keep 10% headroom: "barely fits" becomes swap-thrash in practice.
        budget = int((gpu.vram_free - _OVERHEAD - kv_margin) * 0.9)
        for q in _QUANT_PREFERENCE:
            c = by_quant.get(q)
            if c and c["size"] <= budget:
                c["why"] = (f"{q}: largest quant that fully fits "
                            f"{_gb(gpu.vram_free)} free on {gpu.name}")
                return c
    # Nothing fits VRAM, so layers spill to CPU. Bigger quants are NOT better
    # here — CPU token speed scales with bytes read, so prefer the Q4 class
    # (the standard speed/quality tradeoff) and never exceed half of free RAM.
    ram_budget = int(hw.ram_available * 0.5)
    for q in _CPU_PREFERENCE:
        c = by_quant.get(q)
        if c and c["size"] <= ram_budget:
            where = (f"mostly on CPU ({gpu.name} lacks free VRAM)" if gpu else "CPU")
            c["why"] = (f"{q}: best speed/quality for {where}; "
                        f"larger quants would crawl and eat RAM")
            return c
    smallest = min(candidates, key=lambda c: c["size"])
    smallest["why"] = "smallest available quant (machine is tight on memory)"
    return smallest


# --------------------------------------------------------------------------- #
# Context planning (virtual context / elastic engine)
# --------------------------------------------------------------------------- #

# Discrete context "gears". The engine launches in a modest gear and the
# elastic server upshifts under pressure (or on prompt overflow), so VRAM is
# only ever spent on context a workload actually needed.
CONTEXT_GEARS = [4096, 8192, 16384, 32768, 65536, 131072]

# Launch gear preference: big enough for almost every chat / short agentic
# turn, small enough to leave VRAM for layers. Elasticity covers the rest.
_START_GEAR_TARGET = 16384


def kv_bytes_per_token(meta: GGUFMeta, bytes_per_elem: int = 2) -> int:
    """KV cache cost of ONE context token (K+V, all layers), f16 by default."""
    if not (meta.n_layers and meta.n_embd and meta.n_head):
        # order-of-magnitude fallback: ~1 KiB per layer per token
        return (meta.n_layers or _DEFAULT_LAYERS) * 1024
    head_dim = meta.n_embd // meta.n_head
    return meta.n_layers * meta.n_head_kv * head_dim * 2 * bytes_per_elem


@dataclass
class ContextPlan:
    n_ctx: int        # launch gear
    n_ctx_max: int    # elastic ceiling (resize never exceeds this)
    reason: str

    def describe(self) -> str:
        return self.reason


def _gears_for(meta: GGUFMeta, floor: int) -> list[int]:
    ceiling = meta.n_ctx_train or CONTEXT_GEARS[-1]
    gears = [g for g in CONTEXT_GEARS if floor <= g <= ceiling]
    if not gears:
        gears = [min(max(floor, CONTEXT_GEARS[0]), ceiling)]
    if ceiling not in gears and ceiling > gears[-1]:
        gears.append(ceiling)
    return gears


def plan_context(meta: GGUFMeta, hw: Optional[Hardware] = None, *,
                 floor: int = 4096) -> ContextPlan:
    """Choose the launch context gear and the elastic ceiling for this machine.

    Policy:
      - ceiling: the largest gear whose KV still allows FULL layer offload
        (VRAM), or — CPU / nothing fits — the largest gear whose KV fits a
        conservative slice of available RAM. Never beyond the model's trained
        context.
      - launch gear: min(16k-ish target, ceiling) but never a gear that costs
        full offload when a smaller one keeps it. Elasticity upshifts later;
        layers-on-GPU is the thing worth protecting at launch.
    """
    hw = hw or detect()
    gears = _gears_for(meta, floor)
    per_tok = kv_bytes_per_token(meta)
    gpu = hw.gpu
    train_note = (f"trained ctx {meta.n_ctx_train}" if meta.n_ctx_train
                  else "trained ctx unknown")

    if gpu is None:
        ram_slice = int(hw.ram_available * 0.25)
        fitting = [g for g in gears if g * per_tok <= ram_slice] or [gears[0]]
        ceiling = fitting[-1]
        start = min(_START_GEAR_TARGET, ceiling)
        start = max([g for g in gears if g <= start] or [gears[0]])
        return ContextPlan(start, ceiling,
                           f"CPU: KV {_gb(per_tok * start)} at {start} ctx in RAM; "
                           f"elastic up to {ceiling} ({_gb(per_tok * ceiling)} KV; "
                           f"{train_note})")

    # Ceiling: largest gear that still fully offloads. Falls back to the
    # smallest gear (best-effort partial offload) when nothing does.
    full = [g for g in gears if plan(meta, g, hw).fits_fully]
    if full:
        ceiling = full[-1]
        start = min(_START_GEAR_TARGET, ceiling)
        start = max([g for g in gears if g <= start] or [gears[0]])
        return ContextPlan(start, ceiling,
                           f"launch {start} ctx (KV {_gb(per_tok * start)}), elastic up to "
                           f"{ceiling} with full offload on {gpu.name} "
                           f"({_gb(gpu.vram_free)} free; {train_note})")

    start = gears[0]
    p = plan(meta, start, hw)
    return ContextPlan(start, start,
                       f"tight VRAM: {start} ctx is the floor (KV {_gb(per_tok * start)}; "
                       f"{p.n_gpu_layers}/{p.n_layers} layers on {gpu.name}; {train_note})")


def resolve_context(value: "int | str", model_path: str,
                    hw: Optional[Hardware] = None) -> tuple[int, int, str]:
    """Turn an `n_ctx` setting into concrete (n_ctx, n_ctx_max, reason).

    Integers (and integer strings) pass through pinned — the user said a
    number, so the engine launches there and never resizes past it.
    "auto"/"" plans from the GGUF + hardware.
    """
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        return value, value, "explicit setting"
    text = str(value).strip().lower()
    if text not in ("", "auto"):
        try:
            v = int(text)
            return v, v, "explicit setting"
        except ValueError:
            pass
    cp = plan_context(read_gguf_meta(model_path), hw)
    return cp.n_ctx, cp.n_ctx_max, cp.reason

