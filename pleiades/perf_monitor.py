"""Live CPU/GPU/RAM utilization sampling for the desktop app's Hardware tab --
a Task-Manager/Mission-Center style live view, not a one-shot snapshot.

Deliberately separate from hardware.py's detect(): that function is a
capacity/placement-planning snapshot ("cheap enough to call per launch", per
its own docstring), not built for repeated ~1Hz polling. This module IS --
sample() is meant to be called on every tick of a live view and stays cheap
each time.

GPU utilization (compute busy%, as opposed to VRAM capacity, which
hardware.py already handles) is genuinely not available everywhere -- it's
driver/vendor/OS-dependent in a way VRAM capacity isn't. Where it can't
honestly be read, GPUSample.utilization is None, not a guess or a fabricated
0. Verified directly on this session's dev box (Intel UHD 620, Kaby Lake,
i915 driver): no gpu_busy_percent sysfs file exists for this generation (that
interface is newer-Xe-driver-only), and the alternative (the i915 perf PMU,
which intel_gpu_top itself uses) requires either root or a lowered
kernel.perf_event_paranoid -- neither of which this module does automatically
on a user's machine. NVIDIA (nvidia-smi) and AMD (the amdgpu driver's
gpu_busy_percent sysfs file, unlike Intel's inconsistent one, is a reliable,
long-standing interface) both work without elevated privileges. Windows GPU
Engine utilization (available non-admin via PDH performance counters) and
Apple Silicon (typically needs `powermetrics`, which needs sudo) are not
implemented yet -- utilization is None there too, honestly, rather than
guessed at without being able to test either platform from this box.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import hardware


@dataclass
class GPUSample:
    vendor: str
    name: str
    vram_total: int          # bytes
    vram_used: int           # bytes
    utilization: Optional[float]  # 0-100, None if unavailable (see module docstring)


@dataclass
class PerfSample:
    cpu_percent: float
    cpu_percent_per_core: list[float] = field(default_factory=list)
    ram_total: int = 0       # bytes
    ram_used: int = 0        # bytes
    gpus: list[GPUSample] = field(default_factory=list)


def _nvidia_utilization() -> dict[str, float]:
    """GPU name -> utilization%, via nvidia-smi. Matched against hardware.py's
    already-detected GPU list by name (nvidia-smi's own --query-gpu=name
    output is the same string hardware.py's _detect_nvidia() already uses)."""
    out = hardware._run(["nvidia-smi", "--query-gpu=name,utilization.gpu",
                         "--format=csv,noheader,nounits"])
    result: dict[str, float] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                result[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return result


def _linux_sysfs_busy_percent(drm_root: Path = Path("/sys/class/drm")) -> dict[str, float]:
    """PCI vendor id ("0x1002" AMD / "0x8086" Intel) -> utilization%, for
    whichever cards on this box actually expose gpu_busy_percent. Reliable
    and long-standing for AMD's amdgpu driver; genuinely absent on some
    Intel generations/drivers (see module docstring) -- a missing file is
    treated as "this card doesn't expose it", not an error. Keyed by vendor
    id rather than card path since hardware.py's own GPU list doesn't carry
    a card path either -- fine as long as a box has at most one non-NVIDIA
    GPU, which is the common case this targets. `drm_root` is injectable for
    tests; real callers always use the default."""
    result: dict[str, float] = {}
    for card in sorted(drm_root.glob("card[0-9]*")):
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip().lower()
            busy = (dev / "gpu_busy_percent").read_text().strip()
        except OSError:
            continue
        try:
            result[vendor] = float(busy)
        except ValueError:
            continue
    return result


_VENDOR_ID = {"amd": "0x1002", "intel": "0x8086"}


def sample() -> PerfSample:
    """One live sample. Blocks ~100ms (psutil's own measurement window for a
    real, non-first-call-garbage CPU percentage) -- fine for ~1Hz polling,
    not meant to be called in a tight loop."""
    import psutil

    per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    cpu_percent = sum(per_core) / len(per_core) if per_core else 0.0

    vm = psutil.virtual_memory()

    hw = hardware.detect()
    nvidia_util = _nvidia_utilization() if any(g.vendor == "nvidia" for g in hw.gpus) else {}
    sysfs_util = (_linux_sysfs_busy_percent()
                  if sys.platform == "linux" and any(g.vendor in ("amd", "intel") for g in hw.gpus)
                  else {})

    gpus = []
    for g in hw.gpus:
        util: Optional[float] = None
        if g.vendor == "nvidia":
            util = nvidia_util.get(g.name)
        elif g.vendor in _VENDOR_ID:
            util = sysfs_util.get(_VENDOR_ID[g.vendor])
        gpus.append(GPUSample(g.vendor, g.name, g.vram_total,
                              max(g.vram_total - g.vram_free, 0), util))

    return PerfSample(cpu_percent=cpu_percent, cpu_percent_per_core=per_core,
                      ram_total=vm.total, ram_used=vm.total - vm.available, gpus=gpus)
