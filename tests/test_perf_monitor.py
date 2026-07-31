"""pleiades.perf_monitor: live CPU/GPU/RAM sampling for the Hardware tab."""

from pleiades import hardware, perf_monitor


def test_sample_returns_real_sane_values():
    # Real call, not mocked -- psutil + /proc/meminfo are fast (~100ms) and
    # deterministic in shape, and this is exactly the "does it actually work"
    # check that matters more than unit-isolating psutil here.
    s = perf_monitor.sample()
    assert 0.0 <= s.cpu_percent <= 100.0
    assert len(s.cpu_percent_per_core) >= 1
    assert all(0.0 <= c <= 100.0 for c in s.cpu_percent_per_core)
    assert s.ram_total > 0
    assert 0 <= s.ram_used <= s.ram_total


def test_gpu_sample_carries_vram_from_hardware_detect(monkeypatch):
    fake_gpu = hardware.GPU("nvidia", "Fake RTX", vram_total=10 * 1024**3, vram_free=4 * 1024**3)
    monkeypatch.setattr(hardware, "detect",
                        lambda: hardware.Hardware(gpus=[fake_gpu], ram_total=1, ram_available=1))
    monkeypatch.setattr(perf_monitor, "_nvidia_utilization", lambda: {"Fake RTX": 42.0})
    s = perf_monitor.sample()
    assert len(s.gpus) == 1
    g = s.gpus[0]
    assert g.vendor == "nvidia"
    assert g.vram_total == 10 * 1024**3
    assert g.vram_used == 6 * 1024**3  # total - free
    assert g.utilization == 42.0


def test_gpu_utilization_is_none_not_fabricated_when_unavailable(monkeypatch):
    # An Intel card whose driver doesn't expose gpu_busy_percent (verified
    # true on this project's own dev box, see perf_monitor.py's docstring)
    # must report None, not a guessed/default 0 -- a 0% reading looks like
    # real data to a UI, silence does not.
    fake_gpu = hardware.GPU("intel", "Onboard IGD", vram_total=0, vram_free=0)
    monkeypatch.setattr(hardware, "detect",
                        lambda: hardware.Hardware(gpus=[fake_gpu], ram_total=1, ram_available=1))
    monkeypatch.setattr(perf_monitor, "_linux_sysfs_busy_percent", lambda: {})
    s = perf_monitor.sample()
    assert len(s.gpus) == 1
    assert s.gpus[0].utilization is None


def test_nvidia_utilization_parses_csv(monkeypatch):
    out = "NVIDIA GeForce RTX 4090, 87\nNVIDIA GeForce RTX 3080, 12\n"
    monkeypatch.setattr(hardware, "_run", lambda cmd, timeout=5.0: out)
    result = perf_monitor._nvidia_utilization()
    assert result == {"NVIDIA GeForce RTX 4090": 87.0, "NVIDIA GeForce RTX 3080": 12.0}


def test_linux_sysfs_busy_percent_skips_cards_without_the_file(tmp_path):
    drm = tmp_path / "drm"
    card0 = drm / "card0" / "device"
    card0.mkdir(parents=True)
    (card0 / "vendor").write_text("0x1002\n")
    (card0 / "gpu_busy_percent").write_text("33\n")
    card1 = drm / "card1" / "device"
    card1.mkdir(parents=True)
    (card1 / "vendor").write_text("0x8086\n")
    # no gpu_busy_percent file -- this card's driver doesn't expose it

    result = perf_monitor._linux_sysfs_busy_percent(drm_root=drm)
    assert result == {"0x1002": 33.0}
    assert "0x8086" not in result
