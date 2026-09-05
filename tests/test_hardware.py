"""pleiades.hardware: GGUF parsing, placement planning, quant selection."""

import struct
import sys

import pytest

from pleiades import hardware
from pleiades.hardware import (GPU, GGUFMeta, Hardware, group_split_ggufs,
                               pick_quant, plan, quant_of, resolve_layers)

GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# synthetic GGUF
# --------------------------------------------------------------------------- #
def _kv_str(key: str, val: str) -> bytes:
    k, v = key.encode(), val.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 8) + \
        struct.pack("<Q", len(v)) + v


def _kv_u32(key: str, val: int) -> bytes:
    k = key.encode()
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 4) + struct.pack("<I", val)


def _kv_arr_u32(key: str, values: list) -> bytes:
    """A GGUF array-of-u32 value (vtype=9/_ARRAY, etype=4/u32) — used to test
    the {arch}.attention.recurrent_layers array-count path in _read_scalar's
    want_array_len branch."""
    k = key.encode()
    body = struct.pack("<I", 4) + struct.pack("<Q", len(values))
    for v in values:
        body += struct.pack("<I", v)
    return struct.pack("<Q", len(k)) + k + struct.pack("<I", 9) + body


def make_gguf(path, n_layers=32, n_embd=4096, n_head=32, n_head_kv=8,
              pad_to=1024, architecture="llama", extra_kvs: bytes = b"",
              n_extra_kvs: int = 0) -> str:
    kvs = (_kv_str("general.architecture", architecture)
           + _kv_u32(f"{architecture}.block_count", n_layers)
           + _kv_u32(f"{architecture}.embedding_length", n_embd)
           + _kv_u32(f"{architecture}.attention.head_count", n_head)
           + _kv_u32(f"{architecture}.attention.head_count_kv", n_head_kv)
           + _kv_u32("general.file_type", 15)
           + extra_kvs)
    blob = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + \
        struct.pack("<Q", 6 + n_extra_kvs) + kvs
    blob += b"\0" * max(0, pad_to - len(blob))
    path.write_bytes(blob)
    return str(path)


def test_gguf_parse(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", n_layers=48, n_embd=8192, n_head=64,
                  n_head_kv=8)
    meta = hardware.read_gguf_meta(p)
    assert meta.n_layers == 48
    assert meta.n_embd == 8192
    assert meta.n_head_kv == 8
    assert meta.quant == "Q4_K_M"
    assert not meta.estimated


def test_gguf_parse_garbage_is_safe(tmp_path):
    p = tmp_path / "junk.gguf"
    p.write_bytes(b"not a gguf at all")
    meta = hardware.read_gguf_meta(p)
    assert meta.estimated and meta.file_size > 0
    meta2 = hardware.read_gguf_meta(tmp_path / "missing.gguf")
    assert meta2.file_size == 0


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #
def _hw(vram_gb=0, ram_gb=32):
    gpus = [GPU("nvidia", "Fake RTX", int(vram_gb * GiB), int(vram_gb * GiB))] \
        if vram_gb else []
    return Hardware(gpus=gpus, ram_total=int(ram_gb * GiB),
                    ram_available=int(ram_gb * GiB), cpu_threads=16)


def _meta(size_gb, n_layers=32):
    return GGUFMeta(path="x", file_size=int(size_gb * GiB), n_layers=n_layers,
                    n_embd=4096, n_head=32, n_head_kv=8)


def test_plan_full_offload_when_it_fits():
    p = plan(_meta(4), n_ctx=8192, hw=_hw(vram_gb=24))
    assert p.n_gpu_layers == -1 and p.fits_fully


def test_plan_partial_offload_when_tight():
    p = plan(_meta(20, n_layers=40), n_ctx=8192, hw=_hw(vram_gb=8))
    assert 0 < p.n_gpu_layers < 40
    assert "partial" in p.reason


def test_plan_cpu_when_no_gpu():
    p = plan(_meta(4), n_ctx=8192, hw=_hw(vram_gb=0))
    assert p.n_gpu_layers == 0 and "no GPU" in p.reason


def test_plan_cpu_when_vram_tiny():
    p = plan(_meta(30, n_layers=40), n_ctx=8192, hw=_hw(vram_gb=1))
    assert p.n_gpu_layers == 0


def test_resolve_layers_passthrough_and_auto(tmp_path):
    assert resolve_layers(-1, "x", 8192)[0] == -1
    assert resolve_layers("7", "x", 8192)[0] == 7
    p = make_gguf(tmp_path / "m.gguf")
    layers, why = resolve_layers("auto", p, 8192, hw=_hw(vram_gb=24))
    assert layers == -1 and why


# --------------------------------------------------------------------------- #
# quant selection
# --------------------------------------------------------------------------- #
def test_quant_of():
    assert quant_of("model-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("model.q8_0.gguf") == "Q8_0"
    assert quant_of("model-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_of("model.gguf") == ""


def test_group_split_ggufs():
    files = [{"name": "big-Q4_K_M-00001-of-00002.gguf", "size": 10},
             {"name": "big-Q4_K_M-00002-of-00002.gguf", "size": 5},
             {"name": "small-Q2_K.gguf", "size": 3},
             {"name": "README.md", "size": 1}]
    out = group_split_ggufs(files)
    by_name = {g["name"]: g for g in out}
    assert by_name["big-Q4_K_M.gguf"]["size"] == 15
    assert len(by_name["big-Q4_K_M.gguf"]["parts"]) == 2
    assert by_name["small-Q2_K.gguf"]["parts"] == ["small-Q2_K.gguf"]


def _files(**sizes_gb):
    return [{"name": f"m-{q}.gguf", "size": int(s * GiB)}
            for q, s in sizes_gb.items()]


def test_pick_quant_prefers_largest_that_fits_vram():
    files = _files(Q8_0=8, Q6_K=6, Q4_K_M=4, Q2_K=2)
    # 8 GiB free: Q8_0 (8 GiB) can't fit with KV + margin, Q6_K (6 GiB) can.
    c = pick_quant(files, _hw(vram_gb=8), n_ctx=4096)
    assert quant_of(c["name"]) == "Q6_K"


def test_pick_quant_falls_back_to_ram():
    # CPU path is capped at 50% of available RAM: Q4_K_M (20 GiB) needs 48 GiB free.
    files = _files(Q8_0=30, Q4_K_M=20, Q2_K=10)
    c = pick_quant(files, _hw(vram_gb=8, ram_gb=48), n_ctx=4096)
    assert quant_of(c["name"]) == "Q4_K_M"
    c2 = pick_quant(files, _hw(vram_gb=8, ram_gb=28), n_ctx=4096)
    assert quant_of(c2["name"]) == "Q2_K"


def test_pick_quant_smallest_when_everything_is_too_big():
    files = _files(Q8_0=300, Q2_K=200)
    c = pick_quant(files, _hw(vram_gb=8, ram_gb=16), n_ctx=4096)
    assert quant_of(c["name"]) == "Q2_K"
    assert "smallest" in c["why"]


def test_pick_quant_cpu_fallback_prefers_q4_not_q8():
    """A big Q8 must not be chosen when it would run mostly on CPU."""
    files = _files(Q8_0=8, Q6_K=6, Q4_K_M=4, Q2_K=2)
    c = pick_quant(files, _hw(vram_gb=2, ram_gb=32), n_ctx=4096)
    assert quant_of(c["name"]) == "Q4_K_M"
    assert "speed" in c["why"]


# --------------------------------------------------------------------------- #
# Windows AMD registry detection (parser is pure; PS invocation is shelled out)
# --------------------------------------------------------------------------- #
def test_parse_win_amd_array():
    out = ('[{"name":"AMD Radeon RX 9070 XT","vram":17163091968},'
           '{"name":"AMD Radeon(TM) Graphics","vram":536870912}]')
    gpus = hardware._parse_win_amd(out)
    assert [g.vendor for g in gpus] == ["amd", "amd"]
    assert gpus[0].vram_total == 17163091968
    assert gpus[0].vram_free == gpus[0].vram_total  # free unknown -> total


def test_parse_win_amd_single_object():
    # ConvertTo-Json collapses single-element arrays into a bare object.
    gpus = hardware._parse_win_amd('{"name":"AMD Radeon RX 9070 XT","vram":17163091968}')
    assert len(gpus) == 1 and gpus[0].name == "AMD Radeon RX 9070 XT"


def test_parse_win_amd_dedupes_per_config_keys():
    out = ('[{"name":"AMD Radeon RX 9070 XT","vram":17163091968},'
           '{"name":"AMD Radeon RX 9070 XT","vram":17163091968}]')
    assert len(hardware._parse_win_amd(out)) == 1


def test_parse_win_amd_garbage_and_zero_vram():
    assert hardware._parse_win_amd("") == []
    assert hardware._parse_win_amd("Oops! not json") == []
    assert hardware._parse_win_amd('[{"name":"x","vram":0}]') == []
    assert hardware._parse_win_amd('[{"vram":"NaNsense"}]') == []


def test_parse_win_intel_array():
    out = ('[{"name":"Intel Arc B580","vram":12884901888},'
           '{"name":"Intel(R) Iris(R) Xe Graphics","vram":2147483648}]')
    gpus = hardware._parse_win_intel(out)
    assert [g.vendor for g in gpus] == ["intel", "intel"]
    assert gpus[0].vram_total == 12884901888
    assert gpus[0].vram_free == gpus[0].vram_total


def test_parse_win_intel_garbage_and_zero_vram():
    assert hardware._parse_win_intel("") == []
    assert hardware._parse_win_intel("Oops! not json") == []
    assert hardware._parse_win_intel('[{"name":"x","vram":0}]') == []


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="_detect_intel() reads Linux sysfs; the Windows runner has no /sys")
def test_detect_intel_linux_sysfs(tmp_path, monkeypatch):
    # Fake /sys/class/drm/card0/device/{vendor,mem_info_vram_total,label}
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    (card / "mem_info_vram_total").write_text(str(16 * GiB) + "\n")
    (card / "mem_info_vram_used").write_text(str(2 * GiB) + "\n")
    (card / "label").write_text("Intel Arc A770\n")
    monkeypatch.setattr(hardware.Path, "glob", lambda self, pat: (
        [card.parent] if str(self) == "/sys/class/drm" and pat == "card[0-9]*" else []))
    gpus = hardware._detect_intel()
    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].name == "Intel Arc A770"
    assert gpus[0].vram_total == 16 * GiB
    assert gpus[0].vram_free == 14 * GiB


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="_detect_intel() reads Linux sysfs; the Windows runner has no /sys")
def test_detect_intel_linux_unreadable_vram_still_registers_gpu(tmp_path, monkeypatch):
    # No mem_info_vram_total/vram/total sysfs file at all (the real-world
    # case on most i915/Xe kernels today) -- must still report the GPU
    # (so backend selection sees it) with vram_total=0, not vanish entirely.
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "vendor").write_text("0x8086\n")
    monkeypatch.setattr(hardware.Path, "glob", lambda self, pat: (
        [card.parent] if str(self) == "/sys/class/drm" and pat == "card[0-9]*" else []))
    gpus = hardware._detect_intel()
    assert len(gpus) == 1
    assert gpus[0].vendor == "intel"
    assert gpus[0].vram_total == 0


# --------------------------------------------------------------------------- #
# mmproj sidecars must never be model candidates
# --------------------------------------------------------------------------- #
def test_mmproj_excluded_from_grouping():
    files = [
        {"name": "mmproj-Some-Model-f16.gguf", "size": 9 * 10**8},
        {"name": "Some-Model-Q4_K_M.gguf", "size": 20 * 10**9},
        {"name": "sub/dir/mmproj-other-F32.gguf", "size": 8 * 10**8},
    ]
    grouped = group_split_ggufs(files)
    assert [g["name"] for g in grouped] == ["Some-Model-Q4_K_M.gguf"]


def test_mmproj_only_repo_yields_nothing():
    files = [{"name": "mmproj-Some-Model-f16.gguf", "size": 9 * 10**8}]
    assert group_split_ggufs(files) == []
    assert pick_quant(files, _hw(vram_gb=16), n_ctx=4096) is None


def test_is_mmproj():
    assert hardware.is_mmproj("mmproj-x-f16.gguf")
    assert hardware.is_mmproj("repo/sub/MMPROJ-x.gguf")
    assert not hardware.is_mmproj("model-with-mmproj-in-middle.gguf")
    assert not hardware.is_mmproj("Some-Model-Q4_K_M.gguf")


def test_find_mmproj_sibling_finds_a_match_in_the_same_directory(tmp_path):
    model = tmp_path / "some-model-Q4_K_M.gguf"
    model.write_bytes(b"fake-model")
    mmproj = tmp_path / "mmproj-some-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    assert hardware.find_mmproj_sibling(str(model)) == str(mmproj)


def test_find_mmproj_sibling_returns_empty_when_none_present(tmp_path):
    model = tmp_path / "text-only-model.gguf"
    model.write_bytes(b"fake-model")
    assert hardware.find_mmproj_sibling(str(model)) == ""


def test_find_mmproj_sibling_picks_the_largest_when_multiple_present(tmp_path):
    model = tmp_path / "some-model.gguf"
    model.write_bytes(b"fake-model")
    small = tmp_path / "mmproj-some-model-Q8_0.gguf"
    small.write_bytes(b"x" * 100)
    big = tmp_path / "mmproj-some-model-f16.gguf"
    big.write_bytes(b"x" * 500)
    assert hardware.find_mmproj_sibling(str(model)) == str(big)


def test_find_mmproj_sibling_nonexistent_model_path():
    assert hardware.find_mmproj_sibling("/no/such/dir/model.gguf") == ""



# --------------------------------------------------------------------------- #
# hybrid-arch KV accounting (2026-07-24 fix: qwen35moe/qwen35/qwen3next were
# missing from _HYBRID_ARCHS entirely, so kv_bytes_per_token() charged full
# dense KV for Ornith's 40 layers instead of just its ~10 real attention
# layers -- see pleiades-project memory for the live-verified production
# impact). Verified directly against llama.cpp's own llm_arch_is_hybrid()
# and the qwen35{,moe}/qwen3next model-build sources (src/models/*.cpp).
# --------------------------------------------------------------------------- #
def test_qwen35moe_recognized_as_hybrid_not_dense():
    p_meta = GGUFMeta(path="x", file_size=1, architecture="qwen35moe")
    assert p_meta.memory_arch == "hybrid"


def test_new_real_hybrid_archs_all_recognized():
    # Every one of these is a real llm_arch_is_hybrid() entry in the pinned
    # production llama.cpp (verified 2026-07-24) -- including the two
    # non-underscore spellings straight from LLM_ARCH_NAMES.
    for arch in ("qwen35", "qwen35moe", "falcon-h1", "plamo2", "granitehybrid",
                 "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe", "kimi-linear"):
        assert GGUFMeta(path="x", file_size=1, architecture=arch).memory_arch == "hybrid", arch


def test_qwen35moe_falls_back_to_full_attention_interval_default(tmp_path):
    # No {arch}.attention.recurrent_layers, no {arch}.full_attention_interval
    # in the GGUF (Ornith's real-world case) -- must fall back to the
    # llama.cpp-internal default interval of 4: every 4th layer (1-indexed)
    # is full attention, the rest recurrent. 40 layers -> 10 attention / 30
    # recurrent, matching the real production model exactly.
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="qwen35moe")
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "hybrid"
    assert meta.recurrent_layer_count == 30


def test_qwen35moe_honors_explicit_full_attention_interval(tmp_path):
    extra = _kv_u32("qwen35moe.full_attention_interval", 8)
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="qwen35moe", extra_kvs=extra, n_extra_kvs=1)
    meta = hardware.read_gguf_meta(p)
    # interval=8 -> full-attn layers at i=7,15,23,31,39 -> 5 attention / 35 recurrent
    assert meta.recurrent_layer_count == 35


def test_qwen35moe_honors_explicit_recurrent_layers_array(tmp_path):
    # When the GGUF DOES publish the explicit array (llama.cpp reads it via
    # get_key_or_arr before ever consulting full_attention_interval), its
    # length wins outright over the family-default interval computation.
    extra = _kv_arr_u32("qwen35moe.attention.recurrent_layers", [0, 1, 2, 4, 5])
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="qwen35moe", extra_kvs=extra, n_extra_kvs=1)
    meta = hardware.read_gguf_meta(p)
    assert meta.recurrent_layer_count == 5


def test_hybrid_arch_without_known_pattern_stays_conservative(tmp_path):
    # jamba is a real hybrid arch but NOT in the qwen35-family interval-
    # fallback set, and this synthetic GGUF publishes neither metadata key
    # -- recurrent_layer_count must stay 0 (unknown), and kv_bytes_per_token
    # must keep charging the full conservative all-layers estimate rather
    # than guessing. This is the safety net the fix must not remove.
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="jamba")
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "hybrid"
    assert meta.recurrent_layer_count == 0
    dense_equivalent = GGUFMeta(path="x", file_size=1, architecture="llama",
                                n_layers=40, n_embd=2048, n_head=16, n_head_kv=2)
    assert hardware.kv_bytes_per_token(meta) == hardware.kv_bytes_per_token(dense_equivalent)


def test_kv_bytes_per_token_scales_down_for_known_hybrid_split(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="qwen35moe")
    meta = hardware.read_gguf_meta(p)
    dense_equivalent = GGUFMeta(path="x", file_size=1, architecture="llama",
                                n_layers=40, n_embd=2048, n_head=16, n_head_kv=2)
    hybrid_kv = hardware.kv_bytes_per_token(meta)
    dense_kv = hardware.kv_bytes_per_token(dense_equivalent)
    # 10/40 attention layers -> exactly 4x less KV per token than treating
    # all 40 as attention.
    assert hybrid_kv == dense_kv // 4


def test_kv_bytes_per_token_respects_bytes_per_elem_for_quantized_kv_cache(tmp_path):
    p = make_gguf(tmp_path / "m.gguf", n_layers=40, n_embd=2048, n_head=16,
                  n_head_kv=2, architecture="qwen35moe")
    meta = hardware.read_gguf_meta(p)
    assert hardware.kv_bytes_per_token(meta, bytes_per_elem=1) == \
        hardware.kv_bytes_per_token(meta, bytes_per_elem=2) // 2


def test_dense_arch_kv_math_completely_unaffected(tmp_path):
    # Regression guard: ordinary dense models must produce byte-identical
    # kv_bytes_per_token output before and after this fix.
    p = make_gguf(tmp_path / "m.gguf", n_layers=32, n_embd=4096, n_head=32, n_head_kv=8)
    meta = hardware.read_gguf_meta(p)
    assert meta.memory_arch == "dense"
    assert meta.recurrent_layer_count == 0
    assert hardware.kv_bytes_per_token(meta) == 32 * 8 * (4096 // 32) * 2 * 2
