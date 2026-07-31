"""launch.py's build_command(): --mmproj wiring (native llama-server only),
plus the PLEIADES_ENGINE=pleiades_native branch's flag-mode CLI + autofit
placement wiring (GPU/MoE offload). --mmproj is still excluded from the
pleiades_native path (that engine has no vision support); GPU/MoE placement is
NOT -- the native C++ engine now shares this function's autofit machinery."""

import os


from pleiades import launch, runtime
from pleiades.autofit import Placement, RuntimeCaps
from pleiades.config import Settings


def _fake_gguf(tmp_path, name="model.gguf"):
    p = tmp_path / name
    p.write_bytes(b"GGUF-fake-not-a-real-model")
    return p


def _force_native(monkeypatch, native_path="/fake/llama-server"):
    """Make build_command() believe the native runtime is available at
    `native_path`, without touching any real binary on disk."""
    monkeypatch.setattr(runtime, "find_native", lambda prefer_moe_opts=False: native_path)
    monkeypatch.setattr(runtime, "caps", lambda prefer_moe_opts=False: RuntimeCaps(native=True, moe_offload=False))
    monkeypatch.setattr(runtime, "native_env", lambda p: {})


def test_mmproj_is_wired_into_the_native_llama_server_command(tmp_path, monkeypatch):
    _force_native(monkeypatch)
    model = _fake_gguf(tmp_path)
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="vlm",
                                mmproj=str(mmproj), settings=Settings())
    assert "--mmproj" in plan.cmd
    idx = plan.cmd.index("--mmproj")
    assert plan.cmd[idx + 1] == str(mmproj)
    assert "vision" in plan.why.lower()


def test_no_mmproj_flag_when_mmproj_is_blank(tmp_path, monkeypatch):
    _force_native(monkeypatch)
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="text-model",
                                settings=Settings())
    assert "--mmproj" not in plan.cmd


def test_no_mmproj_flag_when_the_file_does_not_actually_exist(tmp_path, monkeypatch):
    _force_native(monkeypatch)
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="text-model",
                                mmproj=str(tmp_path / "does-not-exist-mmproj.gguf"),
                                settings=Settings())
    assert "--mmproj" not in plan.cmd


def test_mmproj_is_never_wired_into_the_llama_cpp_python_fallback(tmp_path, monkeypatch):
    # No native runtime available at all -> python-engine branches.
    monkeypatch.setattr(runtime, "find_native", lambda prefer_moe_opts=False: None)
    monkeypatch.setattr(runtime, "caps", lambda prefer_moe_opts=False: RuntimeCaps(native=False))
    monkeypatch.setenv("PLEIADES_ENGINE", "llama_cpp")
    model = _fake_gguf(tmp_path)
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="vlm",
                                mmproj=str(mmproj), settings=Settings())
    assert "--mmproj" not in plan.cmd
    assert "llama_cpp.server" in " ".join(plan.cmd)


def test_mmproj_is_never_wired_into_the_elastic_python_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "find_native", lambda prefer_moe_opts=False: None)
    monkeypatch.setattr(runtime, "caps", lambda prefer_moe_opts=False: RuntimeCaps(native=False))
    monkeypatch.delenv("PLEIADES_ENGINE", raising=False)
    model = _fake_gguf(tmp_path)
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="vlm",
                                mmproj=str(mmproj), settings=Settings())
    assert "--mmproj" not in plan.cmd
    assert "pleiades.inference.server" in " ".join(plan.cmd)


def test_mmproj_is_wired_into_the_pleiades_native_cpp_engine(tmp_path, monkeypatch):
    # Phase 9.4.2 (docs/specs/2026-07-21-native-inference-engine-design.md):
    # the engine gained real vision support (libmtmd) this pass -- this test
    # used to pin the OPPOSITE behavior (mmproj silently dropped on this
    # branch) as a regression guard for a since-closed limitation. Updated
    # rather than deleted so a future regression here is still caught.
    from pleiades import runtime as _rt
    monkeypatch.setenv("PLEIADES_ENGINE", "pleiades_native")
    monkeypatch.setattr(_rt, "find_native_cpp_engine", lambda: "/fake/pleiades-engine-server")
    model = _fake_gguf(tmp_path)
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="vlm",
                                mmproj=str(mmproj), settings=Settings())
    assert "--mmproj" in plan.cmd
    assert plan.cmd[plan.cmd.index("--mmproj") + 1] == str(mmproj)
    assert "+mmproj (vision)" in plan.why
    assert "pleiades-native-engine" in plan.why


def test_native_command_sets_separate_decode_and_batch_thread_counts(tmp_path, monkeypatch):
    # 2026-07-24: decode (memory-bandwidth-bound) and prefill/batch
    # (compute-bound) want different thread counts -- real measurement on
    # an 8-physical-core box showed decode peaking at half that. -tb should
    # stay at the full physical-core estimate, -t at half of it.
    _force_native(monkeypatch)
    monkeypatch.setattr(launch.os, "cpu_count", lambda: 16)  # 8 physical (SMT=2)
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="text-model",
                                settings=Settings())
    assert "-t" in plan.cmd and "-tb" in plan.cmd
    t = int(plan.cmd[plan.cmd.index("-t") + 1])
    tb = int(plan.cmd[plan.cmd.index("-tb") + 1])
    assert tb == 8
    assert t == 4
    assert t < tb


def test_native_command_decode_threads_never_below_two(tmp_path, monkeypatch):
    _force_native(monkeypatch)
    monkeypatch.setattr(launch.os, "cpu_count", lambda: 2)  # tiny box: 2 logical
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="text-model",
                                settings=Settings())
    t = int(plan.cmd[plan.cmd.index("-t") + 1])
    tb = int(plan.cmd[plan.cmd.index("-tb") + 1])
    assert tb == 4  # existing floor
    assert t == 2   # new floor


def test_native_command_includes_slot_save_path(tmp_path, monkeypatch):
    _force_native(monkeypatch)
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="slottest",
                                settings=Settings())
    assert "--slot-save-path" in plan.cmd
    idx = plan.cmd.index("--slot-save-path")
    slot_path = plan.cmd[idx + 1]
    assert "slottest" in slot_path
    assert slot_path.endswith(os.sep)  # required: server does raw string concat, not path join
    assert os.path.isdir(slot_path.rstrip(os.sep))


# -- PLEIADES_ENGINE=pleiades_native: flag-mode CLI + autofit placement ----- #

def _force_native_cpp(monkeypatch, bin_path="/fake/pleiades-engine-server"):
    """Route build_command() into the native C++ engine branch."""
    from pleiades import runtime as _rt
    monkeypatch.setenv("PLEIADES_ENGINE", "pleiades_native")
    monkeypatch.setattr(_rt, "find_native_cpp_engine", lambda: bin_path)


def test_pleiades_native_uses_flag_mode_not_legacy_positional(tmp_path, monkeypatch):
    # The branch used to emit 6-positional-arg legacy mode with ngl hardcoded
    # to 0. It must now emit http_server.cpp's flag CLI instead.
    _force_native_cpp(monkeypatch)
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8091, name="nat",
                                n_gpu_layers="auto", settings=Settings())
    assert plan.cmd[1].startswith("--")  # not the legacy `<bin> <model> <host> ...`
    assert plan.cmd[plan.cmd.index("--model") + 1] == str(model)
    for flag in ("--host", "--port", "--ctx", "--ngl", "--alias"):
        assert flag in plan.cmd


def test_pleiades_native_all_gpu_layers_is_negative_ngl(tmp_path, monkeypatch):
    # place() emits n_gpu_layers == -1 for full-GPU/MoE strategies. This engine
    # takes "all layers" as a NEGATIVE ngl (ModelManager::load -> llama), NOT
    # the 999 sentinel the llama-server branch uses -- it must pass -1 straight
    # through and never emit 999.
    _force_native_cpp(monkeypatch)
    monkeypatch.setattr(launch, "place",
                        lambda meta, ctx, caps=None: Placement("full_gpu", n_gpu_layers=-1, est_tps=100.0))
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8091, name="nat",
                                n_gpu_layers="auto", settings=Settings())
    assert plan.cmd[plan.cmd.index("--ngl") + 1] == "-1"
    assert "999" not in plan.cmd


def test_pleiades_native_moe_placement_emits_n_cpu_moe(tmp_path, monkeypatch):
    # A MoE split (n_cpu_moe > 0, from the model's own expert structure) must
    # reach the engine as --n-cpu-moe.
    _force_native_cpp(monkeypatch)
    monkeypatch.setattr(launch, "place",
                        lambda meta, ctx, caps=None: Placement("moe_cpu", n_gpu_layers=-1, n_cpu_moe=48, est_tps=20.0))
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8091, name="moe",
                                n_gpu_layers="auto", settings=Settings())
    assert plan.cmd[plan.cmd.index("--n-cpu-moe") + 1] == "48"
    assert "n_cpu_moe=48" in plan.why


def test_pleiades_native_explicit_ngl_override_wins_and_skips_moe(tmp_path, monkeypatch):
    # An explicit n_gpu_layers overrides autofit and suppresses the auto MoE
    # split (matching the sibling native branch's `forced is None` gating).
    _force_native_cpp(monkeypatch)
    monkeypatch.setattr(launch, "place",
                        lambda meta, ctx, caps=None: Placement("moe_cpu", n_gpu_layers=-1, n_cpu_moe=48, est_tps=20.0))
    model = _fake_gguf(tmp_path)
    plan = launch.build_command(str(model), "127.0.0.1", 8091, name="nat",
                                n_gpu_layers=10, settings=Settings())
    assert plan.cmd[plan.cmd.index("--ngl") + 1] == "10"
    assert "--n-cpu-moe" not in plan.cmd
    assert "override" in plan.why


def test_pleiades_native_moe_offload_caps_enabled(tmp_path, monkeypatch):
    # The branch must pass place() a RuntimeCaps with moe_offload=True (the
    # engine supports --n-cpu-moe), NOT probe the possibly-absent llama-server
    # binary -- otherwise a MoE model would never get a split. Capture the caps
    # place() is actually called with.
    _force_native_cpp(monkeypatch)
    seen = {}

    def _capture_place(meta, ctx, caps=None):
        seen["caps"] = caps
        return Placement("full_gpu", n_gpu_layers=-1, est_tps=100.0)

    monkeypatch.setattr(launch, "place", _capture_place)
    model = _fake_gguf(tmp_path)
    launch.build_command(str(model), "127.0.0.1", 8091, name="nat",
                         n_gpu_layers="auto", settings=Settings())
    assert seen["caps"].moe_offload is True
