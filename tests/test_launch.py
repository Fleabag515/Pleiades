"""launch.py's build_command(): --mmproj wiring, scoped to the native
llama-server runtime only (see pleiades/launch.py's build_command docstring
for why the python fallback and PLEIADES_ENGINE=pleiades_native are
deliberately excluded)."""

from pleiades import launch, runtime
from pleiades.autofit import RuntimeCaps
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


def test_mmproj_is_never_wired_into_the_pleiades_native_cpp_engine(tmp_path, monkeypatch):
    from pleiades import runtime as _rt
    monkeypatch.setenv("PLEIADES_ENGINE", "pleiades_native")
    monkeypatch.setattr(_rt, "find_native_cpp_engine", lambda: "/fake/pleiades-engine-server")
    model = _fake_gguf(tmp_path)
    mmproj = tmp_path / "mmproj-model-f16.gguf"
    mmproj.write_bytes(b"fake-mmproj")
    plan = launch.build_command(str(model), "127.0.0.1", 8090, name="vlm",
                                mmproj=str(mmproj), settings=Settings())
    assert "--mmproj" not in " ".join(plan.cmd)
    assert "pleiades-native-engine" in plan.why
