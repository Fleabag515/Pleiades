"""fetch_model()'s mmproj sidecar handling: the foundry's HF download
pipeline used to just never fetch mmproj-*.gguf files at all (hardware.
is_mmproj() only ever excluded them from the model-candidate pool) -- these
tests confirm it now downloads a companion mmproj file when the repo has
one, and registers it on the Model entry so ModelManager.add()'s own
sibling-file auto-detection (already covered in tests/test_hardware.py /
test_models.py) has something real to find.

No real network calls: list_repo_ggufs() and _download() are monkeypatched
so this exercises fetch_model()'s own selection/wiring logic only.
"""

from pleiades import fetch
from pleiades.models import ModelManager


def _install_fakes(monkeypatch, tmp_path, files, downloaded):
    def fake_list_repo_ggufs(repo, *, client=None):
        return files

    def fake_download(repo, filename, dest, progress=None, *, client=None):
        downloaded.append(filename)
        dest.write_bytes(b"fake-bytes-for-" + filename.encode())

    monkeypatch.setattr(fetch, "list_repo_ggufs", fake_list_repo_ggufs)
    monkeypatch.setattr(fetch, "_download", fake_download)
    monkeypatch.setattr(fetch, "models_dir", lambda: tmp_path)


def test_fetch_model_downloads_and_registers_the_mmproj_sidecar(tmp_path, monkeypatch):
    files = [
        {"name": "SomeVLM-Q4_K_M.gguf", "size": 900_000_000},
        {"name": "mmproj-SomeVLM-f16.gguf", "size": 100_000_000},
    ]
    downloaded = []
    _install_fakes(monkeypatch, tmp_path, files, downloaded)

    entry = fetch.fetch_model("org/SomeVLM-GGUF", name="somevlm", quant="Q4_K_M")

    assert "mmproj-SomeVLM-f16.gguf" in downloaded
    assert entry["mmproj"]
    assert entry["mmproj"].endswith("mmproj-SomeVLM-f16.gguf")

    ModelManager().remove("somevlm", delete_file=False)


def test_fetch_model_with_no_mmproj_in_repo_leaves_it_blank(tmp_path, monkeypatch):
    files = [{"name": "TextModel-Q4_K_M.gguf", "size": 900_000_000}]
    downloaded = []
    _install_fakes(monkeypatch, tmp_path, files, downloaded)

    entry = fetch.fetch_model("org/TextModel-GGUF", name="textmodel", quant="Q4_K_M")

    assert not any("mmproj" in d.lower() for d in downloaded)
    assert entry["mmproj"] == ""

    ModelManager().remove("textmodel", delete_file=False)


def test_fetch_model_picks_the_largest_mmproj_when_multiple_offered(tmp_path, monkeypatch):
    files = [
        {"name": "SomeVLM2-Q4_K_M.gguf", "size": 900_000_000},
        {"name": "mmproj-SomeVLM2-Q8_0.gguf", "size": 50_000_000},
        {"name": "mmproj-SomeVLM2-f16.gguf", "size": 100_000_000},
    ]
    downloaded = []
    _install_fakes(monkeypatch, tmp_path, files, downloaded)

    entry = fetch.fetch_model("org/SomeVLM2-GGUF", name="somevlm2", quant="Q4_K_M")

    assert "mmproj-SomeVLM2-f16.gguf" in downloaded
    assert "mmproj-SomeVLM2-Q8_0.gguf" not in downloaded
    assert entry["mmproj"].endswith("mmproj-SomeVLM2-f16.gguf")

    ModelManager().remove("somevlm2", delete_file=False)


def test_fetch_model_already_downloaded_mmproj_is_not_refetched(tmp_path, monkeypatch):
    files = [
        {"name": "SomeVLM3-Q4_K_M.gguf", "size": 900_000_000},
        {"name": "mmproj-SomeVLM3-f16.gguf", "size": 100_000_000},
    ]
    downloaded = []
    _install_fakes(monkeypatch, tmp_path, files, downloaded)

    dest_dir = tmp_path / "org__SomeVLM3-GGUF"
    dest_dir.mkdir(parents=True)
    (dest_dir / "mmproj-SomeVLM3-f16.gguf").write_bytes(b"already-here")

    entry = fetch.fetch_model("org/SomeVLM3-GGUF", name="somevlm3", quant="Q4_K_M")

    assert "mmproj-SomeVLM3-f16.gguf" not in downloaded  # not re-downloaded
    assert entry["mmproj"].endswith("mmproj-SomeVLM3-f16.gguf")

    ModelManager().remove("somevlm3", delete_file=False)
