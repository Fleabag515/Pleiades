# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the bundled Pleiades backend (--onedir).

Reproducible build:

    ~/Pleiades/.venv/bin/python3.12 -m PyInstaller \
        --noconfirm --clean \
        desktop/backend-build/backend.spec

Run from the repo root (paths below are relative to it). Output lands in
desktop/backend-build/dist/pleiades-backend/ (folder + launcher binary,
NOT onefile — see desktop/backend-build/README.md for why).
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(SPEC)), '..', '..'))

datas = []
binaries = []
hiddenimports = []

# llama_cpp ships its native .so libs (libllama, libggml*, libmtmd) under
# llama_cpp/lib/ and loads them via ctypes at runtime, not via normal Python
# import machinery, so PyInstaller's static analysis can't discover them on
# its own -- collect_all() walks the package directory and grabs everything.
for pkg in ("llama_cpp", "uvicorn", "fastapi", "pydantic", "pydantic_core", "starlette", "anyio", "httpx", "httpcore",
            "camoufox", "browserforge", "apify_fingerprint_datapoints", "playwright", "language_tags"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as e:
        print(f"[backend.spec] collect_all({pkg!r}) failed: {e}")

# uvicorn's protocol/loop/lifespan implementations are picked dynamically by
# string name (config.py does importlib.import_module on them), so static
# analysis misses them without an explicit collect_submodules.
hiddenimports += collect_submodules("uvicorn")

# Optional extras that may or may not be installed in the source venv --
# harmless to list if missing (PyInstaller just won't find them), but pull
# them in when present so `pleiades ui` doesn't lose features in the bundle.
for pkg in ("discord", "anthropic", "pypdf", "docx", "openpyxl", "PIL"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

hiddenimports += [
    "pleiades",
    "pleiades.cli",
    "pleiades.config",
    "pleiades.profiles",
    "pleiades.models",
    "pleiades.vault",
    "pleiades.anamnesis",
    "pleiades.webui",
    "pleiades.webui.server",
    "pleiades.webui._guard",
]

# The webui's static frontend (index.html/app.js/styles.css/next.html) is
# served straight off disk by FastAPI's StaticFiles -- it isn't imported by
# Python, so collect_all(pleiades) below (data only) is what actually pulls
# it into the bundle; keep an explicit fallback path too so a bundle built
# from a slightly different pleiades/ layout still works.
pleiades_datas, pleiades_binaries, pleiades_hidden = collect_all(
    "pleiades", include_py_files=False
)
datas += pleiades_datas
binaries += pleiades_binaries
hiddenimports += pleiades_hidden

static_dir = os.path.join(REPO_ROOT, "pleiades", "webui", "static")
if os.path.isdir(static_dir):
    datas.append((static_dir, "pleiades/webui/static"))

a = Analysis(
    ["entry.py"],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pleiades-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pleiades-backend",
)
