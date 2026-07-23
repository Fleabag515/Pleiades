"""Speech-fallback transcription (audio-fallback design decision, 2026-07-23).

Scope decision, spelled out because it's deliberate and not an oversight: a
full council review (two independent reviewers, both verified against the
real codebase, not just asked to opine) found ZERO existing audio pathway
anywhere in Pleiades -- grepped for whisper/audio/transcri/.wav/.mp3 across
pleiades/ and engine/src/, nothing but chat-"transcript" false positives. No
model on this machine takes audio input, and llama.cpp's own audio-input
serving (--mmproj audio encoders) is rare and immature. Both reviewers
independently recommended shipping FALLBACK-ONLY for v1: do not attempt real
audio-capable-model passthrough this round. That is the scope of this
module -- it never touches --mmproj or the native engine's audio wiring, and
it is NOT how a real audio-capable model (if one is ever registered) would
consume audio. See engine.py's audio-capable check, which skips this module
entirely once that day comes.

Within the fallback, this module does speech-to-text ONLY, via whisper.cpp's
GGUF-model-driven `whisper-server` (github.com/ggerganov/whisper.cpp), the
same "tiny, separately-managed local server, POST bytes, get text back"
shape as pleiades/tools/vision_caption.py. Unlike vision_caption.py -- which
only ever points at an already-running, independently-managed server --
this module also manages its own lifecycle end to end (install-if-missing,
download the model, start on demand, stop), the same job pleiades/runtime.py
does for the llama-server binary itself.

Music/sound/genre tagging ("what IS this audio", not just what was said) is
explicitly NOT attempted here. Whisper is speech-only and cannot do this at
all. The realistic options for it (LAION CLAP, PANNs/AST classifiers,
Qwen2-Audio-Instruct) all require adding a heavy Python ML stack (torch +
transformers, realistically 2GB+ of new dependencies, with real risk of
conflicting with the CUDA build llama-cpp-python already uses on this
machine) for a feature neither council reviewer had high confidence
recommending a specific model for. Shipping that anyway would be exactly the
"ship something you're not confident actually works" the review flagged as
the riskiest part of this ask -- so it's deferred, not built, this round.
engine.py's fallback note says plainly that only a transcript is available,
not a music/sound description.

Env-gated, following PLEIADES_VISION_FALLBACK_URL's naming convention, but
with the opposite default polarity on purpose (2026-07-25 fix -- see below):
  PLEIADES_AUDIO_FALLBACK_ENABLED   "1"/"true"/"yes"/"on" (or unset/blank)
                                    to allow this module to install/manage
                                    its own whisper-server on demand. ON
                                    BY DEFAULT: unlike vision_caption.py
                                    (which has nothing sensible to default
                                    to -- it only ever points at an
                                    externally-run server nothing here
                                    provisions), this module's whole job
                                    is to self-provision end to end
                                    (install binary, download model,
                                    start server) the first time it is
                                    actually needed, so "enabled" can
                                    safely be the default: absence of any
                                    real audio attachment still means a
                                    complete no-op (no download, no
                                    process, no network call) -- only
                                    is_configured() flips, not whether
                                    anything actually runs. Set this to a
                                    falsy value ("0"/"false"/"no"/"off")
                                    to explicitly opt OUT (e.g. a
                                    disk/network-constrained machine that
                                    never wants whisper.cpp installed).
                                    Fixed 2026-07-25: this previously
                                    defaulted OFF, which meant a fully
                                    built and live-verified transcription
                                    pipeline silently never ran on any
                                    real chat turn because nothing in
                                    install.sh/.env.example ever set it --
                                    see git log for the incident this
                                    traces back to.
  PLEIADES_AUDIO_FALLBACK_URL       point at an ALREADY-running
                                    whisper-server instead (skips
                                    install/lifecycle entirely -- same
                                    shape as the vision fallback). Takes
                                    priority over ENABLED when both are set.
  PLEIADES_AUDIO_FALLBACK_MODEL     ggml model filename stem (default:
                                    "large-v3-turbo-q5_0", ~570MB, the
                                    council's recommended pick; set to
                                    "small" if conservative on disk/load
                                    time).
  PLEIADES_AUDIO_FALLBACK_HOST/PORT our managed server's bind address
                                    (default 127.0.0.1:8791).
  PLEIADES_WHISPER_BIN              explicit override for an already-
                                    installed whisper-server binary
                                    (e.g. `brew install whisper-cpp` on
                                    macOS, which has no prebuilt release
                                    zip -- see _platform_asset()).

Runs on CPU by design, not GPU: this is a fallback path for models that
already occupy the GPU, so it deliberately never competes for VRAM.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

from .. import config

_DEFAULT_MODEL = "large-v3-turbo-q5_0"
_WHISPER_CPP_GITHUB_REPO = "ggerganov/whisper.cpp"   # release zips/tarballs (whisper-server binary)
_WHISPER_CPP_HF_REPO = "ggerganov/whisper.cpp"       # Hugging Face (ggml *.bin model files)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _falsy(v: str) -> bool:
    return v.strip().lower() in ("0", "false", "no", "off")


def _enabled() -> bool:
    """On by default (fixed 2026-07-25 -- see module docstring): the
    fallback self-provisions its own binary/model/server on first real
    use, so there is nothing unsafe about defaulting to "enabled". Only an
    explicit falsy value opts out; blank/unset/anything else is treated
    as enabled so a stray typo in this env var fails safe (transcription
    still runs) rather than failing silent (transcription silently never
    runs, which is exactly the bug this default flip fixes)."""
    return not _falsy(os.environ.get("PLEIADES_AUDIO_FALLBACK_ENABLED", ""))


def _url_override() -> str:
    return os.environ.get("PLEIADES_AUDIO_FALLBACK_URL", "").strip().rstrip("/")


def is_configured() -> bool:
    """True when the audio fallback is usable: either pointed at an
    already-running server, or (the default) allowed to manage its own.
    When this is False, every other function below is a strict no-op (no
    network call, no process spawned, no download) -- but note that even
    when True, nothing actually happens until transcribe() is called with
    a real audio path; "configured" means "allowed to self-provision on
    demand", not "already running"."""
    return bool(_url_override()) or _enabled()


def _model_name() -> str:
    return os.environ.get("PLEIADES_AUDIO_FALLBACK_MODEL", "").strip() or _DEFAULT_MODEL


def _host() -> str:
    return os.environ.get("PLEIADES_AUDIO_FALLBACK_HOST", "").strip() or "127.0.0.1"


def _port() -> int:
    try:
        return int(os.environ.get("PLEIADES_AUDIO_FALLBACK_PORT", "").strip() or "8791")
    except ValueError:
        return 8791


def _runtime_dir() -> Path:
    d = config.PLEIADES_HOME / "runtime" / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _models_dir() -> Path:
    d = config.PLEIADES_HOME / "models" / "whisper"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _running_json() -> Path:
    return config.PLEIADES_HOME / "audio-fallback-running.json"


def model_path() -> Path:
    return _models_dir() / f"ggml-{_model_name()}.bin"


# --------------------------------------------------------------------------- #
# Binary resolution + install (mirrors runtime.py's find_native()/install())
# --------------------------------------------------------------------------- #
def find_binary() -> Optional[str]:
    """Path to whisper-server. Resolution order: PLEIADES_WHISPER_BIN
    (explicit override) -> managed install under
    ~/.pleiades/runtime/whisper -> PATH. Same shape as
    runtime.find_native()."""
    exe = "whisper-server.exe" if os.name == "nt" else "whisper-server"
    override = os.environ.get("PLEIADES_WHISPER_BIN", "").strip()
    if override and Path(override).is_file():
        return override
    managed = sorted(_runtime_dir().rglob(exe))
    if managed:
        return str(managed[0])
    return shutil.which(exe)


def _platform_asset() -> Optional[str]:
    """Which whisper.cpp GitHub release asset to fetch for this machine.

    whisper.cpp's release matrix is much smaller than llama.cpp's (no MoE/
    backend picking needed -- this is a CPU-only fallback by design, see
    module docstring): one plain CPU build per OS/arch is enough. Verified
    live against the v1.9.1 release asset list (2026-07-23): Windows and
    Ubuntu (x64/arm64) ship a whisper-server binary in a
    whisper-bin-<os>[-<arch>].(zip|tar.gz); macOS does NOT -- its only
    release asset is an xcframework (Swift/iOS bindings, no CLI server), so
    on macOS this returns None and install() raises with a clear pointer to
    `brew install whisper-cpp` + PLEIADES_WHISPER_BIN instead of pretending
    to fetch something that doesn't exist.
    """
    if sys.platform == "darwin":
        return None
    if os.name == "nt":
        return "whisper-bin-x64.zip"
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    return f"whisper-bin-ubuntu-{arch}.tar.gz"


def _native_env(binary: str) -> dict:
    """whisper-server ships bundled shared libs beside the binary (same
    relocatable-build situation runtime.native_env() handles for
    llama-server)."""
    d = str(Path(binary).parent)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    return {"LD_LIBRARY_PATH": f"{d}:{cur}" if cur else d}


def install(log: Optional[Callable[[str], None]] = None) -> str:
    """Download the official whisper.cpp CPU server prebuilt for this
    machine (mirrors runtime.install()'s shape for llama-server). Returns
    the path to the installed whisper-server binary."""
    import httpx

    say = log or (lambda _m: None)
    asset_name = _platform_asset()
    if not asset_name:
        raise RuntimeError(
            "No prebuilt whisper.cpp server for this platform (macOS ships "
            "no whisper-server release binary). Try `brew install "
            "whisper-cpp` and set PLEIADES_WHISPER_BIN to the installed "
            "whisper-server path, or build from source: "
            "https://github.com/ggerganov/whisper.cpp"
        )
    say("looking up the latest whisper.cpp release...")
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        r = c.get(f"https://api.github.com/repos/{_WHISPER_CPP_GITHUB_REPO}/releases/latest")
        r.raise_for_status()
        rel = r.json()
        asset = next((a for a in rel.get("assets", []) if a.get("name") == asset_name), None)
        if not asset:
            raise RuntimeError(
                f"whisper.cpp release {rel.get('tag_name', '')} did not contain "
                f"the expected asset '{asset_name}'."
            )
        say(f"downloading {asset['name']} ({rel.get('tag_name', '')}) ...")
        dest = _runtime_dir() / asset["name"]
        with c.stream("GET", asset["browser_download_url"]) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(1024 * 1024):
                    f.write(chunk)

    say("extracting...")
    target = _runtime_dir() / "install"
    if target.exists():
        shutil.rmtree(target)
    if dest.name.endswith(".tar.gz"):
        with tarfile.open(dest, "r:gz") as t:
            t.extractall(target, filter="data")
    else:
        with zipfile.ZipFile(dest) as z:
            z.extractall(target)
    dest.unlink(missing_ok=True)

    exe = "whisper-server.exe" if os.name == "nt" else "whisper-server"
    found = list(target.rglob(exe))
    if not found:
        raise RuntimeError("whisper.cpp archive did not contain whisper-server.")
    binary = found[0]
    if os.name != "nt":
        for f in binary.parent.iterdir():   # binary + bundled shared libs
            try:
                f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
            except OSError:
                pass
    say(f"installed {binary}")
    return str(binary)


def ensure_model(log: Optional[Callable[[str], None]] = None) -> Path:
    """Download the configured whisper ggml model from Hugging Face if not
    already present locally. Returns the local path either way."""
    import httpx

    say = log or (lambda _m: None)
    dest = model_path()
    if dest.is_file():
        return dest
    url = f"https://huggingface.co/{_WHISPER_CPP_HF_REPO}/resolve/main/{dest.name}"
    say(f"downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.Client(timeout=httpx.Timeout(30.0, read=120.0), follow_redirects=True) as c:
        with c.stream("GET", url) as r:
            if r.status_code == 404:
                raise RuntimeError(
                    f"No such whisper.cpp model file on Hugging Face: {dest.name} "
                    f"(check PLEIADES_AUDIO_FALLBACK_MODEL)"
                )
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1024 * 1024):
                    f.write(chunk)
    tmp.rename(dest)
    say(f"model ready: {dest}")
    return dest


# --------------------------------------------------------------------------- #
# Lifecycle (start on demand / stop) -- single managed instance, mirrors the
# start/stop shape of models.py's ModelManager but for one dedicated server
# rather than a multi-model registry.
# --------------------------------------------------------------------------- #
def _load_running() -> dict:
    p = _running_json()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_running(data: dict) -> None:
    _running_json().write_text(json.dumps(data, indent=2), encoding="utf-8")


def _pid_alive(pid: "int | None") -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just owned by someone else
    except OSError:
        return False
    return True


def _server_healthy(host: str, port: int) -> bool:
    """whisper-server has no dedicated health endpoint; hitting `/` with a
    short timeout is enough to tell "something is listening and answering
    HTTP" apart from "nothing is there yet" during startup. Any HTTP
    response at all (even a 404 from the missing static `public` folder)
    counts as healthy -- we only ever call /inference for real work."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=1.5):
            return True
    except urllib.error.HTTPError:
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _ensure_server(timeout: float = 180.0) -> Optional[str]:
    """Start our managed whisper-server if it isn't already up. Returns its
    base URL, or None if it couldn't be installed/started for any reason
    (never raises -- callers treat that identically to "not configured")."""
    host, port = _host(), _port()
    run = _load_running()
    if run and _pid_alive(run.get("pid")) and _server_healthy(host, port):
        return f"http://{host}:{port}"

    binary = find_binary()
    if not binary:
        try:
            binary = install()
        except Exception:
            return None
    try:
        model = ensure_model()
    except Exception:
        return None

    threads = max(1, (os.cpu_count() or 4) // 2)   # leave headroom for the character's own model
    cmd = [binary, "--host", host, "--port", str(port), "-m", str(model), "-t", str(threads)]

    logdir = config.PLEIADES_HOME / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logf = open(logdir / "audio-fallback.log", "ab")

    kwargs: dict = {"env": {**os.environ, **_native_env(binary)}}
    if os.name == "posix":
        kwargs["start_new_session"] = True   # survive the CLI/engine process exiting

    try:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, **kwargs)
    except OSError:
        return None
    _save_running({"pid": proc.pid, "host": host, "port": port})

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_healthy(host, port):
            return f"http://{host}:{port}"
        if proc.poll() is not None:
            return None
        time.sleep(0.5)
    return None


def _base_url() -> Optional[str]:
    """The URL to POST /inference to: an explicit override if configured,
    else our own managed instance (installed/started on demand)."""
    override = _url_override()
    if override:
        return override
    if not _enabled():
        return None
    return _ensure_server()


def stop() -> bool:
    """Stop our managed instance. No-op (returns False) if pointed at an
    external server via PLEIADES_AUDIO_FALLBACK_URL, or nothing is running."""
    run = _load_running()
    pid = run.get("pid")
    if not pid:
        return False
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    _save_running({})
    return True


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def _multipart_body(boundary: str, audio: Path) -> bytes:
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{audio.name}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        audio.read_bytes(),
        b"\r\n",
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="response_format"\r\n\r\n',
        b"json\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts)


def transcribe(audio_path: str, *, timeout: float = 120.0) -> "str | None":
    """Ask the configured whisper.cpp fallback to transcribe an audio file
    to plain text (speech only -- see module docstring for why music/sound
    tagging is a separate, not-yet-built step).

    Returns None -- and never raises -- if the fallback isn't configured,
    the file doesn't exist, the server can't be reached/started, or the
    response is unusable. Same no-op-when-unconfigured contract as
    vision_caption.caption_screenshot.
    """
    if not is_configured() or not audio_path:
        return None
    p = Path(audio_path)
    if not p.is_file():
        return None
    base_url = _base_url()
    if not base_url:
        return None

    boundary = "----pleiadesaudiofallback"
    body = _multipart_body(boundary, p)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    text = (data.get("text") or "").strip()
    return text or None
