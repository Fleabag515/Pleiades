"""Paths and settings for Pleiades.

Everything character-scoped lives under PLEIADES_HOME (~/.pleiades by default):

    ~/.pleiades/
    ├── master.key                 # generated vault key (mode 0600) if no env key
    └── profiles/<name>/
        ├── profile.json           # non-secret profile config
        ├── vault.db               # encrypted secrets (Fernet)
        └── browser/               # Camoufox persistent context

Settings come from the environment (and an optional .env in the CWD). No secrets
are stored in this module; secrets live in the per-profile vault.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# .env loading (tiny, dependency-free)
# --------------------------------------------------------------------------- #
def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file without overriding existing vars."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path.cwd() / ".env")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _home() -> Path:
    override = os.environ.get("PLEIADES_HOME")
    return Path(override).expanduser() if override else Path.home() / ".pleiades"


PLEIADES_HOME: Path = _home()
PROFILES_DIR: Path = PLEIADES_HOME / "profiles"
MASTER_KEY_PATH: Path = PLEIADES_HOME / "master.key"


def ensure_home() -> None:
    """Create the Pleiades home + profiles dirs if missing."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / name


def vault_path(name: str) -> Path:
    return profile_dir(name) / "vault.db"


def browser_dir(name: str) -> Path:
    return profile_dir(name) / "browser"


def profile_json_path(name: str) -> Path:
    return profile_dir(name) / "profile.json"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def _int(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, "").strip() or default)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings, loaded from the environment via `Settings.load()`."""

    # Anamnesis control API
    anamnesis_control_url: str = "http://127.0.0.1:9000"

    # Our in-process inference engine (llama.cpp OpenAI-compatible server)
    model_path: str = ""
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080
    n_ctx: int = 8192
    n_gpu_layers: int = 0
    chat_format: str = ""  # blank = let llama.cpp auto-detect

    # Web search
    searxng_url: str = "http://127.0.0.1:8888"

    # Optional override: point a character at a different OpenAI-compatible backend
    # instead of our inference server. Leave blank to use our server.
    backend_base_url: str = ""
    backend_api_key: str = ""

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            anamnesis_control_url=os.environ.get(
                "PLEIADES_ANAMNESIS_CONTROL_URL", cls.anamnesis_control_url
            ),
            model_path=os.environ.get("PLEIADES_MODEL_PATH", ""),
            inference_host=os.environ.get("PLEIADES_INFERENCE_HOST", cls.inference_host),
            inference_port=_int("PLEIADES_INFERENCE_PORT", cls.inference_port),
            n_ctx=_int("PLEIADES_N_CTX", cls.n_ctx),
            n_gpu_layers=_int("PLEIADES_N_GPU_LAYERS", cls.n_gpu_layers),
            chat_format=os.environ.get("PLEIADES_CHAT_FORMAT", ""),
            searxng_url=os.environ.get("PLEIADES_SEARXNG_URL", cls.searxng_url),
            backend_base_url=os.environ.get("PLEIADES_BACKEND_BASE_URL", ""),
            backend_api_key=os.environ.get("PLEIADES_BACKEND_API_KEY", ""),
        )

    @property
    def inference_base_url(self) -> str:
        """The OpenAI-compatible URL our inference server exposes."""
        return f"http://{self.inference_host}:{self.inference_port}/v1"

    @property
    def upstream_base_url(self) -> str:
        """What Anamnesis `upstream.baseUrl` should point at.

        Defaults to our own inference server; overridable via PLEIADES_BACKEND_BASE_URL.
        """
        return self.backend_base_url or self.inference_base_url


# Email provider presets (host, port). Default is generic — fill in manually.
EMAIL_PRESETS: dict[str, dict[str, object]] = {
    "generic": {"imap_host": "", "imap_port": 993, "smtp_host": "", "smtp_port": 587},
    "gmail": {
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "note": "Requires a Google app password (2-Step Verification on).",
    },
    "mail.com": {
        "imap_host": "imap.mail.com",
        "imap_port": 993,
        "smtp_host": "smtp.mail.com",
        "smtp_port": 587,
        "note": "Enable POP3/IMAP in mail.com settings; use an app password.",
    },
    "outlook": {
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
        "note": "Modern auth may require OAuth.",
    },
}
