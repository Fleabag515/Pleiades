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
import re
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


_NAME_BAD = re.compile(r'[/\\\x00-\x1f<>:"|?*]')


def validate_name(name: str) -> str:
    """Reject profile names that could escape PROFILES_DIR (e.g. '../..').

    Names are used directly as directory names (and by the web UI, where
    DELETE /api/profiles/{name} removes the directory), so they must never
    contain path separators or resolve outside the profiles dir.
    """
    if (not name or len(name) > 64 or _NAME_BAD.search(name)
            or name in {".", ".."} or name.startswith((".", " ")) or name.endswith((".", " "))):
        raise ValueError(
            f"Invalid profile name {name!r}: no path separators or control/special "
            "characters, no leading/trailing dots or spaces, max 64 chars."
        )
    return name


def profile_dir(name: str) -> Path:
    return PROFILES_DIR / validate_name(name)


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


def _layers(env: str, default: "int | str") -> "int | str":
    """n_gpu_layers from the environment: an int, or 'auto' (hardware-planned)."""
    raw = os.environ.get(env, "").strip()
    if not raw:
        return default
    if raw.lower() == "auto":
        return "auto"
    try:
        return int(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Model tiers (agent harness)
# --------------------------------------------------------------------------- #
# Anthropic model IDs (optional cloud tiers).
OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"


@dataclass
class Tier:
    """A (backend, model) pairing for a class of work."""

    backend: str          # "openai" (llama.cpp) | "anthropic" (API) | "claude-code" (subscription) | "ollama"
    model: str
    effort: str = "high"  # anthropic only: low|medium|high|max
    max_tokens: int = 8192


# Local-first defaults. Pleiades is an inference engine: the default brain runs on
# our own llama.cpp OpenAI-compatible server (backend "openai"). Cloud/Ollama tiers
# are optional so the workspace *can* call hosted models.
DEFAULT_TIERS: dict[str, dict] = {
    "local":      {"backend": "openai", "model": "local", "max_tokens": 4096},
    "chat":       {"backend": "openai", "model": "local", "max_tokens": 8192},
    "coder":      {"backend": "openai", "model": "local", "max_tokens": 16000},
    # Subagent roles route to these by name; local-first by default, but a user
    # can repoint any of them at a cloud model in config.json (e.g. research ->
    # anthropic/Sonnet) without touching code.
    "research":   {"backend": "openai", "model": "local", "max_tokens": 8192},
    "fast":       {"backend": "openai", "model": "local", "max_tokens": 2048},
    "cloud":      {"backend": "anthropic", "model": OPUS,   "effort": "high",   "max_tokens": 8192},
    "cloud-fast": {"backend": "anthropic", "model": SONNET, "effort": "medium", "max_tokens": 8192},
    "claude":     {"backend": "claude-code", "model": "", "effort": "high", "max_tokens": 8192},
    "ollama":     {"backend": "ollama", "model": "qwen2.5:7b", "max_tokens": 4096},
}


@dataclass
class Settings:
    """Project-wide settings — the single source of truth.

    Loaded via `Settings.load()`: built-in defaults, then an optional config.json at
    the project root, then environment variables (PLEIADES_*, ANTHROPIC_API_KEY,
    OLLAMA_HOST, OPENAI_HOST) which win. No secrets are stored here; per-character
    secrets live in the vault.
    """

    # --- Anamnesis control API (the canonical context manager / memory proxy) ---
    anamnesis_control_url: str = "http://127.0.0.1:9000"

    # --- Our in-process inference engine (llama.cpp OpenAI-compatible server) ---
    model_path: str = ""
    inference_host: str = "127.0.0.1"
    inference_port: int = 8080
    n_ctx: int = 8192
    # "auto" (default) plans GPU offload from detected hardware at launch;
    # an int (-1 all layers, 0 CPU-only) overrides. See pleiades.hardware.
    n_gpu_layers: "int | str" = "auto"
    chat_format: str = ""  # blank = let llama.cpp auto-detect

    # --- Autofit: speed/quality preference for quant choice (speed|balanced|quality) ---
    autofit_preference: str = "balanced"

    # --- Web search ---
    searxng_url: str = "http://127.0.0.1:8888"

    # --- Optional override: a different OpenAI-compatible backend for a character ---
    backend_base_url: str = ""
    backend_api_key: str = ""

    # --- Agent harness: model routing ---
    tiers: dict = field(default_factory=dict)
    default_tier: str = "local"
    anthropic_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    # Where the "openai" backend posts. Blank -> derived from our inference engine in
    # load(); identity.bind_character() repoints it at a character's Anamnesis proxy.
    openai_host: str = ""

    # --- Agent harness: execution policy + loop ---
    exec_policy: str = "ask"          # ask | allow | deny (read-only tools always run)
    workspace_root: str = "."
    max_steps: int = 40           # hard cap for Claude-lab/subagents; the main loop is unbounded
    eval_interval: int = 40       # inject a self-evaluation nudge every N agent turns (0 = off)
    max_subagent_depth: int = 3
    context_budget: int = 180_000
    tool_mode: str = "auto"           # auto | all | search
    tool_search_threshold: int = 18

    # --- Agent harness: in-process memory tier (augments Anamnesis) ---
    memory_dir: str = "memory"
    embed_enabled: bool = True
    embed_model: str = "nomic-embed-text"

    # --- MCP servers to mount at startup ---
    mcp_servers: list = field(default_factory=list)

    # --- resolved at load() ---
    root: str = "."

    @classmethod
    def load(cls, root: "str | Path | None" = None) -> "Settings":
        import json

        root_path = Path(root or os.environ.get("PLEIADES_ROOT") or Path.cwd()).resolve()
        s = cls()
        s.root = str(root_path)
        s.tiers = {k: Tier(**v) for k, v in DEFAULT_TIERS.items()}

        # config.json overlay (project root), if present.
        cfg_path = root_path / "config.json"
        if cfg_path.is_file():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}
            for k, v in data.items():
                if k.startswith("_"):
                    continue
                if k == "tiers" and isinstance(v, dict):
                    for tn, tv in v.items():
                        try:
                            s.tiers[tn] = Tier(**tv)
                        except TypeError:
                            pass
                elif hasattr(s, k):
                    setattr(s, k, v)

        # env overlay (highest priority).
        s.anamnesis_control_url = os.environ.get("PLEIADES_ANAMNESIS_CONTROL_URL", s.anamnesis_control_url)
        s.model_path = os.environ.get("PLEIADES_MODEL_PATH", s.model_path)
        s.inference_host = os.environ.get("PLEIADES_INFERENCE_HOST", s.inference_host)
        s.inference_port = _int("PLEIADES_INFERENCE_PORT", s.inference_port)
        s.n_ctx = _int("PLEIADES_N_CTX", s.n_ctx)
        s.n_gpu_layers = _layers("PLEIADES_N_GPU_LAYERS", s.n_gpu_layers)
        s.chat_format = os.environ.get("PLEIADES_CHAT_FORMAT", s.chat_format)
        s.searxng_url = os.environ.get("PLEIADES_SEARXNG_URL", s.searxng_url)
        s.autofit_preference = os.environ.get("PLEIADES_AUTOFIT", s.autofit_preference)
        s.backend_base_url = os.environ.get("PLEIADES_BACKEND_BASE_URL", s.backend_base_url)
        s.backend_api_key = os.environ.get("PLEIADES_BACKEND_API_KEY", s.backend_api_key)
        s.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", s.anthropic_api_key)
        s.ollama_host = os.environ.get("OLLAMA_HOST", s.ollama_host)
        s.openai_host = os.environ.get("OPENAI_HOST", s.openai_host)
        s.embed_model = os.environ.get("PLEIADES_EMBED_MODEL", s.embed_model)
        if os.environ.get("PLEIADES_EXEC_POLICY"):
            s.exec_policy = os.environ["PLEIADES_EXEC_POLICY"]
        s.eval_interval = _int("PLEIADES_EVAL_INTERVAL", s.eval_interval)

        # The "openai" backend defaults to our local inference engine.
        if not s.openai_host:
            s.openai_host = s.inference_base_url

        # Resolve workspace/memory relative to root.
        if not Path(s.workspace_root).is_absolute():
            s.workspace_root = str(root_path / s.workspace_root)
        if not Path(s.memory_dir).is_absolute():
            s.memory_dir = str(root_path / s.memory_dir)
        return s

    def tier(self, name: str) -> "Tier":
        return self.tiers.get(name) or self.tiers[self.default_tier]

    @property
    def inference_base_url(self) -> str:
        """The OpenAI-compatible URL our inference server exposes."""
        return f"http://{self.inference_host}:{self.inference_port}/v1"

    @property
    def upstream_base_url(self) -> str:
        """What Anamnesis `upstream.baseUrl` should point at (our engine by default)."""
        return self.backend_base_url or self.inference_base_url

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        d["tiers"] = {k: (asdict(v) if isinstance(v, Tier) else v)
                      for k, v in self.tiers.items()}
        if d.get("anthropic_api_key"):
            d["anthropic_api_key"] = "***"
        if d.get("backend_api_key"):
            d["backend_api_key"] = "***"
        return d


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
