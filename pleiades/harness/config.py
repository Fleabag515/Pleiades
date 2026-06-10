"""
Config — one place for model routing, execution policy, and paths.

Loads from (in order, later wins):
  1. built-in defaults below
  2. config.json next to the project root (if present)
  3. environment variables (PLEIADES_*, ANTHROPIC_API_KEY, OLLAMA_HOST)

Model tiers map a task's difficulty to a concrete (backend, model). The main
chat loop uses `chat`; subagents pick a tier by name. This is the "agents of
varying quality for different tasks" knob — edit the tiers, not the code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# Model IDs (Anthropic, current as of build). Opus 4.8 is the most capable.
OPUS = "claude-opus-4-8"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"


@dataclass
class Tier:
    """A (backend, model) pairing for a class of work."""
    backend: str          # "anthropic" | "ollama"
    model: str
    effort: str = "high"  # anthropic only: low|medium|high|max
    max_tokens: int = 8192


# Default tier table. "chat" is the main brain; the rest are subagent targets.
DEFAULT_TIERS: dict[str, dict] = {
    # Big reasoning / planning / hard coding -> Claude Opus.
    "chat":    {"backend": "anthropic", "model": OPUS,   "effort": "high",  "max_tokens": 8192},
    "coder":   {"backend": "anthropic", "model": OPUS,   "effort": "high",  "max_tokens": 16000},
    "research":{"backend": "anthropic", "model": SONNET, "effort": "medium","max_tokens": 8192},
    # Cheap / fast / private -> local Ollama. Swap the model to whatever you pulled.
    "fast":    {"backend": "ollama",    "model": "qwen2.5:7b",  "max_tokens": 4096},
    "local":   {"backend": "ollama",    "model": "qwen2.5:7b",  "max_tokens": 4096},
}


@dataclass
class Config:
    # --- model routing ---
    tiers: dict[str, Tier] = field(default_factory=dict)
    default_tier: str = "chat"

    # --- credentials / endpoints ---
    anthropic_api_key: str | None = None
    ollama_host: str = "http://localhost:11434"
    openai_host: str = "http://127.0.0.1:8084/v1"

    # --- execution policy ---
    # The PC is the agent's home, but unsandboxed is the WORST case, not the
    # default. "ask" prompts before side-effecting actions; "allow" runs freely;
    # "deny" blocks them. Read-only tools always run.
    exec_policy: str = "ask"          # ask | allow | deny
    workspace_root: str = "."         # default cwd for file/shell tools
    max_steps: int = 40               # agent loop safety cap
    max_subagent_depth: int = 3       # how deep subagents may nest

    # --- context management ---
    # Approximate token budget of the model's context window. At 60% the agent
    # is warned in-band; at 75% old tool outputs in the transcript are squeezed
    # (a pre-compress checkpoint of working memory is saved first).
    context_budget: int = 180_000

    # --- tool scaling ---
    # With a big fleet (dozens of MCP tools), sending every schema each turn
    # bloats context. "search" mode hands the model only discovery tools
    # (find_tools / call_tool) and loads schemas on demand. "auto" switches to
    # search once the catalog passes tool_search_threshold.
    tool_mode: str = "auto"           # auto | all | search
    tool_search_threshold: int = 18

    # --- memory (anamnesis) ---
    memory_dir: str = "memory"
    # Semantic recall via local Ollama embeddings. If the model isn't pulled or
    # Ollama isn't running, recall silently degrades to keyword matching.
    embed_enabled: bool = True
    embed_model: str = "nomic-embed-text"

    # --- MCP servers to mount at startup ---
    # Each: {name, command, args:[...], cwd?, env?, safe_tools?:[...]}.
    # Pleiades speaks MCP as a client, so any MCP server's tools become Pleiades
    # tools (e.g. the whole NyzKh fleet: discord, browser, deep-research, ...).
    mcp_servers: list[dict] = field(default_factory=list)

    # --- paths (filled in at load) ---
    root: str = "."

    # -------------------------------------------------------------------
    @classmethod
    def load(cls, root: str | Path | None = None) -> "Config":
        root = Path(root or os.environ.get("PLEIADES_ROOT") or Path.cwd()).resolve()
        cfg = cls(root=str(root))
        cfg.tiers = {k: Tier(**v) for k, v in DEFAULT_TIERS.items()}

        # config.json overlay
        cfg_path = root / "config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k == "tiers":
                    for tname, tval in v.items():
                        cfg.tiers[tname] = Tier(**tval)
                elif hasattr(cfg, k):
                    setattr(cfg, k, v)

        # env overlay (highest priority for secrets/endpoints)
        cfg.anthropic_api_key = (
            os.environ.get("ANTHROPIC_API_KEY") or cfg.anthropic_api_key
        )
        cfg.ollama_host = os.environ.get("OLLAMA_HOST", cfg.ollama_host)
        cfg.embed_model = os.environ.get("PLEIADES_EMBED_MODEL", cfg.embed_model)
        cfg.openai_host = os.environ.get("OPENAI_HOST", cfg.openai_host)
        if os.environ.get("PLEIADES_EXEC_POLICY"):
            cfg.exec_policy = os.environ["PLEIADES_EXEC_POLICY"]

        # resolve workspace + memory relative to root
        if not Path(cfg.workspace_root).is_absolute():
            cfg.workspace_root = str(root / cfg.workspace_root)
        if not Path(cfg.memory_dir).is_absolute():
            cfg.memory_dir = str(root / cfg.memory_dir)
        return cfg

    def tier(self, name: str) -> Tier:
        return self.tiers.get(name) or self.tiers[self.default_tier]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tiers"] = {k: asdict(v) for k, v in self.tiers.items()}
        if d.get("anthropic_api_key"):
            d["anthropic_api_key"] = "***"  # never echo secrets
        return d
