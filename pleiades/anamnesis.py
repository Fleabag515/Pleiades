"""Thin client for the Anamnesis control API + per-character proxy resolution.

Anamnesis is the upstream memory proxy. It runs a daemon with a control REST API
on 127.0.0.1:9000 and exposes each character as its own OpenAI-compatible proxy on
an auto-assigned port. Pleiades never reimplements memory — it just talks to these.

See CLAUDE.md §2 for the full contract.

This is the CANONICAL Anamnesis: the context manager / memory proxy. The harness's in-process working-memory tier (pleiades.harness.anamnesis) augments it per-character; it does not replace it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import httpx

# Mirrors src/lib/char-config.js's DEFAULTS + PERSONA_SHARED in the Anamnesis
# daemon: the control API's POST /characters never fills in a default config
# itself (that only happens client-side, inside the Node CLI's interactive
# wizard), so any client that wants to create a character over HTTP has to
# hand over a complete config. Keep these in sync if upstream changes them.
_CONFIG_DEFAULTS = {
    "context": {
        "tokenBudget": 50000, "systemReserveTokens": 4096, "recencyTurns": 8,
        "rotatingSlots": 6, "charsPerToken": 3.5, "minChunkChars": 50,
    },
    "memory": {
        "consolidationIntervalMs": 120000, "consolidationBatchSize": 50,
        "sceneClusterThreshold": 0.72, "minSceneSize": 2, "decayPruneThreshold": 0.05,
    },
    "extraction": {"maxRetries": 2, "timeoutMs": 45000, "startupBacklogLimit": 200},
    "foresight": {"maxRetries": 2, "timeoutMs": 45000, "startupBacklogLimit": 200},
    "inference": {"gpuLayerBudgetMB": 512},
    "history": {"maxAgeDays": 90},
}
_PERSONA_SHARED = {
    "timeoutMs": 45000,
    "drift": {"enabled": True, "checkEveryNTurns": 4, "driftThreshold": 0.55},
    "evolution": {"enabled": True, "consolidateAfterNObservations": 8, "maxEvolutionChars": 600},
    "injection": {"maxProfileChars": 700},
}


class AnamnesisError(RuntimeError):
    """Raised when the Anamnesis daemon is unreachable or returns an error."""


class Anamnesis:
    """Client for the Anamnesis control API (default base http://127.0.0.1:9000)."""

    def __init__(self, control_url: str = "http://127.0.0.1:9000", timeout: float = 30.0):
        self.control_url = control_url.rstrip("/")
        self._client = httpx.Client(base_url=self.control_url, timeout=timeout)

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Anamnesis":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------- #
    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise AnamnesisError(
                f"Cannot reach Anamnesis at {self.control_url} ({e}). "
                "Is the daemon running? Try `anamnesis status`."
            ) from e
        if resp.status_code >= 400:
            raise AnamnesisError(f"{method} {path} -> {resp.status_code}: {resp.text}")
        if resp.content and resp.headers.get("content-type", "").startswith("application/json"):
            return resp.json()
        return resp.text

    # -- control API -------------------------------------------------------- #
    def status(self) -> dict:
        return self._request("GET", "/status")

    def list_characters(self) -> list[dict]:
        data = self._request("GET", "/characters")
        # Tolerate either a bare list or {"characters": [...]}.
        if isinstance(data, dict):
            return data.get("characters", [])
        return data or []

    def get_character(self, name: str) -> dict:
        return self._request("GET", f"/characters/{name}")

    def _pick_port(self, base: int = 8084, max_scan: int = 200) -> int:
        """Best-effort free-looking port for a new character's proxy.

        The daemon re-resolves a free port on start anyway (character-manager.js's
        startCharacter() calls findFreePort and rewrites proxy.port if taken), so
        this just needs to avoid colliding with another *registered* character.
        """
        try:
            used = {c.get("port") for c in self.list_characters() if isinstance(c.get("port"), int)}
        except AnamnesisError:
            used = set()
        for i in range(max_scan):
            candidate = base + i
            if candidate not in used:
                return candidate
        raise AnamnesisError(f"Could not find a free-looking port starting from {base}.")

    def _build_default_config(self, name: str, port: int) -> dict:
        """Python port of char-config.js's buildConfig(), 'auto' persona variant."""
        db_path = str(Path.home() / ".anamnesis" / "characters" / name / "history.db")
        persona = {
            "enabled": True,
            "source": {
                "type": "auto",
                "openclaw": {"soulPath": "~/.openclaw/SOUL.md"},
                "file": {"path": "~/.anamnesis/character.md"},
                "inline": {"content": ""},
            },
            **_PERSONA_SHARED,
        }
        return {
            "proxy": {"port": port, "host": "127.0.0.1"},
            "upstream": {"baseUrl": "", "apiKey": "", "disableThinking": True},
            "inference": dict(_CONFIG_DEFAULTS["inference"]),
            "extraction": dict(_CONFIG_DEFAULTS["extraction"]),
            "context": dict(_CONFIG_DEFAULTS["context"]),
            "memory": dict(_CONFIG_DEFAULTS["memory"]),
            "history": {"dbPath": db_path, "maxAgeDays": _CONFIG_DEFAULTS["history"]["maxAgeDays"]},
            "foresight": dict(_CONFIG_DEFAULTS["foresight"]),
            "persona": persona,
        }

    def create_character(self, name: str, **cfg: Any) -> dict:
        """Create a character with a complete config.

        Anamnesis's control API (control-server.js: POST /characters) expects
        {"name": ..., "config": {...}} and never fills in defaults server-side
        (that only happens inside the Node CLI's interactive wizard) -- a bare
        {"name": ..., "upstream": {...}} 400s with a cryptic
        "Buffer.from(undefined)" error because the handler reads body.config,
        finds nothing, and tries to JSON.stringify(undefined) to disk. So we
        build a full default config here and merge any caller-supplied
        overrides (dicts merge one level deep; anything else replaces).
        """
        config = self._build_default_config(name, self._pick_port())
        for key, value in cfg.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
        payload = {"name": name, "config": config}
        return self._request("POST", "/characters", json=payload)

    def update_character(self, name: str, **cfg: Any) -> dict:
        # Anamnesis exposes edit via PATCH on the character resource.
        return self._request("PATCH", f"/characters/{name}", json=cfg)

    def start(self, name: str) -> dict:
        return self._request("POST", f"/characters/{name}/start")

    def stop(self, name: str) -> dict:
        return self._request("POST", f"/characters/{name}/stop")

    def delete(self, name: str) -> Any:
        return self._request("DELETE", f"/characters/{name}")

    # -- helpers ------------------------------------------------------------ #
    def exists(self, name: str) -> bool:
        try:
            self.get_character(name)
            return True
        except AnamnesisError:
            return False

    @staticmethod
    def _extract_port(character: dict) -> Optional[int]:
        """Pull the proxy port out of a character payload (tolerant of shapes)."""
        for key in ("port",):
            if isinstance(character.get(key), int):
                return character[key]
        proxy = character.get("proxy") or {}
        if isinstance(proxy, dict) and isinstance(proxy.get("port"), int):
            return proxy["port"]
        cfg = character.get("config") or {}
        cfg_proxy = cfg.get("proxy") or {} if isinstance(cfg, dict) else {}
        if isinstance(cfg_proxy, dict) and isinstance(cfg_proxy.get("port"), int):
            return cfg_proxy["port"]
        return None

    def proxy_port(self, name: str) -> int:
        """Resolve a character's proxy port from the API, falling back to the registry."""
        port = self._extract_port(self.get_character(name))
        if port is not None:
            return port
        port = self._port_from_registry(name)
        if port is not None:
            return port
        raise AnamnesisError(f"Could not resolve a proxy port for character '{name}'.")

    @staticmethod
    def _port_from_registry(name: str) -> Optional[int]:
        registry = Path.home() / ".anamnesis" / "registry.json"
        if not registry.is_file():
            return None
        import json

        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        entry = data.get(name) if isinstance(data, dict) else None
        if isinstance(entry, dict):
            port = entry.get("port") or (entry.get("proxy") or {}).get("port")
            if isinstance(port, int):
                return port
        return None

    def proxy_base_url(self, name: str) -> str:
        """The character's OpenAI-compatible endpoint: http://127.0.0.1:<port>/v1."""
        return f"http://127.0.0.1:{self.proxy_port(name)}/v1"

    def is_running(self, name: str) -> bool:
        char = self.get_character(name)
        status = (char.get("status") or char.get("state") or "").lower()
        if status:
            return status in {"running", "active", "started", "up"}
        # Fall back: a resolvable port that accepts connections.
        port = self._extract_port(char)
        return port is not None

    def set_upstream(self, name: str, upstream: dict) -> bool:
        """Point a character's upstream at `upstream` (e.g. {"baseUrl": ...}).

        Anamnesis 0.7+ supports PATCH /characters/:name (updateConfig), so the
        config is updated via the daemon. Strategy: skip if already correct; try
        PATCH; fall back to editing ~/.anamnesis/characters/<name>/config.json
        directly for older daemons (the documented layout, CLAUDE.md §2). A
        running proxy holds its config in memory, so we restart it afterward so
        the change actually applies. Returns True if restarted.
        """
        import json

        try:
            char = self.get_character(name)
        except AnamnesisError:
            return False  # nothing to point yet; creation will set it
        cfg = char.get("config") if isinstance(char.get("config"), dict) else char
        cur = (cfg.get("upstream") or {}) if isinstance(cfg, dict) else {}
        if not cur:
            # The daemon's character payload is minimal ({name, port, active,
            # running}) — without this disk fallback every chat looks like a
            # config change and restarts the proxy mid-boot, forever.
            disk = Path.home() / ".anamnesis" / "characters" / name / "config.json"
            if disk.is_file():
                try:
                    cur = json.loads(disk.read_text(encoding="utf-8")).get("upstream") or {}
                except (json.JSONDecodeError, OSError):
                    cur = {}
        if cur and all(cur.get(k) == v for k, v in upstream.items()):
            return False

        try:
            self.update_character(name, upstream=upstream)
        except AnamnesisError:
            path = Path.home() / ".anamnesis" / "characters" / name / "config.json"
            if not path.is_file():
                raise AnamnesisError(
                    f"Cannot reconfigure '{name}': daemon rejected the update and "
                    f"{path} does not exist."
                )
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                raise AnamnesisError(f"Cannot read {path}: {e}") from e
            merged = data.get("upstream") or {}
            merged.update(upstream)
            data["upstream"] = merged
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if self.is_running(name):
            self.stop(name)
            self.start(name)
            return True
        return False

    def ensure_running(self, name: str, *, create_if_missing: bool = True, **create_cfg: Any) -> str:
        """Create (optionally) + start a character if needed; return its proxy base URL."""
        if not self.exists(name):
            if not create_if_missing:
                raise AnamnesisError(f"Character '{name}' does not exist.")
            self.create_character(name, **create_cfg)
        if not self.is_running(name):
            self.start(name)
        # Give the proxy a moment to bind its port.
        for _ in range(20):
            try:
                return self.proxy_base_url(name)
            except AnamnesisError:
                time.sleep(0.25)
        return self.proxy_base_url(name)
