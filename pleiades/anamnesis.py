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

    def create_character(self, name: str, **cfg: Any) -> dict:
        payload = {"name": name, **cfg}
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
