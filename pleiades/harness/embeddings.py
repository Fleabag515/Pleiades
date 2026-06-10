"""
embeddings.py — local semantic vectors for anamnesis recall.

Uses Ollama's /api/embed endpoint (e.g. nomic-embed-text, ~80MB) so the core
stays offline-capable. No Ollama, no model, no problem: the Embedder marks
itself dead after the first hard failure and Anamnesis silently falls back to
keyword recall. Vectors are cached in a sidecar JSON next to the notes, keyed
by content hash, so each note is embedded at most once per edit.

    emb = Embedder(host="http://localhost:11434", model="nomic-embed-text")
    vecs = emb.embed(["what model handles streaming?"])   # -> [[...]] or None
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Callable


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class Embedder:
    """Batch text -> vector via Ollama, with a session-level circuit breaker.

    embed_fn can be injected for testing: (model, [texts]) -> [[float]].
    """

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "nomic-embed-text",
                 embed_fn: Callable[[str, list[str]], list[list[float]]] | None = None,
                 timeout: float = 30.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._embed_fn = embed_fn
        self._dead = False           # set after first hard failure; stays down

    @property
    def alive(self) -> bool:
        return not self._dead

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch. Returns None (and trips the breaker) on failure."""
        if self._dead or not texts:
            return None if self._dead else []
        try:
            if self._embed_fn is not None:
                return self._embed_fn(self.model, texts)
            return self._embed_ollama(texts)
        except Exception:
            self._dead = True
            return None

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/embed", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        vecs = data.get("embeddings")
        if not isinstance(vecs, list) or len(vecs) != len(texts):
            raise ValueError("bad embedding response")
        return vecs


class VectorCache:
    """Sidecar store: slug -> {hash, vec}. Invalidated when the model changes."""

    def __init__(self, path: str | Path, model: str) -> None:
        self.path = Path(path)
        self.model = model
        self._data: dict = {"model": model, "notes": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if loaded.get("model") == model:
                    self._data = loaded
            except Exception:
                pass  # corrupt cache -> rebuild lazily

    def get(self, slug: str, text_hash: str) -> list[float] | None:
        entry = self._data["notes"].get(slug)
        if entry and entry.get("hash") == text_hash:
            return entry.get("vec")
        return None

    def put(self, slug: str, text_hash: str, vec: list[float]) -> None:
        self._data["notes"][slug] = {"hash": text_hash, "vec": vec}

    def drop(self, slug: str) -> None:
        self._data["notes"].pop(slug, None)

    def flush(self) -> None:
        try:
            self.path.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception:
            pass  # cache is an optimization, never fatal
