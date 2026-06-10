"""Smart model fetching from Hugging Face.

`fetch_model("TheBloke/SomeModel-GGUF")` lists the repo's GGUF files, picks the
best quantization for the detected hardware (pleiades.hardware.pick_quant),
streams it to ~/.pleiades/models/, and registers it with the ModelManager —
with `n_gpu_layers="auto"` so placement is planned at launch.

Uses the public HF API over httpx; no extra dependencies, no token needed for
public repos (set HF_TOKEN for gated ones).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, Optional

import httpx

from . import config, hardware

HF_API = "https://huggingface.co"


class FetchError(RuntimeError):
    pass


def models_dir() -> Path:
    d = config.PLEIADES_HOME / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def list_repo_ggufs(repo: str, *, client: Optional[httpx.Client] = None) -> list[dict]:
    """[{name, size}] for every .gguf in an HF repo (recursive)."""
    c = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        r = c.get(f"{HF_API}/api/models/{repo}/tree/main",
                  params={"recursive": "true"}, headers=_headers())
        if r.status_code == 404:
            raise FetchError(f"No such Hugging Face repo: '{repo}'")
        if r.status_code in (401, 403):
            raise FetchError(f"'{repo}' is gated/private — set HF_TOKEN.")
        r.raise_for_status()
        out = []
        for item in r.json():
            path = item.get("path", "")
            if item.get("type") == "file" and path.lower().endswith(".gguf"):
                out.append({"name": path, "size": int(item.get("size", 0))})
        return out
    except httpx.HTTPError as e:
        raise FetchError(f"Hugging Face API error for '{repo}': {e}") from e
    finally:
        if client is None:
            c.close()


def search_gguf_repos(query: str, limit: int = 8,
                      *, client: Optional[httpx.Client] = None) -> list[dict]:
    """Search HF for GGUF model repos: [{id, downloads, likes}]."""
    c = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        r = c.get(f"{HF_API}/api/models",
                  params={"search": query, "filter": "gguf", "sort": "downloads",
                          "direction": "-1", "limit": str(limit)},
                  headers=_headers())
        r.raise_for_status()
        return [{"id": m.get("id", ""), "downloads": m.get("downloads", 0),
                 "likes": m.get("likes", 0)} for m in r.json()]
    except httpx.HTTPError as e:
        raise FetchError(f"Hugging Face search failed: {e}") from e
    finally:
        if client is None:
            c.close()


def _download(repo: str, filename: str, dest: Path,
              progress: Optional[Callable[[int, int], None]] = None,
              *, client: Optional[httpx.Client] = None) -> None:
    url = f"{HF_API}/{repo}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    c = client or httpx.Client(timeout=httpx.Timeout(30.0, read=120.0),
                               follow_redirects=True)
    try:
        with c.stream("GET", url, headers=_headers()) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(done, total)
        tmp.replace(dest)
    except httpx.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise FetchError(f"Download failed for {filename}: {e}") from e
    finally:
        if client is None:
            c.close()


def remote_gguf_meta(repo: str, filename: str,
                     *, client: Optional[httpx.Client] = None):
    """Parse a GGUF header straight off Hugging Face via a ranged GET.

    The header (incl. all metadata KVs) usually fits well inside 32 MB; the
    tensor data never needs to be downloaded. Returns GGUFMeta (possibly
    size-only on failure) — the caller fixes file_size itself.
    """
    import tempfile

    from .hardware import read_gguf_meta
    url = f"{HF_API}/{repo}/resolve/main/{filename}"
    c = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        headers = {**_headers(), "Range": "bytes=0-33554431"}
        with c.stream("GET", url, headers=headers) as r:
            if r.status_code not in (200, 206):
                raise FetchError(f"header probe got HTTP {r.status_code}")
            with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
                budget = 32 * 1024 * 1024
                for chunk in r.iter_bytes(1024 * 1024):
                    tmp.write(chunk)
                    budget -= len(chunk)
                    if budget <= 0:
                        break
                path = tmp.name
        meta = read_gguf_meta(path)
        os.unlink(path)
        return meta
    except (httpx.HTTPError, OSError):
        return None
    finally:
        if client is None:
            c.close()


def _default_name(repo: str) -> str:
    base = repo.split("/")[-1]
    base = re.sub(r"(?i)[-_.](gguf|gguf-imatrix)$", "", base)
    return base.lower()


def fetch_model(repo: str, *, name: str = "", quant: str = "", n_ctx: int = 8192,
                hw: Optional[hardware.Hardware] = None,
                progress: Optional[Callable[[str, int, int], None]] = None,
                client: Optional[httpx.Client] = None) -> dict:
    """Pick, download, and register the best GGUF from an HF repo.

    Returns the registered model entry plus 'chosen' (filename), 'why', and
    'local_path'. `quant` (e.g. "Q4_K_M") overrides the hardware-based pick.
    `progress(filename, done_bytes, total_bytes)` is optional.
    """
    files = list_repo_ggufs(repo, client=client)
    if not files:
        raise FetchError(f"'{repo}' contains no .gguf files.")

    if quant:
        grouped = hardware.group_split_ggufs(files)
        matches = [g for g in grouped
                   if hardware.quant_of(g["name"]) == quant.upper()]
        if not matches:
            avail = sorted({hardware.quant_of(g["name"]) for g in grouped} - {""})
            raise FetchError(f"No {quant.upper()} in '{repo}'. "
                             f"Available: {', '.join(avail) or 'unknown'}")
        chosen = min(matches, key=lambda g: g["size"])
        chosen["why"] = f"{quant.upper()}: requested explicitly"
    else:
        from . import runtime
        from .autofit import choose_quant
        grouped = hardware.group_split_ggufs(files)
        # Probe the smallest file's header so MoE structure informs the choice.
        meta_hint = None
        if grouped:
            smallest = min(grouped, key=lambda g: g["size"])
            meta_hint = remote_gguf_meta(repo, sorted(smallest["parts"])[0],
                                         client=client)
        pref = config.Settings.load().autofit_preference
        choice = choose_quant(files, hw or hardware.detect(), n_ctx=n_ctx,
                              preference=pref, caps=runtime.caps(),
                              meta_hint=meta_hint)
        if choice is None:
            raise FetchError(f"'{repo}' contains no usable .gguf files.")
        chosen = choice.entry

    dest_dir = models_dir() / repo.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)
    for part in chosen["parts"]:
        dest = dest_dir / Path(part).name
        if dest.is_file() and dest.stat().st_size > 0:
            continue  # already downloaded
        _download(repo, part, dest,
                  (lambda d, t, _p=part: progress(_p, d, t)) if progress else None,
                  client=client)

    # llama.cpp loads split GGUFs via the first part.
    first = dest_dir / Path(sorted(chosen["parts"])[0]).name

    from .models import ModelManager
    model_name = name or _default_name(repo)
    mm = ModelManager()
    if mm.get(model_name):
        entry = mm.get(model_name)
    else:
        entry = mm.add(model_name, str(first), n_ctx=n_ctx, n_gpu_layers="auto")
    entry = dict(entry)
    entry.update(chosen=chosen["name"], why=chosen.get("why", ""),
                 local_path=str(first), size=chosen["size"])
    return entry
