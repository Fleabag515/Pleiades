"""Thin vision-fallback captioning (docs/specs/2026-07-22-vision-grounded-
browser-use-design.md, "Thin vision-fallback layer for text-only models").

This is deliberately NOT part of the Phase 2-6 vision-escalation-for-clicking
plan. It is a much smaller, always-separate side channel: a tiny (well under
1GB) dedicated captioning model, served by its own llama-server instance
completely independent of any character's assigned brain model, asked ONE
question ("what's on this screenshot, briefly") whenever BrowserTool's
accessibility-tree extraction (the 'elements' action) comes back empty --
the closest cheap proxy this project has for "this page is probably
canvas/WebGL/custom-rendered and text-based interaction won't find
anything here."

Deliberately NOT wired into ModelManager/Foundry -- that machinery is
Phase 2's job, and it's for the character's OWN vision-capable model.
This module is configured by a single env var pointing at an already-
running, independently-managed llama-server instance serving a small
mmproj-paired model (e.g. ggml-org/SmolVLM-256M-Instruct-GGUF, ~280MB
total for both GGUFs). Absent that env var, this whole module is a no-op:
today's "No interactive elements found" message is completely unchanged.

Why the raw /completions endpoint instead of /v1/chat/completions: verified
LIVE (2026-07-23, real headless-Chromium screenshot, real llama-server,
ggml-org/SmolVLM-256M-Instruct-GGUF on this project's pinned llama-server
build) that the chat-completions path's differential-autoparser
chat-template renderer silently drops the image/media-marker content part
for this model -- it renders down to plain "User: ...<end_of_utterance>"
text with the marker missing entirely, so mtmd then sees "1 bitmap, 0
markers" and the request 400s. The low-level /completions endpoint's
``{"prompt_string": ..., "multimodal_data": [...]}`` shape sidesteps chat-
template rendering entirely and worked correctly end to end (confirmed the
model correctly named colors/shapes in a synthetic canvas screenshot,
though see the design doc for where it also missed/hallucinated things --
this is a coarse signal, not a grounded one). Using the raw endpoint is
also simply appropriate here: this is a single fixed non-chat utility
prompt, not a multi-turn conversation, so there's nothing lost by skipping
the chat-template machinery.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

# SmolVLM/Idefics3-style special tokens. This is inherently model-specific --
# swap PLEIADES_VISION_FALLBACK_PROMPT if pointing this at a different tiny
# captioner with different chat markup.
_DEFAULT_PROMPT_TEMPLATE = (
    "<|im_start|>User:<__media__>Describe the most important thing a user "
    "could click or interact with on this screen, in ONE short factual "
    "sentence. If you are not sure, say so plainly.<end_of_utterance>\n"
    "Assistant:"
)


def _base_url() -> str:
    return os.environ.get("PLEIADES_VISION_FALLBACK_URL", "").strip().rstrip("/")


def is_configured() -> bool:
    """True when a fallback captioning server is configured via env var."""
    return bool(_base_url())


def caption_screenshot(png_bytes: bytes, *, timeout: float = 8.0) -> "str | None":
    """Ask the configured thin vision-fallback model to describe a
    screenshot in plain text.

    Returns None -- and never raises -- if the fallback isn't configured,
    is unreachable, times out, or responds with anything unusable. This is
    an optional descriptive signal layered on top of Phase 1's
    accessibility-tree extraction, not a dependency any caller can assume
    is present.
    """
    base_url = _base_url()
    if not base_url or not png_bytes:
        return None

    prompt_template = os.environ.get("PLEIADES_VISION_FALLBACK_PROMPT", _DEFAULT_PROMPT_TEMPLATE)
    body = {
        "prompt": {
            "prompt_string": prompt_template,
            "multimodal_data": [base64.b64encode(png_bytes).decode("ascii")],
        },
        "max_tokens": 40,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        f"{base_url}/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    text = (data.get("content") or "").strip()
    return text or None
