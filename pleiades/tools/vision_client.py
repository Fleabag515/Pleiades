"""Shared vision-fallback client for the main chat path (docs/specs/2026-07-23-
vision-routing-design.md, step 5).

Distinct from vision_caption.py (browser-use's screenshot captioner) on
purpose -- see the "Why /v1/chat/completions, not the raw endpoint" note
below for the evidence that led to NOT reusing vision_caption.py's raw
/completions transport here, despite both modules solving a similar-shaped
problem (describe an image via a small, separately-run, on-demand vision
model). vision_caption.py itself is left untouched: its transport is
proven-in-place for its one real caller (browser.py's canvas/WebGL
fallback) and there is no forcing reason to touch working code.

Purpose here: when a character's assigned chat model is NOT vision-capable
(engine.Engine._model_vision_capable() returns False) but the user attached
an image anyway, this module asks a separate small vision-capable model to
describe the image in plain text, and that DESCRIPTION -- not the raw image
-- gets folded into the text-only model's prompt. The character never sees
pixels; it sees "the user attached an image: <description>".

Configuration (env-gated, on-demand-start -- same pattern as
vision_caption.py, NOT an always-resident model):
  PLEIADES_VISION_FALLBACK_URL   base URL of an already-running llama-server
                                 instance serving a small vision-capable
                                 model + its mmproj sidecar (e.g.
                                 `llama-server -m Qwen2.5-VL-3B-Instruct-
                                 Q4_K_M.gguf --mmproj mmproj-Qwen2.5-VL-3B-
                                 Instruct-Q8_0.gguf --jinja`). Same variable
                                 vision_caption.py reads -- the two modules
                                 are expected to point at the SAME captioner
                                 process in the common case (no reason to
                                 run two separate vision servers), each
                                 asking it a different question for a
                                 different caller.
  PLEIADES_VISION_FALLBACK_MODEL optional: the model id string to send in
                                 the request body. llama-server ignores it
                                 for local single-model serving; harmless to
                                 leave blank.

Why /v1/chat/completions, not the raw endpoint (evidence, not assumption):
vision_caption.py's docstring documents a real, LIVE-verified failure of
/v1/chat/completions for ggml-org/SmolVLM-256M-Instruct on this project's
pinned llama-server build (the chat-template renderer silently dropped the
image marker; mtmd logged "1 bitmap, 0 markers" then 400'd), which is why it
uses the raw `/completions` + `multimodal_data` shape instead. Two
follow-up live tests this session (2026-07-23, same pinned llama-server
build, real test images, real HTTP responses -- see
docs/specs/2026-07-23-vision-routing-design.md for full transcripts) found
the OPPOSITE for two newer/different models:
  - ggml-org/SmolVLM-500M-Instruct-GGUF: /v1/chat/completions correctly
    named specific colors/shapes in two different synthetic test images,
    both with and without --jinja.
  - ggml-org/Qwen2.5-VL-3B-Instruct-GGUF: /v1/chat/completions correctly
    and precisely described shapes, colors, AND rendered text in a test
    image, in 0.8s.
Conversely, the raw-endpoint marker convention vision_caption.py relies on
(`<__media__>` wrapped in SmolVLM/Idefics3-specific `<|im_start|>User:...
<end_of_utterance>` chat markup) does NOT generalize: sending that exact
prompt string to the Qwen2.5-VL-3B server above 400s with "Failed to
tokenize prompt" -- those special tokens aren't in Qwen's vocabulary. The
raw-endpoint shape sidesteps chat-template rendering entirely, which is
exactly why it needs a hand-written, model-specific marker in the first
place; it was never actually more "generalizable" than the chat-completions
path, just proven for one specific small model. Given a choice between
(a) auditing chat-template marker behavor per-model to keep the raw
transport working, or (b) using the SAME chat-completions transport the
rest of this codebase (engine.py) already speaks to every model, (b) is
simpler, is what's actually verified working for the two models being
considered as the real fallback captioner, and needs no per-model prompt
maintenance. vision_caption.py is not "wrong" -- it's solving for a
specific older model that this evidence doesn't cover either way, so it is
deliberately left as-is rather than "fixed" without re-verifying its one
real caller.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

_DEFAULT_QUESTION = (
    "Describe this image factually and specifically -- objects, colors, "
    "layout, and any visible text. Be concrete; do not guess at things "
    "you can't actually see."
)


def _base_url() -> str:
    return os.environ.get("PLEIADES_VISION_FALLBACK_URL", "").strip().rstrip("/")


def is_configured() -> bool:
    """True when a fallback vision-captioning server is configured via env var.

    Same PLEIADES_VISION_FALLBACK_URL vision_caption.py reads -- see this
    module's docstring for why sharing one env var is the right default.
    """
    return bool(_base_url())


def describe_image(image_bytes: bytes, question: str = _DEFAULT_QUESTION, *,
                   mime: str = "image/png", timeout: float = 20.0,
                   max_tokens: int = 220) -> "str | None":
    """Ask the configured fallback vision model to describe an image via the
    standard OpenAI /v1/chat/completions content-parts shape.

    Returns None -- and never raises -- if the fallback isn't configured,
    is unreachable, times out, or responds with anything unusable. Callers
    (e.g. engine.py, when the character's own assigned model isn't
    vision-capable) treat this purely as an optional enrichment: absent a
    usable description, the turn proceeds exactly as it would have before
    this module existed (attachment noted as present, no description).
    """
    base_url = _base_url()
    if not base_url or not image_bytes:
        return None

    data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    body = {
        "model": os.environ.get("PLEIADES_VISION_FALLBACK_MODEL", "") or "vision-fallback",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None
    try:
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError):
        return None
    return text or None
