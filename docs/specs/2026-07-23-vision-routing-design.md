# Vision routing (capability detection + `--mmproj` + content-parts) — design, live verification, honest status

**Status:** capability detection + native `--mmproj` wiring + content-parts
message building shipped and live-verified against real models on this
machine. Fallback captioner (step 5) shipped and live-verified. **Origin:**
a full council review (Fable + Opus, working independently, both checked
against real source) confirmed local multimodal vision serving does not
exist in Pleiades despite looking easy: `launch.py` never passed `--mmproj`
anywhere; `hardware.py`'s `is_mmproj()` existed only to *exclude* mmproj
sidecars from model registration; `engine.py`'s whole message-building path
treated `content` as a plain string everywhere. This work was scoped
deliberately narrow and verification-first, per that review's explicit
warning against assuming any of this "just works."

## What shipped

1. **Capability detection.** `models.Model` gained `mmproj: str = ""` and
   `capabilities: str = ""` fields plus a `vision` property.
   `ModelManager.add()` auto-detects a sibling `mmproj-*.gguf` next to the
   model file via `hardware.find_mmproj_sibling()` (new — the mirror image
   of the existing `is_mmproj()`, which still only ever *excludes*).
   `fetch_model()` now also downloads the largest mmproj file offered by an
   HF repo alongside the chosen model (it used to just never fetch it),
   so auto-detection has something real to find for models pulled through
   the foundry, not just manually-arranged ones. Cloud models
   (`openrouter:`/`ollama-cloud:`) get a manual override instead
   (`Profile.model_capabilities`, CSV, e.g. `"vision"`);
   `POST /api/profiles/{name}/cloud-model` best-effort auto-populates it
   from OpenRouter's own `architecture.modality` when the caller doesn't
   set it explicitly.

2. **`--mmproj` wiring.** `launch.py`'s `build_command()` takes an `mmproj`
   param, appended as `--mmproj <path>` ONLY in the native llama-server
   branch. Deliberately NOT wired into the `llama_cpp.server` python
   fallback or `PLEIADES_ENGINE=pleiades_native` — both council reviews
   agreed vision should stay scoped to native llama-server for now (neither
   other runtime has any multimodal support to hand it to). Covered by
   `tests/test_launch.py`.

3. **Content-parts message building**, gated on `Engine._model_vision_capable
   (profile)`. When attachments are present:
   - **vision-capable model:** the fresh user turn's content becomes
     `[{"type":"text",...}, {"type":"image_url","image_url":{"url":"data:
     ...;base64,..."}}]`.
   - **non-vision-capable model:** falls through to the new fallback
     captioner (step 5) instead of ever sending an image marker the model
     can't render.
   - **history tail** (`chats.recent_messages`, resent every turn):
     `Engine._degrade_history_attachments()` strips any past image content
     down to a `[attached image: <name>]` text placeholder — a past image
     is never re-sent as base64 on a later turn (would re-prefill
     1000+ image tokens per round on a model that may already be
     CPU-offloaded for MoE experts — a real latency regression).
   - `feature/attach-cache` had landed no commits when this work started
     (`git diff main..feature/attach-cache` was empty), so this was first
     built against the documented attachment shape
     (`{"mime","data_b64","name"}`) from this project's working notes. That
     branch landed a real commit (`63146c7`) mid-session; its real shape
     turned out to be `{"name","path","mime"}` — a real file already saved
     to `pleiades/attachments.py`'s per-chat cache directory, resolved via
     `resolve_attachments()`, NOT inline base64. `_build_user_content()` was
     updated (`_read_attachment_bytes()`) to read real bytes off `path`
     when present, falling back to `data_b64`/`data` for tests/simpler
     callers. Also added `_is_image_attachment()` (mime-prefix, falling back
     to a filename-extension guess) so non-image attachments (pdf, docx,
     audio — attach-cache's own `_attachments_note` path-injection already
     covers those) are left alone by the vision path entirely. The two
     branches are complementary by design, per attach-cache's own
     docstring: it deliberately does none of the "can a model see/hear
     this" work (that's this branch's job); it only answers "where is the
     file" for tool use. Not yet merged into one branch — both are
     independent feature branches off `main`; whoever merges them will hit
     a textual conflict in `stream_events()`'s signature and the
     `_base_messages()` call site (attach-cache added its own
     `attachments_note: Optional[str]` param there), which is expected and
     resolvable: BOTH parameters are needed side by side (`attachments_note`
     for the real-path text note, `attachments` for the vision content-parts
     array) since they solve different problems.

## Live verification (step 4) — real evidence, not assumed

**4b. Does `/v1/chat/completions` actually render `image_url` correctly on
this machine's pinned native llama-server build (`e3546c7`)?** Tested with
two REAL downloaded models, each with its real mmproj sidecar, `--mmproj`
wired exactly the way `launch.py` now wires it:

- `ggml-org/SmolVLM-500M-Instruct-GGUF` (Q8_0 + its Q8_0 mmproj): asked to
  describe two different synthetic test images. Correctly named the
  specific shapes/colors in both ("Blue rectangle and yellow triangle on a
  white background", correctly read the rendered word "PLEIADES" on a red
  circle). Verified both **with and without** `--jinja`.
- `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF` (Q4_K_M + its Q8_0 mmproj): asked
  to describe a third test image (blue square top-left, red circle
  bottom-right, "FOUNDRY" text bottom-center) — response named all three
  correctly, in 0.8s, via `/v1/chat/completions`.
- Re-ran the SECOND test through Pleiades' **own** `ModelManager.add()` +
  `ModelManager.start()` (not a manual CLI invocation) against the real
  `~/.pleiades` home: mmproj auto-detected correctly, the real native
  llama-server process came up with `--mmproj` in its actual argv, and a
  real chat-completion request against it returned the correct description.
  Test model deregistered/stopped afterward; nothing left registered.

**This contradicts vision_caption.py's own documented finding** (that
`/v1/chat/completions` silently drops the image marker for
`ggml-org/SmolVLM-256M-Instruct-GGUF` on this same pinned build). That
finding is not disputed — it's a different, smaller/older model, not
re-tested here — but it clearly does not generalize to the two models
tested above. `/v1/chat/completions` is real, verified working for image
input on this machine's current native llama-server build, at least for
these two models. `vision_caption.py` itself is left untouched (its
raw-`/completions` transport is proven for its own real caller and there
was no forcing reason to touch it).

**4a. Does an OpenAI content-parts array survive the Anamnesis proxy
unmangled?** Traced the real code (not assumed):
- `proxy.js` already defensively flattens array content for STORAGE
  (`extractContentText`, pre-existing) — never crashes on it.
- `selector.js`'s `select()` builds `recencyMsgs` as a direct slice of the
  incoming array's own message objects and spreads them untouched into the
  final array that gets forwarded upstream — an image content-parts message
  passes through completely unmangled. Confirmed with two new integration
  tests against the real `Selector` class (not a reimplementation).
- **Found and fixed a real bug** (silent, not a crash): `_est()`'s
  per-message token-budget estimate used raw `.length`, which for a string
  is the char count (correct) but for an ARRAY is the element count (~2) —
  a vision turn was costed at ~1 token instead of its real size, letting
  the budget math believe more room was free than truly was and risking an
  assembled request overflowing the upstream server's real context window.
  Fixed in `_estOne()` (real text length + a flat 600-token estimate per
  image part). 5 new tests; full Anamnesis suite (207 tests) + lint pass.
  Committed separately in the Anamnesis checkout (`~/.local/share/
  anamnesis`), not touching the live running daemon.

## Step 5 — fallback captioner for non-vision models

`pleiades/tools/vision_client.py` (new): env-gated
(`PLEIADES_VISION_FALLBACK_URL`, same var `vision_caption.py` already uses),
on-demand-start, not always-resident — matches `vision_caption.py`'s
pattern exactly. `describe_image()` sends the image via
`/v1/chat/completions` content-parts, **not** the raw `/completions` +
`multimodal_data` shape `vision_caption.py` uses. This is a deliberate,
evidence-based deviation from the original plan (which assumed the raw
shape was "what actually works"): the raw endpoint's marker convention
(`<__media__>` wrapped in SmolVLM/Idefics3-specific chat markup) does NOT
generalize — sending that exact prompt to the Qwen2.5-VL-3B server used
above returns a hard 400 ("Failed to tokenize prompt"), because those
special tokens aren't in Qwen's vocabulary. Given `/v1/chat/completions` is
independently verified working for the two real candidate models (see
above), reusing the SAME transport `engine.py` already speaks to every
other model is simpler and needs no per-model prompt-template maintenance.
Live-verified end to end: `describe_image()` against a real running
Qwen2.5-VL-3B-Instruct server returned a correct, specific description of a
real test image.

Model choice: **Qwen2.5-VL-3B-Instruct** (Q4_K_M ~1.9GB + Q8_0 mmproj
~0.85GB, ≈2.8GB VRAM total), over SmolVLM-500M. This machine (RTX 2080 Ti,
11GB) had 9.4GB free VRAM with nothing else loaded at test time
(`pleiades hw`); both council reviews called SmolVLM too weak for
describing something like an uploaded logo usefully, and the live test
output above (correctly identifying shapes, colors, AND rendered text) bore
that out qualitatively vs. SmolVLM-500M's shorter, less structured answers
in the same test. ~2.8GB is comfortable headroom next to this machine's
smaller daily-driver models but WILL contend for VRAM if run alongside the
35B MoE model with its ~9GB partial-offload footprint — this is a real,
inherent tradeoff of loading a second model on-demand on an 11GB card, not
a bug; the env-gated on-demand-start pattern means it's the operator's call
whether/when to have it running.

`Engine._build_user_content()` wires this in: when attachments exist but
the assigned model isn't vision-capable, each attachment gets described via
`vision_client.describe_image()` and folded into the text prompt as
`[attached image: <name>] <description>`; if the fallback isn't configured,
the model still learns an image was attached (just not what's in it)
rather than the attachment silently vanishing with no trace.

## Known gaps / explicitly NOT done this round

- No desktop-app/webui UI wiring for the new `mmproj`/`capabilities` fields
  (API + CLI support exists; the model-add/edit forms weren't touched).
  Out of this round's backend-focused scope.
- `feature/attach-cache` hadn't landed any code as of this writing — the
  actual chat endpoint (`POST /api/chats/{id}`) doesn't yet accept an
  `attachments` field; `engine.py`'s new parameters are ready to receive it
  the moment that branch wires a caller through.
- Fallback-captioner quality is still "small dedicated vision model," not
  the character's own brain — by design (a separate on-demand side channel,
  same as `vision_caption.py`), but worth knowing it's not the same
  fidelity as a genuinely vision-native character model.
- No typechecker is configured in this project (no mypy/pyright/ty in
  `pyproject.toml` or the venv) — verified correctness via the full pytest
  suite (406 tests, 1 pre-existing unrelated failure in `test_sandbox.py`
  confirmed present on `main` too) instead.

## Recommendation

Real vision routing (steps 1-4) is verified working end to end on real
hardware for two real models via the standard `/v1/chat/completions` path,
through Pleiades' own `ModelManager`/`launch.py` code (not just a manual
CLI invocation), and the Anamnesis proxy hop in between is verified
non-mangling (plus one real bug fixed). This is ready to ship as "real"
vision routing, not fallback-only — contrary to the pessimistic prior
assumption, live testing did not surface a blocking failure in either of
the two risk points the council flagged. The one open item is UI wiring
(cosmetic, not correctness) and integration with `feature/attach-cache`
once that branch actually exists.
