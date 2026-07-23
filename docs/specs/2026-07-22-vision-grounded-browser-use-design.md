# Vision-grounded browser-use — design & phased plan

**Status:** proposed, Phase 1 in progress. **Origin:** Fleagle, 2026-07-22 —
reported "the model doesn't have much coherency in browser-use... I don't
think vision models are seeing the browser and clicking based on their
vision, we need a better browser interaction system more native to each
model." Brainstormed live (not council-first this time — verified against
real source first, then council, then a real product comparison) before any
code was written.

## Where browser-use actually stands today (verified against source, not
assumed)

`pleiades/tools/browser.py`'s `BrowserTool` is the one production chat-path
browser tool (webui/desktop chat AND Discord/`pleiades work` both go through
it, via a runtime feature-detect between two backends — the desktop panel's
live Playwright session in `webui/browser_view.py`, and `harness/builtins/
browser.py`'s Camoufox singleton for callers with no panel event loop). Both
backends are **100% CSS-selector/text based** today: click by selector or
visible text, type by selector, read page text, and an existing `_hint()`
helper that lists clickable text / input-field selectors back to the model
when a click/fill fails, so it can retry with something real instead of
guessing again. `screenshot` exists but only saves a PNG to disk for a human
to look at — **it is never fed back to the model.** There is no vision in
the loop anywhere today, for any character, regardless of what model they're
running. This reframes the report: it isn't "vision clicking is inaccurate,"
it's "there's no vision-grounded interaction at all yet."

Separately confirmed: `pleiades/hardware.py::is_mmproj()` already exists, but
only to *exclude* mmproj sidecar files from the Foundry's "pick a model"
list — nothing anywhere pairs an mmproj file with its base GGUF or passes
`--mmproj`/`--jinja` to `launch.py`. A vision-capable model cannot actually
be *served* with vision today even if one were downloaded.

## Approach: accessibility-tree first, vision as escalation (not vision as
default)

Fleagle asked directly how Anthropic's own Claude in Chrome does this, since
it's a real, shipped answer to the same problem. Checked (tool schemas +
public docs, not assumed): its primary mode is the **accessibility tree**
(the same semantic structure screen readers use — roles, labels, ARIA
attributes, enabled/disabled state), read via Chrome's debugger API; a
`find` tool matches elements by natural-language description against that
tree; `read_page` returns it with stable element refs; clicks mostly
resolve by ref. Screenshots/raw-coordinate clicking (`computer` tool) exist
but are the *fallback* path, used only when the accessibility tree can't
represent the target (canvas, custom-rendered widgets) — not the default
mechanism. Sources: claudechrome.com/blog/how-claude-chrome-works,
usecarly.com/blog/what-is-claude-in-chrome, and the actual
`mcp__claude-in-chrome__*` tool schemas available in this environment.

Independently, qwen3.8-max (thinking) and Kimi were each asked (without
seeing each other's answer) whether a *small* local vision model (~4-8B
class — see model choice below, not a frontier VLM) should click by raw
coordinates or via a "set-of-marks" overlay (numbered boxes on DOM-extracted
interactive elements, model picks a number). Both converged independently
on set-of-marks + DOM grounding + a textual mark list alongside the image
(text understanding first, vision second — small VLMs are much stronger at
that combination) + a structured `{mark, confidence, reason}` response +
fallback to the plain selector tool on low confidence. Raw coordinates
demoted to a last-resort path for genuinely non-DOM content only.

These two independent answers (a real shipped product, and two models asked
the small-VLM-specific version of the same question) point the same
direction, so that's the design: **accessibility-tree extraction is the
primary, always-available mechanism (works for every model, no vision
required, covers the large majority of real interactions and is a strict
upgrade over today's CSS-selector guessing) — vision/set-of-marks is an
escalation path used only when the tree can't resolve the target or the
match is ambiguous, gated on the character's assigned model actually having
vision** — raw-coordinate clicking is reserved for canvas/WebGL/custom
widgets as the true last resort.

## Model choice for the vision tier

`HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive` (base: Google's
`gemma-4-e4b-it`, architecture tag `gemma4`) — natively multimodal
(text/image/video/audio), 4B active params, 131K context, quants from
4.2GB-7.6GB plus a 945MB f16 mmproj sidecar. Comfortably fits the 2080 Ti's
11GB VRAM even at a high quant with room for context. Requires `--mmproj
<path> --jinja` on llama-server. Chosen by Fleagle directly, verified real
against the live HF page before committing to it in this design.

## Phased plan

**Phase 1 (in progress this session):** accessibility-tree extraction +
stable-ref-based click/type/read, added to `BrowserTool`'s panel backend
(`webui/browser_view.py`'s live Playwright session — the one actually in
use for desktop-app chat, which is where this bug was reported) as new
capability alongside the existing selector/text actions (not a
replacement — old behavior stays as-is, `ref` is an additional option).
Works with ANY model today, no vision required. The Camoufox/harness
backend (Discord, `pleiades work`) gets the same treatment as a fast-follow
once Phase 1 is proven on the panel backend, not blocking this phase.

**Phase 2 (not started):** Foundry/`ModelManager`/`launch.py` wiring to
recognize a GGUF+mmproj pair as one servable unit and launch it correctly
(`--mmproj` + `--jinja`); a way to mark a character's assigned model as
vision-capable so the escalation path knows whether it's available at all.

**Phase 3 (not started):** screenshot + set-of-marks overlay renderer —
draw numbered boxes over the Phase 1 accessibility-tree elements onto a
real screenshot, paired with the textual mark list.

**Phase 4 (not started):** vision-escalation logic in `BrowserTool` — try
Phase 1 ref-based resolution first; on failure/ambiguity AND if the
character's model is vision-capable (Phase 2), escalate to a vision call
with the Phase 3 overlay, parse `{mark, confidence, reason}`, map back to a
stable ref, dispatch. No vision-capable model assigned → clean fallback to
today's selector/text behavior, unchanged.

**Phase 5 (not started):** raw-coordinate last-resort path for genuinely
non-DOM content (canvas/WebGL/custom widgets), gated behind Phase 4's
escalation having already failed.

**Phase 6 (not started):** download the Gemma-4-E4B model via Foundry,
assign it to a test character, live end-to-end verification across all
three tiers on a real page.

## Addendum, 2026-07-23 — thin vision-fallback layer for text-only models

Fleagle raised a new idea to fold into this plan: instead of (or alongside)
requiring a character's *own* assigned model to be vision-capable for the
Phase 4 escalation tier, use a small, dedicated, always-separate
image-understanding layer that works on behalf of *any* model — so a
character running a text-only brain still gets some vision-fallback,
without needing Phase 2's Foundry/mmproj-pairing work done for their
specific model. This was investigated (real source, real local model, real
live test — not assumed) rather than designed on paper, per this doc's own
established practice.

### Three mechanisms considered

1. **A CLIP-style embedding model** that encodes the screenshot into a
   vector, to be compared/matched against *something*. Rejected outright:
   there is no corpus here to search. Phase 1's problem is "describe/locate
   what's on THIS page right now," not "find similar images from a
   collection" — embeddings are the right tool for retrieval, and there is
   no retrieval step in this pipeline. Adopting this would mean inventing a
   whole extra subsystem (an index, a matching threshold, a "what am I
   matching against" answer) to solve a problem embeddings don't actually
   address. Wrong mechanism for this job.
2. **A tiny captioning/VQA model** that describes the screenshot as plain
   text, for any text-only model to read. Model-agnostic, cheap, no new
   subsystem — just an HTTP call to a separately-run tiny server. This is
   what got built (see below).
3. **Doubling down on Phase 1's set-of-marks/text-list idea harder** (i.e.
   is there a real gap once you think it through, or does the
   accessibility tree already cover this?). Real gap confirmed: Phase 1
   only sees the DOM. A `<canvas>` element — a game board, a chart, a
   custom drawing surface — is one DOM node as far as accessibility
   extraction is concerned; nothing about what's *drawn inside it* is
   visible to `_panel_elements`. Same for content rendered via WebGL, or a
   custom widget library that never sets `role`/`aria-label` on anything
   users actually interact with. Phase 1 cannot see into these by
   construction, no matter how hard the DOM-side approach is pushed —this
   is a real, if narrow, residual case.

### What's actually available on this machine

`~/.pleiades/models/` (checked before starting): four GGUFs currently
downloaded (Ternary-Bonsai-4B, Qwen3.6-35B-A3B, Qwen2.5-0.5B-Instruct,
Huihui-Ornith-35B) — **none are vision-capable**, so there was no
"double-duty" existing download to reuse. Any thin fallback needs its own
new, small download.

### Model choice — and why much smaller than Phase 2's Gemma-4-E4B

Phase 2's `Gemma-4-E4B` (4B active params, 4.2–7.6GB quants) was chosen
because it needs to be a real, general-purpose *chat* model for a character
to reason with — vision is one of several things it needs to do well.
This fallback layer has exactly one job (describe a screenshot, briefly)
and needs to be cheap and fast above all, so a much smaller,
non-chat-capable model is the right choice: **`ggml-org/SmolVLM-256M-
Instruct-GGUF`** — 256M params, Q8_0 quant + mmproj sidecar totalling
**~280MB** (175MB model + 104MB mmproj), vs. multiple GB for Gemma-4-E4B.
Runs on the same already-installed llama-server binary
(`~/.local/llama-cpp/llama-b8429/llama-server --mmproj ... --jinja`) as any
other model on this box — no new runtime, no new dependency.

### Live verification (real headless Chromium, real llama-server, real model)

Downloaded the model, launched a standalone llama-server on a scratch port,
rendered a real `<canvas>` page with Playwright (blue background, white
circle, and a yellow "SUBMIT" button drawn via `fillRect`/`fillText` — i.e.
exactly the kind of content Phase 1's DOM extraction cannot see into), and
captioned the real screenshot end to end:

- **First finding (a real bug, not a design flaw):** the standard
  `/v1/chat/completions` endpoint 400s for this model on this build —
  `tokenize: error: number of bitmaps (1) does not match number of markers
  (0)`. Traced into `common/chat.cpp`: this build's chat-template renderer
  (a "differential autoparser" that probes the template with synthetic
  messages to auto-detect reasoning/tool-call format) collapses the
  message down to a plain `System:`/`User:`/`Assistant:` concatenation for
  this model and **drops the image content part entirely** — the media
  marker never reaches the tokenizer. The low-level `/completions`
  endpoint's `{"prompt": {"prompt_string": ..., "multimodal_data": [...]}}`
  shape sidesteps chat-template rendering completely and works correctly.
  `pleiades/tools/vision_caption.py` uses this endpoint for exactly this
  reason (documented in the module docstring) — appropriate anyway, since
  this is one fixed non-chat utility prompt, not a conversation.
- **Captioning result on the canvas screenshot:** *"The image is a
  rectangular blue background with a white circle in the center."* —
  correct as far as it goes, but **it missed the yellow SUBMIT button
  entirely** — the one element on the page a user would actually want to
  click. On a second synthetic screenshot (pink background, green
  rectangle, "PLAY GAME" text) with more `max_tokens` allowed, it correctly
  read the text "Play Game" but then **hallucinated invented page structure**
  ("Top Section", commentary on "clean, modern font") not present in the
  image at all.
- **Performance, once warm:** ~95ms prompt processing + ~450ms generation
  for a short caption (16 tokens) — comfortably cheap and fast, confirming
  the "thin" framing is achievable. (First cold-start call took ~16s,
  consistent with one-time model/kernel warmup, not per-call cost.)

### Honest verdict on marginal value

Phase 1 already covers the large majority of real pages, including many
that *look* canvas-heavy but still have real accessible controls around the
canvas (toolbars, buttons, form fields) that `_panel_elements` finds fine.
The genuinely uncovered residual — a canvas/WebGL surface where the
interactive content itself has no DOM representation — is real but narrow,
and **this thin captioning layer does not solve the actual hard part of
that residual case**. It cannot produce a click coordinate, a DOM ref, or
any other actionable target — it's demonstrated, on a synthetic page built
specifically to test this, that it can miss the one interactive element
that matters while confidently describing decorative background elements
instead, and it hallucinates invented detail when given room to elaborate.
A text-only model handed this caption is in almost the same position as
before: it knows roughly what kind of page it's looking at, but it still
cannot reliably click anything within a canvas that Phase 1 didn't already
find. Turning "there's a canvas with a game board" into an actual working
click still requires either the character's own real vision-capable model
with Phase 3's set-of-marks overlay (Phase 4, DOM-grounded and far more
reliable — the marks come from real elements, not a guess) or Phase 5's
raw-coordinate last resort. **This fallback's real value is descriptive,
not interactive**: it lets any model — vision-capable or not — get a rough
qualitative read of an otherwise-opaque page ("this looks like a drawing
app," "this looks like a blocked/CAPTCHA page," "this looks broken/blank")
well enough to decide on a *strategy* (retry, ask the user, try a different
approach), rather than blindly guessing. That is real but modest value,
clearly smaller than what Phases 2–5 deliver for actual clicking accuracy,
and it should not be oversold as solving the canvas/custom-widget
interaction problem.

### What was built (this session)

Given the above, a deliberately small, strictly additive, off-by-default
slice:

- **`pleiades/tools/vision_caption.py`** — a standalone module (stdlib
  only: `urllib.request`, no new dependency) with `is_configured()` and
  `caption_screenshot(png_bytes) -> str | None`. Reads the fallback
  server's URL from `PLEIADES_VISION_FALLBACK_URL` (unset = disabled, zero
  behavior change, zero network calls). Never raises — any failure
  (unreachable, timeout, malformed response, empty caption) returns `None`
  silently, since this is an optional signal, not a dependency.
  Deliberately **not** wired into `ModelManager`/Foundry the way Phase 2
  will wire the character's own vision model — that would overstate this
  slice's maturity; it's a raw env-var escape hatch pointing at a
  separately-run llama-server, not a first-class configured setting yet.
- **`BrowserTool`'s `elements` action** (`pleiades/tools/browser.py`) now
  calls this *only* when Phase 1's extraction returns an empty list (the
  cheapest available proxy for "this is probably canvas/custom-rendered"),
  and only appends output if a caption actually comes back — otherwise
  today's exact `"No interactive elements found on the current page."`
  message is unchanged, byte for byte. The appended text is explicitly
  labeled `"NOT grounded to any clickable element, purely descriptive"` so
  a model doesn't mistake it for something it can click by ref.
- **Tests:** `tests/test_vision_caption.py` (disabled-by-default/no-network
  behavior, base64 payload shape, timeout/malformed-response/empty-content
  handling, custom prompt template override) and additions to
  `tests/test_browser_tool.py` (fallback never triggers when elements are
  found; unconfigured/`None`-caption leaves the message byte-for-byte
  unchanged; configured + real caption gets appended with the "not
  grounded" disclaimer). 264 passed, 2 pre-existing unrelated failures
  (`test_models.py::test_start_relocates_occupied_port` — a port-race flake
  confirmed present on `main` before this change; `test_sandbox.py::
  test_run_sandboxed_reports_memory_kill` — pre-existing MemoryError-vs-
  SIGKILL flake, also confirmed present on `main`, unrelated to this
  change).

### What was deliberately skipped

- The CLIP-embedding + retrieval mechanism (wrong tool for this problem,
  see above).
- Any attempt to have the captioner emit or approximate click coordinates
  — the live test showed it isn't reliable enough for that (missed the one
  real button on a 3-shape test page), and pretending otherwise would give
  a false sense of grounding that Phase 4's real DOM-anchored set-of-marks
  approach actually has.
- Foundry/`ModelManager` integration for this fallback model — deliberately
  left as a raw env var. Wiring it in as a persisted, UI-configurable
  setting is a small follow-up if/when this is actually used in practice,
  but doing it now would overstate how proven this slice is.
- Extending the hook to the Camoufox/harness backend (Discord, `pleiades
  work`) — mirrors Phase 1's own scoping decision to prove this on the
  panel backend first.
