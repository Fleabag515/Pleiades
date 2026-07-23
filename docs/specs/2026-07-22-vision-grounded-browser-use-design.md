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
