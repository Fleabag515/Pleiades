# Genuinely removing the context window — design & phased plan

**Status:** proposed, not started. **Origin:** Fleagle, 2026-07-21 — "my goal is to
fully remove that context window entirely ... genuine real-time scalability, which
we've demoed before with a qwen model ... consult the council ... im looking for
something absolutely game changing." Council = qwen3.8-max (thinking mode) + a
fresh Opus subagent, independently briefed on the same facts below and asked to
weigh in. Both converged hard on the same shape; this doc is that synthesis.

**Relationship to [[2026-07-21-native-inference-engine-design.md]]:** that doc is
about the *server/engine layer* (how Pleiades talks to a model, how resize works).
This doc is about the *model layer* (what kind of memory a model's architecture
actually has). They compose: the native engine's context governor should read the
model-memory-profile this doc introduces (Phase 0 below) rather than assuming every
model is a dense transformer with linearly-growing KV.

## The prior system Fleagle remembered — found

His separate "New Brain" research project already built exactly this pattern for
his own character model, Mark: a frozen 4-bit-quantized **Qwen3.5-9B-Base**
backbone where 24 of 32 layers were transplanted to **Gated DeltaNet** (linear
attention — a real, fixed-size O(1) recurrent state per layer, independent of
sequence length; literally no growing KV cache for those layers), keeping 8
full-attention layers backed by a bounded Cascading-Cache. Live, measured, on his
actual RTX 2080 Ti. Required training only ~6.13M small adapter parameters against
the frozen ~9B backbone — not retraining the whole model. **This is "the qwen
model" — confirmed, not a false memory.** In the literature this general pattern is
called *linearization*.

## Phase 0 — Documentation discovery (done, findings below)

**llama.cpp already has native support for genuinely recurrent/hybrid models —
confirmed by reading the actual source** (`~/llamacpp-cuda/llama.cpp`,
`src/llama-arch.cpp`): `LLM_ARCH_MAMBA`, `MAMBA2`, `JAMBA` (hybrid), `RWKV6`,
`RWKV6QWEN2`, `RWKV7`, and critically **`LLM_ARCH_QWEN3NEXT`** (a Qwen-family
hybrid linear-attention model). Public API: `llama_model_is_recurrent()`,
`llama_model_is_hybrid()` (`include/llama.h:616,619`), and a per-layer
`llama_hparams::is_recurrent(il)` (`src/llama-hparams.h:298`). **llama.cpp's own
memory manager already distinguishes fixed-size recurrent-layer state from
growing attention-layer KV within one hybrid model — serving such a model
requires zero new engine code.**

**Real, current (2026) research/production landscape** (web-search confirmed):
production hybrid models already ship publicly — Jamba 1.5 Large (~398B/94B
active), RWKV-7 G1, Falcon Mamba, Codestral Mamba — mixing real attention layers
with Mamba/DeltaNet-style O(1) recurrent layers. Separately, "linearization" is
an active, real academic subfield: converting an *existing* pretrained
quadratic-attention checkpoint into a hybrid/recurrent one via cheap distillation
rather than full retraining. Named published methods: **LoLCATs** (attention-transfer
to a linear mixer per head, then a small LoRA touching ~0.2% of weights recovers
quality; reported ~1 day of training, scales to 70B-405B, ≤1% MMLU gap — *at
large-lab GPU scale, not a single 2080 Ti*), **MOHAWK** (3-phase distillation,
~3B tokens / ~0.1% of original pretraining compute), plus SUPRA, RADLADS, T2R,
"Mamba-in-Llama" as related precedent.

**Council verdict on sequencing (both consultants independently agreed):** do NOT
jump straight to "convert any downloaded model" (option C in the original brief —
generalizing New Brain's full bespoke approach). The real path is a truth-layer
fix first, then a curated free win, then a narrow surgical pipeline — not a
research bet shipped as a headline promise.

## Phase 0.A — Model memory-profile truth layer

**Goal:** stop Pleiades' hardware/context planner from treating every model as
"dense transformer, KV grows linearly with tokens." Add a per-model memory
profile, read from llama.cpp's own `llama_model_is_recurrent`/`is_hybrid` +
per-layer `is_recurrent()` at load time:

```
memory_arch: dense | recurrent | hybrid
attention_cache_policy: unbounded | bounded_window | cascading | sink_plus_window | none
recurrent_state_bytes_per_slot: int
bounded_attention_bytes_per_slot: int
max_effective_window: int | null
```

**Critical nuance from qwen's review, worth stating plainly:** "hybrid" does not
automatically mean "no context window." A hybrid model with a few full-attention
layers and an *unbounded* KV cache still has a growing cache — just a smaller one
than a dense transformer. Only a hybrid with *bounded* attention (sliding-window,
sink, or cascading) is genuinely O(1) end to end. The profile above has to capture
that distinction, not just "has some recurrent layers."

**VRAM math to fix in `pleiades/hardware.py`:**
```
dense:     VRAM ≈ weights + activations + KV(seq_len)          # today's only model
recurrent: VRAM ≈ weights + activations + recurrent_state(slots)
hybrid:    VRAM ≈ weights + activations + recurrent_state(slots) + bounded_attention_cache(slots)
```
Today's code estimates every model as the first case — it would badly
mis-plan a hybrid model (either reject a perfectly fine config or under-provision).

**Verification:** load a real hybrid GGUF (once one is downloaded per Phase 1),
run a 100k+ token synthetic stream, confirm VRAM stays flat after warmup and
there's no hard context-window cutoff.

**This phase is small (days, not weeks) and is a hard prerequisite for
everything else below, including consuming Phase 2's own output.**

## Phase 0.5 — Dense-model compatibility mode (StreamingLLM / attention sinks)

**Goal:** the cheapest available win, and both consultants said to sequence it
*before* the hybrid-model work: for Pleiades' existing ordinary dense models
(everything already downloaded), add attention-sink / sliding-window behavior so
a long conversation never hits a hard wall — near-zero training, no new
checkpoint, llama.cpp already has the underlying primitives. This is explicitly
**not** genuine O(1) memory — it's bounded-window-plus-eviction, with Anamnesis
doing the actual relevant-content recall. Frame it to Fleagle as the compatibility
layer, not the headline feature — the real "removes the window" story is Phases
1 and 2 below.

## Phase 1 — Ship the free win: curated hybrid/recurrent models

**Goal:** since llama.cpp already serves genuinely O(1)/bounded-hybrid
architectures with zero new engine work, add a Foundry catalog tag
(`memory_mode: recurrent | hybrid_bounded | dense`) and surface known-good
hybrid GGUFs sized for the RTX 2080 Ti (Qwen3Next-class small hybrids,
RWKV6Qwen2, Mamba2/Falcon-Mamba-class, small Jamba-class if VRAM allows).

**Verification:** the same 100k+ token flat-VRAM stream test from Phase 0.A,
run against a real downloaded hybrid checkpoint. **If this passes, Fleagle
already has the user-facing experience he's asking for** — "the conversation
does not hit a wall" — for any model built this way, with zero training
performed by Pleiades.

## Phase 2 — Build a narrow, offline "linearization foundry"

**Goal:** this is where Fleagle's own "skeleton + weight transplant" idea
becomes real — grounded in the published LoLCATs/MOHAWK recipes instead of
invented from scratch. Both consultants were explicit that this must be an
**offline conversion pipeline that runs once and caches its output**, not
runtime weight-dissection on every launch:

```
donor dense checkpoint → choose skeleton architecture (target one llama.cpp
already understands, e.g. the Qwen3Next/gated-DeltaNet family) → freeze donor
weights → replace selected attention layers with linear/recurrent blocks →
distill against the donor's own logits/hidden states → merge adapters →
export GGUF → validate in llama.cpp → register in the Foundry catalog
```

**Realistic scope, per both consultants — read this before promising anything
to Fleagle:**
- **Target dense checkpoints ≤~9B first.** LoLCATs/MOHAWK's "cheap" claims are
  cheap *relative to full pretraining on large-lab GPUs*, not cheap on a single
  11GB RTX 2080 Ti. 9B frozen-backbone + small adapters is plausible locally
  (matches New Brain's own precedent exactly). 14B is painful. **35B dense is not
  realistic to train locally on this card** — export to a rented cloud GPU for
  anything that size, bring the resulting GGUF back to serve locally.
- **Don't linearize MoE models first.** Mark's own live 35B-A3B models are MoE;
  published linearization work targets dense attention, and replacing attention
  layers can interact badly with an MoE router (collapse, expert underuse). Scope
  MoE linearization as a later, harder track — not where Phase 2 starts.
- **Don't linearize every layer blindly.** New Brain's own 24-transplanted /
  8-retained split is a strong prior, not an arbitrary choice — attention layers
  differ in importance (copying exact tokens, long-range reference resolution,
  persona/instruction adherence). Start hybrid (most layers linear, a handful of
  attention layers kept), not fully recurrent.
- **The remaining attention layers still need bounding.** If 24 layers go
  recurrent but 8 stay full-attention with a normal unbounded KV cache, the
  result is "much smaller KV," not "no KV." The retained attention layers need
  sliding-window/sink/cascading treatment too (reuse Phase 0.5's mechanism) for
  the end-to-end claim "no context window" to actually hold.
- **Target an architecture llama.cpp already implements.** If the output
  doesn't match an existing arch (`QWEN3NEXT`, `JAMBA`, etc.), Pleiades would need
  new GGUF metadata/tensor-naming/kernel work — turning "cheap linearization"
  into "fork and maintain a custom llama.cpp backend." That defeats the entire
  point of Phase 1 being free. Pick the skeleton to match what's already there.
- **Quantization risk is different for recurrent state than dense weights** —
  validate post-quantization (persona stability, long-stream coherence, no
  numeric blowup/decay in the gates), don't assume Q4/K is as safe as it is for
  ordinary MLP weights.
- **Base models before instruct models.** Published linearization results mostly
  validate on base-model benchmarks; a character/chat model's instruct behavior,
  tool use, and persona stability can degrade even when perplexity looks fine.
  Validate architecture on a base donor first, then test instruct donors with
  chat-distribution calibration data, evaluated with real conversational tests
  — not just MMLU/perplexity.
- **Recurrent state doesn't support cache-style rewind/branching.** Editing an
  earlier turn means restoring a prior state snapshot and replaying forward, not
  truncating a KV cache. This changes the engine/UX contract (checkpoint the
  recurrent state alongside the transcript at each turn) — not a blocker, but
  needs to be designed for, not discovered after the fact.

**Verification:** long-stream evaluation (100k+ tokens), not short-benchmark —
VRAM flatness, coherence over the full stream, and a real conversational eval
set, not just perplexity/MMLU.

## Phase 3 — Keep the full New Brain approach scoped to Mark

Both consultants independently said not to generalize New Brain's complete
bespoke approach (custom byte-level front/back end + from-scratch adapter
training) as the general Pleiades feature — it's the right call for a signature
character model where per-model training cost is acceptable, but it's more
expensive and more fragile than Phase 2's narrower recipe for the actual goal
here (letting *any* downloaded model shed its context window). Phase 2 is the
generalized, cheaper version of the same idea; Phase 3 stays New Brain's own
track, untouched.

## Anamnesis — does not change

Both consultants agreed on this independently, and it's worth stating plainly
so it doesn't get "simplified away" by mistake later: **Anamnesis's job stays
identical regardless of whether the backbone underneath is quadratic-attention
or recurrent.** Anamnesis decides *what content* a turn needs; the backbone
decides *how state is stored*. A genuinely O(1) recurrent backbone changes the
*cost curve* of a long conversation, not the *relevance problem* — recurrent/
linear layers have lossy, bounded state (they forget silently rather than
truncating explicitly), so Anamnesis's job of re-injecting the right salient
context matters *more*, not less. **Do not couple them** — "the model remembers
everything now, retire the memory proxy" would be the expensive mistake here.

## What "game-changing," honestly, should mean

In priority order, per the council: **Phase 0.A** (hybrid-aware planner, unlocks
everything else) → **Phase 0.5** (StreamingLLM on today's models, near-free,
de-risks the planner work) → **Phase 1** (Foundry serves a real hybrid GGUF,
genuine O(1), zero training — this is the fast, real, honest "no wall" win) →
**Phase 2** (the offline linearization foundry, scoped to ≤9B dense donors
first — the actual differentiator: *any* small-to-mid dense model downloaded
through Pleiades can be *made* recurrent). That's the real, defensible version
of "absolutely game-changing" — not the from-scratch skeleton-every-launch
idea, which both consultants independently said costs more for no extra
capability over Phase 2's offline-bake approach.

---
*Written 2026-07-21 following a qwen3.8-max + Opus council consult, per
Fleagle's direction. Companion to
[[2026-07-21-native-inference-engine-design.md]] — that doc's context governor
(Phase 1/2 there) should consume this doc's Phase 0.A model-memory-profile
rather than duplicate assumptions about KV growth.*
