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

## REPRIORITIZATION (second council stress-test, 2026-07-21) — read this first

Fleagle asked, bluntly, after the first pass shipped: **"is this really the best
method?"** Ran it back through the council with deliberately adversarial framing
(don't confirm, find the hole). Both qwen3.8-max and a fresh Opus independently
landed on the same correction, and it changes the lead recommendation:

**The first pass led with the wrong headline.** Recognizing and serving
already-hybrid/recurrent GGUFs (Phase 0.A/Phase 1 below) is correct engineering
and worth keeping, but it's *opportunistic* — it only helps for whatever narrow
slice of models the open-source community happens to publish as Mamba/RWKV/
Jamba/Qwen3-Next-class, which is smaller and often weaker at chat/agentic work
than the mainstream dense/MoE models people actually want to run (Llama,
Mistral, DeepSeek, most Qwen variants). Selling that as "the" answer to
"remove the context window" oversells a bonus catalog as the main event.

**The corrected framing — and the actual best available method within "no
training, ever":** you don't remove the context window, you *demote it from a
wall to a bounded working cache*. Anamnesis is already the real unbounded
memory (disk-backed, no limit). The model's attention window doesn't need to
disappear — it needs to be sized, per turn, to exactly what Anamnesis decides
that turn needs, via [[2026-07-21-native-inference-engine-design.md]]'s
already-planned elastic resize. That's not O(1) at the attention-math level,
but it's **O(working-set) in practice**, and it covers essentially every model,
not just an exotic minority. This is what actually delivers "never hit an
artificial wall, never waste VRAM reserving unused context" — which is the
real, underlying thing Fleagle described wanting.

**Revised priority order (supersedes the original "what game-changing should
mean" section below, kept further down for history):**
1. **Elastic context engine + Anamnesis is the lead.** Already planned in the
   companion doc. This is the honest scalability story for the 95%-mainstream-
   model case: memory moves from "infinite context window" to "external memory
   (Anamnesis) + bounded working context (elastic KV)," controlled by product
   logic (what Anamnesis chooses to inject), not raw model context size.
2. **Instrument before building more.** Measure real turn sizes flowing through
   Anamnesis today. If turns really do stay in the 16-32k range Fleagle
   described earlier, the "hard wall" problem may already be nearly solved by
   elastic resize alone — check before prioritizing further engineering here.
3. **Prefix/prompt KV caching — already live, verified in real production
   logs (checked before writing this in, not assumed).** The council flagged
   this as high-value, expecting it to be missing. It isn't: native
   `llama-server`'s slot manager already does automatic longest-common-prefix
   (LCP) reuse across requests — confirmed directly in a real log,
   `~/.pleiades/logs/model-qwythos-9b-claude-mythos-5-1m.log`: cold prompts
   process at single-digit-to-low-hundreds tok/s, while requests matching an
   existing slot by LCP similarity (`selected slot by LCP similarity, sim_best
   = 0.48/0.39`) hit **1893-2370 tok/s prompt processing — a 20-300x speedup
   from reuse, already happening today, zero code changes**. Separately,
   llama-server also ships a RAM-backed idle-slot prompt cache (`--cache-ram`,
   default 8192 MiB when the flag is simply omitted — PR ggml-org/llama.cpp
   #16391) that Pleiades' `launch.py` doesn't override, so it's very likely
   also active at its default; not yet confirmed via a fresh startup log at
   sufficient verbosity, worth a quick check but not an engineering task.
   **Net: no work needed here — this win already shipped upstream and
   Pleiades already benefits from it by not getting in its way.**
4. **Dense-model fallback, upgraded and demoted to explicit safety net.** The
   original Phase 0.5 (below) proposed naive StreamingLLM/attention-sinks. Both
   consultants said a heavy-hitter/SnapKV/PyramidKV-class eviction policy is
   meaningfully smarter for near-zero extra engineering cost — evict by
   actual importance, not just position — while being explicit that this is
   *still* lossy and a crash-prevention safety net, not a feature to sell. Its
   real job is to almost never fire, because Anamnesis + elastic resize already
   bound what's live in the prompt.
5. **Hybrid/recurrent GGUF recognition — correct to keep, demoted to a bonus
   lane.** Phase 0.A (shipped) and Phase 1 (below) still ship as-is — don't
   reserve linear-model KV like it's a growing dense cache, and do surface
   genuinely O(1) models for users who specifically want that tradeoff and
   accept the smaller/weaker model pool. Just don't lead with it.

The rest of this document (Phases 0/0.A/0.5/1/2/3 below) is preserved as
written for history and because the underlying engineering is still correct —
read it with the reprioritization above as the actual lead, not these phases'
original framing.

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

## Phase 2 — out of scope for Pleiades (correction, 2026-07-21)

**This phase, as originally drafted below the line, does not belong in
Pleiades and should not be built here.** Fleagle corrected this directly:
Pleiades is an inference engine + connector — it downloads and serves models,
wires them into a character with a tool belt (the harness) and memory
(Anamnesis). **It does not train or convert models.** An offline linearization
pipeline (LoLCATs/MOHAWK-style distillation: freeze a donor, replace attention
layers, distill, export a new checkpoint) is genuine ML research/training work
with its own compute, data-pipeline, and evaluation needs — exactly the kind
of thing [[new-brain-project]] already does as its own separate research
effort for Mark specifically, not a feature of the serving/connector layer.
If Fleagle wants a generalized linearization pipeline built, that's a new,
separate project in the shape of New Brain — planned and scoped on its own,
not folded into Pleiades' repo or roadmap.

**What Pleiades actually does, and what "game-changing" honestly caps out at
for the inference-engine layer:** recognize and correctly serve whatever
already-hybrid/recurrent model a user downloads (Phase 0.A + Phase 1 above),
and give ordinary dense models a graceful long-conversation fallback (Phase
0.5). That is the complete, correctly-scoped feature set — real, shippable,
zero training, and squarely inside what an inference engine + connector
should own.

## Phase 3 — New Brain stays New Brain

[[new-brain-project]] is Fleagle's own separate research track and already
does real linearization-adjacent work (the Qwen3.5-9B-Base transplant for
Mark). It is not a Pleiades feature and isn't generalized by anything in this
doc. If a future, explicitly-scoped research project wants to build a general
linearization pipeline, it would follow New Brain's model — its own project,
its own compute budget, its own repo — not live inside Pleiades.

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

In priority order, scoped correctly to what an inference engine/connector
should own: **Phase 0.A** (hybrid-aware planner, unlocks everything else,
SHIPPED 2026-07-21 commit 8ec41ba) → **Phase 0.5** (StreamingLLM on today's
dense models, near-free) → **Phase 1** (Foundry surfaces and correctly serves
real hybrid/recurrent GGUFs already published on HF — genuine O(1), zero
training, zero conversion work by Pleiades). That's the real, defensible,
correctly-scoped version of "game-changing" for this project: *any*
already-hybrid model a user downloads runs without hitting a context wall,
today, with no training pipeline of any kind inside Pleiades. Building new
recurrent architectures from existing dense checkpoints (what the original
Phase 2 draft proposed) is real, valuable work — it just belongs in a
dedicated research project, not this one.

---
*Written 2026-07-21 following a qwen3.8-max + Opus council consult, per
Fleagle's direction. Companion to
[[2026-07-21-native-inference-engine-design.md]] — that doc's context governor
(Phase 1/2 there) should consume this doc's Phase 0.A model-memory-profile
rather than duplicate assumptions about KV growth.*
